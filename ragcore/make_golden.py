"""生成 L1/L2 评测用的 golden set（半自动，第 2 轮）。

设计原则（见 求职/rag/METRICS.md 第 6.5 节）：
  1. 判定单元是 chunk_id，不是页码；
  2. 反向出题：以"源块"为 gold 出题，源块天然是 gold；但答案可能散落多块，
     故额外用"数值事实"跨块搜索给出「候选扩展 gold」，交由人工确认（宁可标全）；
  3. 负样本必须机器校验"确实缺席"（absent_terms 不得出现在任何块中），
     否则会把「其实手册里有答案」的题误当负样本，污染拒答率；
  4. 生成后自动体检：evidence_quote 是否为源块原文、参考答案数值是否落在源块、
     问题是否近似重复 —— 全部写进人工校核清单，不静默通过。

产物：
  data/eval/golden.jsonl        正式数据集
  data/eval/golden_review.md    人工校核清单（含自动告警）
  data/eval/stats_golden.json   生成统计

用法：
  python -m ragcore.make_golden --dry-run    # 只做选题与负样本校验，不调 LLM
  python -m ragcore.make_golden              # 完整生成
  python -m ragcore.make_golden --refresh    # 不调 LLM：重算候选 gold + 重写校核清单
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

from .config import (
    EVAL_DIR,
    GOLDEN_JSONL,
    GOLDEN_REVIEW_MD,
    STATS_GOLDEN_JSON,
)
from .llm import env_config, get_client

# ---------------------------------------------------------------- 选题配置

# 每章取几个源块出题（按章体量与重要性分配）
PER_CHAPTER = {
    "第一章 总则": 3,
    "第二章 员工行为准则": 3,
    "第三章 员工管理": 4,
    "第四章 考勤和休假": 6,
    "第五章 薪酬福利和绩效管理": 4,
    "第六章 员工沟通": 2,
    "第七章 人才发展与培训": 3,
    "第八章 奖惩规范": 5,
    "第九章 网络信息管理": 3,
}

# 数值事实：用于「候选扩展 gold」的跨块搜索（数值最具辨识度）
NUM_FACT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:个工作日|个自然日|个自然月|个月|天|日|年|元|%|小时|分钟|次|级|人)"
)

# 候选扩展 gold 的门槛：至少 2 个不同数值事实同时命中，才认为是"可能相关"
# （门槛=1 时噪声极大：像 10 / 30 这种数字几乎每块都有）
CAND_MIN_HITS = 2

# 近似重复判定阈值（问题二元组 Jaccard）
DUP_THRESHOLD = 0.55

# 注意：缺席词必须是"答案特征词"，不能只是话题词。
#   反例1「年会」→ 会命中「每【年会】根据市场…」的子串（假阳性）
#   反例2「公积金」→ 手册确实提到缴纳公积金，但**没有**缴存比例，故缺席词应为「缴存比例」
NEGATIVES: list[tuple[str, list[str]]] = [
    ("京东集团 2026 年的营收目标是多少？", ["营收目标", "收入目标"]),
    ("员工股票期权的行权价格是多少？", ["行权价", "行权价格"]),
    ("京东集团目前的市值是多少？", ["市值"]),
    ("员工每年可以报销多少额度的健身费用？", ["健身"]),
    ("公司是否举办年会？通常在哪里举办？", ["公司年会", "年度晚会", "团拜会"]),
    ("员工持股计划的锁定期是多长时间？", ["持股计划"]),
    ("公司对远程办公的天数和审批流程是如何规定的？", ["远程办公", "居家办公"]),
    ("员工子女教育补贴的标准是多少？", ["子女教育", "教育补贴"]),
    ("公司内部转岗需要经过几轮面试？", ["转岗"]),
    ("员工的住房公积金缴存比例是多少？", ["缴存比例"]),
    ("公司对员工使用生成式 AI 工具有什么规定？", ["生成式", "人工智能工具"]),
]


def ensure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def _norm(s: str) -> str:
    """去空白，便于跨块匹配（语料中数字与单位间常有空格）。"""
    return re.sub(r"\s+", "", s or "")


def load_chunks() -> list[dict]:
    from .config import CHUNKS_JSONL

    with CHUNKS_JSONL.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------- 选题

def _is_junk(c: dict) -> bool:
    """排除不适合出题的块：无章节归属、过短。"""
    if not c.get("path"):
        return True
    if c.get("char_len", 0) < 100:
        return True
    return False


def _value_score(c: dict) -> float:
    """源块价值：表格 > 有条款号 > 数值密集 > 长度适中。"""
    s = 0.0
    if c["content"].lstrip().startswith("|"):
        s += 5.0
    if c.get("clause_range"):
        s += 2.0
    s += min(len(NUM_FACT.findall(c["content"])), 5)
    if 200 <= c.get("char_len", 0) <= 700:
        s += 1.0
    return s


def select_source_chunks(chunks: list[dict]) -> list[dict]:
    by_chapter: dict[str, list[dict]] = {}
    for c in chunks:
        if _is_junk(c):
            continue
        by_chapter.setdefault(c["chapter"], []).append(c)

    picked: list[dict] = []
    for chapter, quota in PER_CHAPTER.items():
        pool = sorted(by_chapter.get(chapter, []), key=_value_score, reverse=True)
        picked.extend(pool[:quota])
    # 表格块是高危区（A-017 / A-035），必须覆盖
    picked_ids = {c["id"] for c in picked}
    for c in chunks:
        if c["content"].lstrip().startswith("|") and c["id"] not in picked_ids and not _is_junk(c):
            picked.append(c)
            picked_ids.add(c["id"])
    return picked


# ---------------------------------------------------------------- LLM 出题

SYS_PROMPT = """你是《京东集团员工手册》的评测数据标注助手。你只会看到手册中的一个片段。
请**仅依据该片段**出一道员工视角的检索评测题，并给出参考答案。

硬性规则：
1. 问题必须能**只**依据该片段回答；不得引入片段之外的任何知识。
2. 答案必须是片段事实的忠实转述，必须保留原文中的**具体数值、天数、比例、条件**。
3. 问题要自然、口语化，10~25 字，不要出现「本片段」「上述」「该条」等元表述，不要写条款编号。
4. 不要问「手册中如何规定X」这类，要问员工真正会问的业务问题。
5. evidence_quote 必须是片段中**一字不差**的一段连续原文（不得改写、不得拼接）。

只输出 JSON：
{"question": "...", "answer": "...", "evidence_quote": "片段中一字不差的原文"}"""


def _client():
    """复用 ragcore.llm 的客户端（含显式代理策略，见 llm._http_client）。"""
    cfg = env_config()
    if not (cfg["api_key"] and cfg["base_url"]):
        raise RuntimeError(".env 缺少 OPENAI_API_KEY / OPENAI_BASE_URL")
    return get_client(), cfg


def _parse_json(raw: str) -> dict:
    """容错解析：先整体 json.loads，失败则截取首尾花括号再试。"""
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        pass
    i, j = raw.find("{"), raw.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(raw[i : j + 1])
        except Exception:
            return {}
    return {}


def gen_qa(client, cfg, chunk: dict, attempts: int = 4) -> dict:
    """出题；对「空响应 / 缺字段」同样重试，不只重试异常。

    实测：本模型带 reasoning，JSON 模式下偶发返回空对象 {}（reasoning 吃掉预算），
    故第 3 次起退回纯文本再解析，并降温度。
    """
    user = "【手册片段】\n" + chunk["content"]
    last = None
    for attempt in range(attempts):
        try:
            kwargs: dict = {
                "model": cfg["model"],
                "messages": [
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.7 if attempt == 0 else 0.3,
                "max_tokens": 2000,
            }
            if attempt < 2:          # 前两次 JSON 模式，之后退回纯文本
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            qa = _parse_json(raw)
            if (qa.get("question") or "").strip() and (qa.get("answer") or "").strip():
                return qa
            last = "空/缺字段 raw=%r" % raw[:80]
        except Exception as exc:
            last = "%s: %s" % (type(exc).__name__, exc)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("出题失败: %s" % last)


# ---------------------------------------------------------------- 类型标注 & 候选 gold

def infer_type(chunk: dict) -> str:
    """类型由块属性决定（客观），不交给 LLM（避免抖动）。"""
    if chunk["content"].lstrip().startswith("|"):
        return "table"
    if len(NUM_FACT.findall(chunk["content"])) >= 3:
        return "numeric"
    return "fact"


def expand_gold_candidates(
    answer: str, chunks: list[dict], exclude_id: str, min_hits: int = CAND_MIN_HITS
) -> list[str]:
    """用答案里的数值事实跨块搜索，给出**候选**扩展 gold（需人工确认）。

    门槛：同一候选块至少命中 min_hits 个不同数值事实，否则视为巧合。
    """
    facts = {_norm(f) for f in NUM_FACT.findall(answer)}
    if len(facts) < min_hits:
        return []
    hits: list[str] = []
    for c in chunks:
        if c["id"] == exclude_id or _is_junk(c):
            continue
        body = _norm(c["content"])
        if sum(1 for f in facts if f in body) >= min_hits:
            hits.append(c["id"])
    return hits[:3]


# ---------------------------------------------------------------- 负样本缺席校验

def check_negatives(chunks: list[dict]) -> list[dict]:
    out = []
    for q, absent in NEGATIVES:
        found: list[str] = []
        for term in absent:
            t = _norm(term)
            hit = [c["id"] for c in chunks if t in _norm(c["content"])]
            if hit:
                found.append("%s→%s" % (term, hit[:3]))
        out.append({"question": q, "absent_terms": absent, "violations": found})
    return out


# ---------------------------------------------------------------- 自动体检

def _bigrams(s: str) -> set[str]:
    s = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", s or "")
    return {s[i : i + 2] for i in range(len(s) - 1)}


def verify_goldens(gold: list[dict], chunks: list[dict]) -> dict[str, list[str]]:
    """自动体检，返回 {qid: [告警...]}。全部告警进人工校核清单，不静默通过。"""
    byid = {c["id"]: c for c in chunks}
    flags: dict[str, list[str]] = {g["qid"]: [] for g in gold}

    for g in gold:
        if g["type"] == "negative":
            continue
        src = _norm(byid[g["source_chunk"]]["content"]) if g.get("source_chunk") in byid else ""
        if g.get("evidence_quote") and _norm(g["evidence_quote"]) not in src:
            flags[g["qid"]].append("evidence_quote 非源块原文子串（疑似转述/拼接）")
        miss = [f for f in {_norm(x) for x in NUM_FACT.findall(g["reference_answer"])} if f not in src]
        if miss:
            flags[g["qid"]].append("参考答案数值未在源块找到: " + "/".join(miss))

    for i in range(len(gold)):
        for j in range(i + 1, len(gold)):
            a, b = _bigrams(gold[i]["question"]), _bigrams(gold[j]["question"])
            if not a or not b:
                continue
            jac = len(a & b) / len(a | b)
            if jac >= DUP_THRESHOLD:
                flags[gold[i]["qid"]].append("与 %s 问题高度相似 (%.2f)" % (gold[j]["qid"], jac))
                flags[gold[j]["qid"]].append("与 %s 问题高度相似 (%.2f)" % (gold[i]["qid"], jac))
    return flags


# ---------------------------------------------------------------- 写校核清单

def write_review(gold: list[dict], chunks: list[dict], negs: list[dict], model: str) -> dict:
    flags = verify_goldens(gold, chunks)
    nonzero = [g for g in gold if g["type"] != "negative"]
    negs_in = [g for g in gold if g["type"] == "negative"]
    tcount: dict[str, int] = {}
    for g in gold:
        tcount[g["type"]] = tcount.get(g["type"], 0) + 1

    lines = [
        "# golden set 人工校核清单",
        "",
        "> 共 %d 条：正样本 %d · 负样本 %d（模型 %s）" % (len(gold), len(nonzero), len(negs_in), model),
        "> **已人工校核：%d / %d**" % (sum(1 for g in gold if g.get("reviewed")), len(gold)),
        "> 类型分布：" + str(tcount),
        "",
        "## 校核要点",
        "",
        "1. 问题是否**只靠 gold 块**就能回答（若还需要别处，把该块并入 gold_chunk_ids）；",
        "2. gold_candidates 是机器按「数值事实共现」给出的**候选**，不是结论——确认后并入；",
        "3. 负样本是否真的手册无答案；",
        "4. 处理完把该条 reviewed 改为 true。",
        "",
        "> 自动体检只覆盖「可机器判定」的部分（原文子串、数值落地、问题重复）。",
        "> 语义正确性无法自动判定，必须人工过一遍。",
        "",
        "---",
        "",
    ]

    for g in gold:
        mark = "✅" if g.get("reviewed") else "⬜"
        lines.append("## %s %s · %s · %s" % (mark, g["qid"], g["type"], g["question"]))
        lines.append("")
        for f in flags.get(g["qid"], []):
            lines.append("- ⚠️ **自动告警**: " + f)
        if g.get("review_note"):
            lines.append("- **人工判定**: " + g["review_note"])
        if g.get("negative_kind"):
            lines.append("- **负样本类型**: %s（话题相关但无答案，比普通负样本更有区分度）" % g["negative_kind"])
        gids = ", ".join(g["gold_chunk_ids"]) or "(负样本，空集)"
        lines.append("- **gold_chunk_ids**: " + gids)
        lines.append("- **gold_pages**: %s" % (g["gold_pages"] or "-"))
        lines.append("- **source_path**: %s" % (g["source_path"] or "-"))
        if g.get("related_pages"):
            lines.append("- **相关页**: %s" % g["related_pages"])
            lines.append("- **相关块**: %s" % (", ".join(g.get("related_chunk_ids", [])) or "-"))
            lines.append("- **相关 source_path**: %s" % (g.get("related_path") or "-"))
        if g.get("closest_content"):
            lines.append("- **最接近的内容（不构成答案）**: %s" % g["closest_content"])
        if g.get("gold_candidates"):
            lines.append("- **候选扩展 gold（机器建议，未采纳）**: " + ", ".join(g["gold_candidates"]))
        lines.append("- **参考答案**: %s" % g["reference_answer"])
        if g.get("evidence_quote"):
            lines.append("- **原文依据**: %s" % g["evidence_quote"])
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 负样本缺席校验（机器判定）")
    lines.append("")
    lines.append("| 问题 | 必须缺席词 | 校验 |")
    lines.append("|---|---|---|")
    for n in negs:
        mark = "✅缺席" if not n["violations"] else "⚠️命中 " + str(n["violations"])
        lines.append("| %s | %s | %s |" % (n["question"], "/".join(n["absent_terms"]), mark))
    lines.append("")

    GOLDEN_REVIEW_MD.write_text("\n".join(lines), encoding="utf-8")
    return {"type_dist": tcount, "flags": {k: v for k, v in flags.items() if v}}


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ensure_utf8()
    ap = argparse.ArgumentParser(description="生成评测 golden set")
    ap.add_argument("--dry-run", action="store_true", help="只选题与校验负样本，不调用 LLM")
    ap.add_argument("--refresh", action="store_true",
                    help="不调用 LLM：重算候选 gold 与自动体检，重写 golden.jsonl 与校核清单")
    args = ap.parse_args()

    chunks = load_chunks()
    print("[1/5] 载入 %d 个分块" % len(chunks))
    negs = check_negatives(chunks)
    ok_neg = [n for n in negs if not n["violations"]]
    bad_neg = [n for n in negs if n["violations"]]

    # -------- refresh：复用已有 golden.jsonl，不调 LLM --------
    if args.refresh:
        gold = [json.loads(l) for l in GOLDEN_JSONL.open(encoding="utf-8") if l.strip()]
        print("[refresh] 复用 %d 条已有标注，重算候选 gold 与自动体检" % len(gold))
        byid = {c["id"]: c for c in chunks}
        for g in gold:
            if g["type"] == "negative" or not g.get("source_chunk"):
                g["gold_candidates"] = []
                continue
            if g.get("reviewed"):
                continue          # 尊重人工判定：已校核条目不再被机器候选覆盖
            g["gold_candidates"] = expand_gold_candidates(
                g["reference_answer"], chunks, g["source_chunk"])
        with GOLDEN_JSONL.open("w", encoding="utf-8") as fh:
            for g in gold:
                fh.write(json.dumps(g, ensure_ascii=False) + "\n")
        summary = write_review(gold, chunks, negs, model="(refresh)")
        withflag = len(summary["flags"])
        print("  候选扩展条目: %d / %d" % (
            sum(1 for g in gold if g.get("gold_candidates")), len(gold)))
        print("  有自动告警的条目: %d" % withflag)
        print("  校核清单 %s" % GOLDEN_REVIEW_MD)
        return 0

    # -------- 正常流程 --------
    picked = select_source_chunks(chunks)
    print("[2/5] 选中 %d 个源块出题" % len(picked))
    for c in picked:
        print("      %s  %-7s  p%-6s %s" % (
            c["id"], infer_type(c), ",".join(map(str, c["pages"])), c["path"][:52]))

    print("\n[3/5] 负样本缺席校验：%d/%d 通过" % (len(ok_neg), len(negs)))
    for n in bad_neg:
        print("      ⚠ 需复核: %s  命中: %s" % (n["question"], n["violations"]))

    if args.dry_run:
        print("\n--dry-run：未调用 LLM，未写文件。")
        return 0

    client, cfg = _client()
    print("\n[4/5] 调用 %s 出题（%d 个源块）..." % (cfg["model"], len(picked)))
    goldens: list[dict] = []
    failures: list[str] = []
    for i, c in enumerate(picked, 1):
        try:
            qa = gen_qa(client, cfg, c)
            question = (qa.get("question") or "").strip()
            answer = (qa.get("answer") or "").strip()
            quote = (qa.get("evidence_quote") or "").strip()
            if not question or not answer:
                failures.append("%s: 字段缺失 %s" % (c["id"], qa))
                continue
            goldens.append({
                "qid": "P%02d" % (len(goldens) + 1),
                "question": question,
                "gold_chunk_ids": [c["id"]],
                "gold_candidates": expand_gold_candidates(answer, chunks, c["id"]),
                "gold_pages": c["pages"],
                "reference_answer": answer,
                "evidence_quote": quote,
                "type": infer_type(c),
                "source_chunk": c["id"],
                "source_path": c["path"],
                "reviewed": False,
            })
            print("      [%d/%d] %s" % (i, len(picked), question))
        except Exception as exc:
            failures.append("%s: %s" % (c["id"], exc))
            print("      [%d/%d] 失败 %s: %s" % (i, len(picked), c["id"], exc))

    for j, n in enumerate(ok_neg, 1):
        goldens.append({
            "qid": "N%02d" % j,
            "question": n["question"],
            "gold_chunk_ids": [],
            "gold_candidates": [],
            "gold_pages": [],
            "reference_answer": "手册未规定相关内容（负样本：应回答「未查到」）",
            "evidence_quote": "",
            "type": "negative",
            "source_chunk": "",
            "source_path": "",
            "reviewed": False,
        })

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with GOLDEN_JSONL.open("w", encoding="utf-8") as fh:
        for g in goldens:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")

    summary = write_review(goldens, chunks, negs, model=cfg["model"])
    nonzero = [g for g in goldens if g["type"] != "negative"]

    stats = {
        "step": "make_golden",
        "total": len(goldens),
        "positive": len(nonzero),
        "negative": len(ok_neg),
        "type_dist": summary["type_dist"],
        "source_chunks": len(picked),
        "negative_check_pass": len(ok_neg),
        "negative_check_fail": len(bad_neg),
        "auto_flags": summary["flags"],
        "llm_failures": failures,
        "model": cfg["model"],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    STATS_GOLDEN_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[5/5] 写出 %s" % GOLDEN_JSONL)
    print("      共 %d 条（正 %d / 负 %d）失败 %d" % (
        len(goldens), len(nonzero), len(ok_neg), len(failures)))
    print("      类型分布 %s" % summary["type_dist"])
    print("      候选扩展 gold: %d 条 | 有自动告警: %d 条" % (
        sum(1 for g in goldens if g.get("gold_candidates")), len(summary["flags"])))
    print("      校核清单 %s" % GOLDEN_REVIEW_MD)
    for f in failures:
        print("        " + f)
    print("-" * 62)
    print("⚠ 下一步：人工校核 golden_review.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
