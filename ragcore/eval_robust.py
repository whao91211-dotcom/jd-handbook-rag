"""生产可用性 · 输入稳健性与边界测试（确定性，零 LLM 成本）。

为什么单独立一层：
  L1 指标只测「标准问法」下的排序质量。真实员工会打错字、加口语、写得很长、
  中英混杂、只写半句。本层度量的是**输入扰动下的指标衰减**与**边界输入是否崩溃**。

两类测试：
  1) 机械扰动：对每个 golden 问题生成 8 种确定性变体，重跑检索，测 Recall@5 / MRR 衰减
  2) 边界输入：空串 / 空白 / 单字符 / 超长 / emoji / 控制字符 / HTML / 注入式，检查是否崩溃
  3) 同义改写（可选，需 LLM）：让模型在不复用原句词面的前提下改写问题，
     用来回答「BM25 的优势是不是只来自问题与 gold 块的字面重合」

用法：
  python -m ragcore.eval_robust                     # 机械扰动 + 边界
  python -m ragcore.eval_robust --modes rrf,bm25
  python -m ragcore.eval_robust --paraphrase        # 追加 LLM 同义改写（需 API）
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime

from .config import EVAL_DIR, GOLDEN_JSONL, RETRIEVE_MODES
from .retriever import retrieve
from .eval_ir import recall_at_k, reciprocal_rank, ndcg_at_k

REPORT_RB_MD = EVAL_DIR / "robust_report.md"
REPORT_RB_JSON = EVAL_DIR / "robust_report.json"

FW = str.maketrans("0123456789", "０１２３４５６７８９")


def ensure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def load_golden() -> list[dict]:
    with GOLDEN_JSONL.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ------------------------------------------------------------------ 机械扰动

PUNCT = "？?！!，,。.、；;：:（）()"

# 真实员工的输入习惯（确定性、可复现）
def make_variants(q: str, seed: int = 42) -> dict[str, str]:
    rnd = random.Random(seed)
    v: dict[str, str] = {}
    v["orig"] = q
    v["punct_strip"] = re.sub("[" + PUNCT + "]", "", q)
    v["spacey"] = re.sub(r"(\d)([^\d\s])", r"\1 \2", q)
    v["fullwidth_digit"] = q.translate(FW)
    v["colloquial"] = "请问一下，" + q
    v["verbose"] = "我是刚入职不久的新同事，对公司制度还不太熟悉，想请教一个问题：" + q
    v["noise_prefix"] = "顺便再问个事，" + q + "，谢谢！"
    v["english_mix"] = "Hi，想问下 " + q
    # 错别字：随机删掉一个汉字（不改动关键数字）
    han = [i for i, ch in enumerate(q) if "\u4e00" <= ch <= "\u9fff"]
    if len(han) > 4:
        v["typo_drop"] = q[: han[2]] + q[han[2] + 1 :]
    # 半句截断：只留前半（模拟员工只打了一半）
    v["truncated"] = q[: max(4, len(q) // 2)]
    return v


BOUNDARY_INPUTS = [
    ("empty", ""),
    ("whitespace", "   \t\n  "),
    ("single_char", "假"),
    ("single_punct", "？"),
    ("emoji", "😀😀😀 请假？？？ 🎉"),
    ("control_chars", "请假\x00\x01\x02申请"),
    ("html", "<script>alert(1)</script> 事假要提前几天"),
    ("sql_like", "'; DROP TABLE chunks; -- 事假"),
    ("very_long", "请假" * 2000),
    ("newlines", "事假\n\n\n\n提前几天\n申请"),
    ("only_digits", "1234567890"),
    ("mixed_script", "ﾊﾞｲﾄ 事假 ﾔｽﾐ"),
]


def stable_marker(text: str) -> str:
    r"""边界输入只检查「是否抛出异常 / 是否返回」，不检查语义。"""
    return text


def run_perturbation(golden: list[dict], modes: list[str]) -> dict:
    pos = [g for g in golden if g["type"] != "negative" and g["gold_chunk_ids"]]
    labels = list(make_variants(pos[0]["question"]).keys())
    out: dict[str, dict] = {}
    for mode in modes:
        per_label: dict[str, list] = {l: [] for l in labels}
        for g in pos:
            rel = set(g["gold_chunk_ids"])
            for label, qv in make_variants(g["question"]).items():
                try:
                    ids = [h["id"] for h in retrieve(qv, k=20, mode=mode)] if qv.strip() else []
                except Exception as exc:
                    ids = []
                    per_label[label].append({"qid": g["qid"], "error": "%s: %s" % (type(exc).__name__, exc)})
                    continue
                per_label[label].append({
                    "qid": g["qid"],
                    "recall5": recall_at_k(rel, ids, 5),
                    "mrr": reciprocal_rank(rel, ids),
                    "ndcg5": ndcg_at_k(rel, ids, 5),
                })
        rows = {}
        for label, items in per_label.items():
            ok = [x for x in items if "error" not in x]
            n = len(ok)
            rows[label] = {
                "n": n,
                "recall@5": round(sum(x["recall5"] for x in ok) / n, 4) if n else None,
                "mrr": round(sum(x["mrr"] for x in ok) / n, 4) if n else None,
                "ndcg@5": round(sum(x["ndcg5"] for x in ok) / n, 4) if n else None,
                "errors": len(items) - n,
            }
        out[mode] = rows
    return out


def run_boundary(modes: list[str]) -> list[dict]:
    res = []
    for name, text in BOUNDARY_INPUTS:
        for mode in modes:
            try:
                hits = retrieve(text, k=8, mode=mode)
                res.append({"case": name, "mode": mode, "ok": True,
                            "n_returned": len(hits), "top1": hits[0]["id"] if hits else None})
            except Exception as exc:
                res.append({"case": name, "mode": mode, "ok": False,
                            "error": "%s: %s" % (type(exc).__name__, exc)})
    return res


# ------------------------------------------------------------------ LLM 同义改写

PARA_SYS = """你在为检索系统做稳健性测试。请把用户问题**改写**成同一意图但**用词完全不同**的问法。
硬性要求：
1. 不得复用原问题中的关键名词（如"事假""年假""加班费"），必须换成同义的口语说法；
2. 保持意图完全一致，不得增加或改变任何条件；
3. 长度 10~30 字，像普通员工随口问的。
只输出 JSON：{"paraphrase": "..."}"""


def run_paraphrase(golden: list[dict], modes: list[str]) -> dict:
    from .llm import env_config, get_client
    client, cfg = get_client(), env_config()
    pos = [g for g in golden if g["type"] != "negative" and g["gold_chunk_ids"]]
    items = []
    for g in pos:
        q2 = None
        for _ in range(3):
            try:
                r = client.chat.completions.create(
                    model=cfg["model"],
                    messages=[{"role": "system", "content": PARA_SYS},
                              {"role": "user", "content": g["question"]}],
                    temperature=0.8, max_tokens=1200,
                    response_format={"type": "json_object"},
                )
                q2 = json.loads(r.choices[0].message.content or "{}").get("paraphrase")
                if q2:
                    break
            except Exception:
                pass
        items.append({"qid": g["qid"], "orig": g["question"], "para": q2,
                      "gold": g["gold_chunk_ids"]})
    out = {"items": items, "metrics": {}}
    for mode in modes:
        rows = []
        for it in items:
            if not it["para"]:
                continue
            rel = set(it["gold"])
            ids = [h["id"] for h in retrieve(it["para"], k=20, mode=mode)]
            rows.append({"recall5": recall_at_k(rel, ids, 5), "mrr": reciprocal_rank(rel, ids),
                         "ndcg5": ndcg_at_k(rel, ids, 5)})
        n = len(rows)
        out["metrics"][mode] = {
            "n": n,
            "recall@5": round(sum(x["recall5"] for x in rows) / n, 4) if n else None,
            "mrr": round(sum(x["mrr"] for x in rows) / n, 4) if n else None,
            "ndcg@5": round(sum(x["ndcg5"] for x in rows) / n, 4) if n else None,
        }
    return out


def main() -> int:
    ensure_utf8()
    ap = argparse.ArgumentParser(description="输入稳健性与边界测试")
    ap.add_argument("--modes", default=",".join(RETRIEVE_MODES))
    ap.add_argument("--paraphrase", action="store_true", help="追加 LLM 同义改写测试")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    golden = load_golden()
    n_pos = sum(1 for g in golden if g["type"] != "negative")
    print("[robust] golden 正样本 %d 条 | 模式 %s" % (n_pos, modes))

    print("[robust] 1/3 机械扰动 ...")
    pert = run_perturbation(golden, modes)
    print("[robust] 2/3 边界输入 ...")
    bound = run_boundary(modes)
    para = None
    if args.paraphrase:
        print("[robust] 3/3 LLM 同义改写（%d 次调用）..." % n_pos)
        para = run_paraphrase(golden, modes)

    rep = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "modes": modes, "n_positive": n_pos,
           "perturbation": pert, "boundary": bound, "paraphrase": para}
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_RB_JSON.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# 输入稳健性与边界测试报告", "",
         "> 生成时间：%s" % rep["generated_at"],
         "> 正样本 %d 条 | 模式 %s" % (n_pos, modes),
         "> 机械扰动为确定性生成（seed=42），可复现；零 LLM 成本。", "",
         "## 1. 机械扰动下的指标衰减", ""]
    labels = list(pert[modes[0]].keys())
    for metric in ("recall@5", "mrr", "ndcg@5"):
        L += ["### %s" % metric, "",
              "| 扰动 | " + " | ".join(modes) + " | 最大跌幅 |", "|---" * (len(modes) + 2) + "|"]
        for label in labels:
            vals = [pert[m][label][metric] for m in modes]
            base = pert[modes[0]]["orig"][metric]
            worst = min(v for v in vals if v is not None)
            drop = (worst - base) if base is not None else None
            L.append("| %s | %s | %+.4f |" % (
                label, " | ".join("%.4f" % v for v in vals), drop if drop is not None else 0.0))
        L.append("")
    L += ["## 2. 边界输入（是否崩溃）", "",
          "| 输入 | " + " | ".join(modes) + " |", "|---" * (len(modes) + 1) + "|"]
    for name, _ in BOUNDARY_INPUTS:
        cells = []
        for m in modes:
            row = next((r for r in bound if r["case"] == name and r["mode"] == m), None)
            if row is None:
                cells.append("?")
            elif row["ok"]:
                cells.append("OK(%d)" % row["n_returned"])
            else:
                cells.append("**崩溃**")
        L.append("| %s | %s |" % (name, " | ".join(cells)))
    L.append("")
    if para:
        L += ["## 3. 同义改写稳定性（与字面重合度无关的问法）", "",
              "| 模式 | n | Recall@5 | MRR | nDCG@5 |", "|---|---|---|---|---|"]
        for m in modes:
            d = para["metrics"][m]
            L.append("| %s | %d | %.4f | %.4f | %.4f |" % (m, d["n"], d["recall@5"], d["mrr"], d["ndcg@5"]))
        L += ["", "对照：原始问法（见 report_ir.md）dense MRR 0.7049 / bm25 0.8762 / rrf 0.8131", ""]
    REPORT_RB_MD.write_text("\n".join(L), encoding="utf-8")

    print("\n" + "=" * 70)
    for m in modes:
        print("模式 %-6s 原始 Recall@5=%.4f" % (m, pert[m]["orig"]["recall@5"]))
        for label in labels:
            if label == "orig":
                continue
            d = pert[m][label]["recall@5"] - pert[m]["orig"]["recall@5"]
            mark = "  <== 明显衰减" if d <= -0.05 else ""
            print("    %-14s Recall@5=%.4f  Δ=%+.4f%s" % (label, pert[m][label]["recall@5"], d, mark))
    bad = [r for r in bound if not r["ok"]]
    print("边界输入崩溃: %d 处" % len(bad))
    for r in bad[:8]:
        print("   %s [%s] -> %s" % (r["case"], r["mode"], r["error"]))
    if para:
        print("同义改写:")
        for m in modes:
            d = para["metrics"][m]
            print("   %-6s Recall@5=%.4f MRR=%.4f" % (m, d["recall@5"], d["mrr"]))
    print("=" * 70)
    print("报告：%s" % REPORT_RB_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
