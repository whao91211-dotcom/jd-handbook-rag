"""Step 6：L1 检索层「确定性」评测 —— Recall@k / Precision@k / MRR / nDCG@k + 三档消融。

为什么这一层不用 LLM 打分（见 求职/rag/METRICS.md 第 6.1 节）：
  消融的真实效应量常只有 2~5 个百分点，而 LLM-as-judge 同输入重复打分波动 5~15 个百分点
  → 仪器方差 > 效应量，测出来是噪声。确定性 IR 指标方差为 0，才可用于架构决策。

指标定义（Rel = gold 相关块集合，Ret_k = 返回前 k 个块，N = 题目数）：
  Hit@k       = (1/N)·Σ 1[|Rel ∩ Ret_k| > 0]            最宽松，只看「完全找不到」
  Recall@k    = (1/N)·Σ |Rel ∩ Ret_k| / |Rel|           查「漏没漏」（RAG 最关键）
  Precision@k = (1/N)·Σ |Rel ∩ Ret_k| / k               查「噪声多不多」
  MRR         = (1/N)·Σ 1/rank(首个命中)                只盯首个命中
  nDCG@k      = (1/N)·Σ DCG@k / IDCG@k                  召回 + 排序 + 位置折损

负样本（Rel = ∅）不计入 L1 平均（Recall/Precision 无定义），留待 L2 测拒答率。

用法：
  python -m ragcore.eval_ir
  python -m ragcore.eval_ir --modes rrf,dense,bm25 --k 1,3,5,8,20
  python -m ragcore.eval_ir --mode rrf --k 5          # 单档
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime

from .config import (
    EVAL_DIR,
    GOLDEN_JSONL,
    REPORT_IR_JSON,
    REPORT_IR_MD,
    RETRIEVE_MODES,
)
from .retriever import retrieve

DEFAULT_KS = (1, 3, 5, 8, 20)
MAX_K = 20          # 单题最大召回深度（MRR / 失败诊断都用它）


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


# ------------------------------------------------------------------ 单题指标

def hit_at_k(rel: set, ret: list, k: int) -> float:
    return 1.0 if rel & set(ret[:k]) else 0.0


def recall_at_k(rel: set, ret: list, k: int):
    if not rel:
        return None
    return len(rel & set(ret[:k])) / len(rel)


def precision_at_k(rel: set, ret: list, k: int) -> float:
    if k <= 0:
        return 0.0
    return len(rel & set(ret[:k])) / k


def reciprocal_rank(rel: set, ret: list) -> float:
    for i, doc in enumerate(ret, 1):
        if doc in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rel: set, ret: list, k: int):
    if not rel:
        return None
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ret[:k], 1) if d in rel)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    if idcg == 0:
        return None
    return dcg / idcg


# ------------------------------------------------------------------ 评测

def run(golden: list[dict], modes: list[str], ks: list[int]) -> dict:
    positives = [g for g in golden if g["type"] != "negative" and g["gold_chunk_ids"]]
    negatives = [g for g in golden if g["type"] == "negative"]
    cutoff = max(max(ks), MAX_K)

    # 每题每档：实际返回的块 id 序列
    retrieved: dict[tuple[str, str], list[str]] = {}
    for mode in modes:
        print("[eval] mode=%s 评测 %d 题 ..." % (mode, len(positives)))
        for g in positives:
            hits = retrieve(g["question"], k=cutoff, mode=mode)
            retrieved[(mode, g["qid"])] = [h["id"] for h in hits]

    # ---- 汇总 ----
    agg: dict[tuple[str, int], dict[str, float]] = {}
    agg_global: dict[str, dict[str, float]] = {}
    for mode in modes:
        agg_global[mode] = {
            "mrr": sum(reciprocal_rank(set(g["gold_chunk_ids"]), retrieved[(mode, g["qid"])])
                       for g in positives) / len(positives)
        }
        for k in ks:
            rows = []
            for g in positives:
                rel = set(g["gold_chunk_ids"])
                ret = retrieved[(mode, g["qid"])]
                rows.append({
                    "hit": hit_at_k(rel, ret, k),
                    "recall": recall_at_k(rel, ret, k),
                    "precision": precision_at_k(rel, ret, k),
                    "ndcg": ndcg_at_k(rel, ret, k),
                })
            n = len(rows)
            agg[(mode, k)] = {
                "hit@k": sum(r["hit"] for r in rows) / n,
                "recall@k": sum(r["recall"] for r in rows) / n,
                "precision@k": sum(r["precision"] for r in rows) / n,
                "ndcg@k": sum(r["ndcg"] for r in rows) / n,
            }

    # ---- 分层切片（按 question type）----
    slices: dict[tuple[str, str, int], dict] = {}
    types = sorted({g["type"] for g in positives})
    for mode in modes:
        for t in types:
            sub = [g for g in positives if g["type"] == t]
            for k in ks:
                rows = []
                for g in sub:
                    rel = set(g["gold_chunk_ids"])
                    ret = retrieved[(mode, g["qid"])]
                    rows.append({
                        "recall": recall_at_k(rel, ret, k),
                        "ndcg": ndcg_at_k(rel, ret, k),
                        "rr": reciprocal_rank(rel, ret),
                    })
                m = len(rows)
                slices[(mode, t, k)] = {
                    "n": m,
                    "recall@k": sum(r["recall"] for r in rows) / m,
                    "ndcg@k": sum(r["ndcg"] for r in rows) / m,
                    "mrr": sum(r["rr"] for r in rows) / m,
                }

    # ---- 失败 case（Recall@5 = 0）----
    diag_k = 5 if 5 in ks else ks[-1]
    failures = []
    for mode in modes:
        for g in positives:
            rel = set(g["gold_chunk_ids"])
            ret = retrieved[(mode, g["qid"])]
            if recall_at_k(rel, ret, diag_k) == 0.0:
                failures.append({
                    "mode": mode,
                    "qid": g["qid"],
                    "type": g["type"],
                    "question": g["question"],
                    "gold_chunk_ids": g["gold_chunk_ids"],
                    "gold_path": g.get("source_path", ""),
                    "gold_pages": g.get("gold_pages", []),
                    "returned_top5": ret[:diag_k],
                    "first_hit_rank": next((i for i, d in enumerate(ret, 1) if d in rel), None),
                })

    # ---- 负样本（不入 L1 平均）----
    neg_info = []
    for g in negatives:
        hits = retrieve(g["question"], k=4, mode="rrf")
        neg_info.append({
            "qid": g["qid"],
            "question": g["question"],
            "negative_kind": g.get("negative_kind", "plain"),
            "top1_id": hits[0]["id"] if hits else None,
            "top1_path": hits[0]["path"] if hits else None,
            "n_returned": len(hits),
        })

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(GOLDEN_JSONL),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "modes": modes,
        "ks": ks,
        "cutoff": cutoff,
        "aggregate": {"%s@k=%d" % (m, k): v for (m, k), v in agg.items()},
        "mrr": {m: agg_global[m]["mrr"] for m in modes},
        "slices": {"%s|%s@k=%d" % (m, t, k): v for (m, t, k), v in slices.items()},
        "failures": failures,
        "negatives": neg_info,
        "retrieved_ids": {"%s|%s" % (m, q): v for (m, q), v in retrieved.items()},
    }
    return report


# ------------------------------------------------------------------ 报告

def build_markdown(rep: dict) -> str:
    modes = rep["modes"]
    ks = rep["ks"]
    L: list[str] = []
    L.append("# L1 检索层评测报告（确定性 IR 指标）")
    L.append("")
    L.append("> 生成时间：%s" % rep["generated_at"])
    L.append("> 数据集：%s（正样本 %d · 负样本 %d）" % (rep["dataset"], rep["n_positive"], rep["n_negative"]))
    L.append("> 仪器：确定性计算（非 LLM 打分），**方差为 0**，可复现、可进 CI")
    L.append("> 指标定义见 ragcore/eval_ir.py 模块 docstring 与 METRICS.md 第 6 节")
    L.append("")

    L.append("## 1. 三档消融 · 主指标")
    L.append("")
    L.append("| 模式 | k | Hit@k | **Recall@k** | Precision@k | **nDCG@k** |")
    L.append("|---|---|---|---|---|---|")
    for m in modes:
        for k in ks:
            a = rep["aggregate"]["%s@k=%d" % (m, k)]
            L.append("| %s | %d | %.4f | **%.4f** | %.4f | **%.4f** |" % (
                m, k, a["hit@k"], a["recall@k"], a["precision@k"], a["ndcg@k"]))
    L.append("")

    L.append("## 2. MRR（cutoff=%d）" % rep["cutoff"])
    L.append("")
    L.append("| 模式 | MRR |")
    L.append("|---|---|")
    for m in modes:
        L.append("| %s | %.4f |" % (m, rep["mrr"][m]))
    L.append("")

    L.append("## 3. 分层切片（Recall / nDCG / MRR）")
    L.append("")
    for k in ks:
        L.append("### k=%d" % k)
        L.append("")
        L.append("| 类型 | n | " + " | ".join("%s Recall" % m for m in modes) + " | " +
                 " | ".join("%s nDCG" % m for m in modes) + " | " + " | ".join("%s MRR" % m for m in modes) + " |")
        L.append("|---" * (1 + 3 * len(modes)) + "|")
        types = sorted({key.split("|")[1].split("@")[0] for key in rep["slices"]})
        for t in types:
            row = [t]
            n = rep["slices"]["%s|%s@k=%d" % (modes[0], t, k)]["n"]
            row.append(str(n))
            for m in modes:
                row.append("%.3f" % rep["slices"]["%s|%s@k=%d" % (m, t, k)]["recall@k"])
            for m in modes:
                row.append("%.3f" % rep["slices"]["%s|%s@k=%d" % (m, t, k)]["ndcg@k"])
            for m in modes:
                row.append("%.3f" % rep["slices"]["%s|%s@k=%d" % (m, t, k)]["mrr"])
            L.append("| " + " | ".join(row) + " |")
        L.append("")

    L.append("## 4. 失败清单（Recall@%d = 0）" % (5 if 5 in ks else ks[-1]))
    L.append("")
    if not rep["failures"]:
        L.append("（无）")
    else:
        L.append("共 %d 条（跨模式累计）：" % len(rep["failures"]))
        L.append("")
        L.append("| 模式 | qid | 类型 | 问题 | gold | gold 章节 | 首个命中rank |")
        L.append("|---|---|---|---|---|---|---|")
        for f in rep["failures"]:
            L.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                f["mode"], f["qid"], f["type"], f["question"][:30],
                ", ".join(f["gold_chunk_ids"]), (f["gold_path"] or "")[:26],
                f["first_hit_rank"] if f["first_hit_rank"] else "未命中"))
    L.append("")

    L.append("## 5. 负样本（不计入 L1 平均，留待 L2 测拒答率）")
    L.append("")
    L.append("| qid | 类型 | 问题 | Top1 | Top1 章节 |")
    L.append("|---|---|---|---|---|")
    for n in rep["negatives"]:
        L.append("| %s | %s | %s | %s | %s |" % (
            n["qid"], n["negative_kind"], n["question"][:26],
            n["top1_id"] or "-", (n["top1_path"] or "-")[:30]))
    L.append("")
    return "\n".join(L)


def main() -> int:
    ensure_utf8()
    ap = argparse.ArgumentParser(description="L1 检索层确定性评测")
    ap.add_argument("--modes", default=",".join(RETRIEVE_MODES),
                    help="逗号分隔，默认全部三档")
    ap.add_argument("--k", default=",".join(str(x) for x in DEFAULT_KS),
                    help="逗号分隔的 k 值")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    ks = sorted({int(x) for x in args.k.split(",") if x.strip()})
    bad = [m for m in modes if m not in RETRIEVE_MODES]
    if bad:
        print("[错误] 未知模式 %s，可选 %s" % (bad, list(RETRIEVE_MODES)), file=sys.stderr)
        return 1

    golden = load_golden()
    unreviewed = [g["qid"] for g in golden if not g.get("reviewed")]
    if unreviewed:
        print("[警告] 有 %d 条未人工校核：%s" % (len(unreviewed), unreviewed), file=sys.stderr)
    print("[eval] 载入 %d 条标注（正 %d / 负 %d）" % (
        len(golden),
        sum(1 for g in golden if g["type"] != "negative"),
        sum(1 for g in golden if g["type"] == "negative")))

    rep = run(golden, modes, ks)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IR_JSON.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    md = build_markdown(rep)
    REPORT_IR_MD.write_text(md, encoding="utf-8")

    print("\n" + "=" * 66)
    print("主指标（Recall@k）")
    print("-" * 66)
    header = "%-7s" % "mode" + "".join("%9s" % ("k=%d" % k) for k in ks)
    print(header)
    for m in modes:
        print("%-7s" % m + "".join("%9.4f" % rep["aggregate"]["%s@k=%d" % (m, k)]["recall@k"] for k in ks))
    print("-" * 66)
    print("MRR     " + "".join("  %-6s %.3f" % (m, rep["mrr"][m]) for m in modes))
    print("失败条数 " + "  " + str(len(rep["failures"])))
    print("=" * 66)
    print("报告：%s" % REPORT_IR_MD)
    print("      %s" % REPORT_IR_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
