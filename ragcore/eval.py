"""离线检索评测：人工标注证据页和原文片段，不调用生成模型。"""
from __future__ import annotations

import json
import re
import sys

from .config import DATA_DIR, PAGES_JSONL
from .retriever import retrieve

# (问题, 证据页, 证据片段)：片段已在 PDF 提取文本中人工核对。
ANSWERABLE_CASES = [
    ("员工申请事假有哪些前提？", 23, "年休假及调休休完后申请事假"),
    ("累计工作满十年不满二十年，法定年假多少天？", 23, "每年法定年假标准为 10 天"),
    ("员工每年有多少天全薪福利病假？", 22, "5 天全薪福利病假"),
    ("法定节假日加班如何支付加班费？", 21, "按照国家规定支付加班费"),
    ("试用期员工辞职需要提前多久申请？", 18, "至少提前三日提出书面离职申请"),
    ("手册举例中，北京地区女员工产假是多少天？", 24, "北京地区员工享有 128 天的产假"),
    ("新员工的试用期是多久？", 15, "试用期为一至六个月"),
    ("手册如何界定违反保密义务的行为？", 46, "泄漏公司秘密、违反保密义务的"),
    ("每月十五日后入职，社保和公积金何时办理？", 29, "每月 15 日后入职的员工，于次月办理"),
    ("员工申诉可以拨打什么热线？", 33, "大耳朵热线：4006183638"),
    ("手册对不真实的费用报销有什么规定？", 46, "费用报销不真实"),
]

# 检索命中相关章节也不代表有答案；这两例需在生成阶段测拒答。
UNANSWERABLE_CASES = [
    "员工请事假必须提前几天申请？",
    "差旅费用报销的票据和审批标准是什么？",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def validate_evidence() -> None:
    pages = {p["page"]: p for p in (json.loads(line) for line in PAGES_JSONL.open(encoding="utf-8"))}
    for question, page, evidence in ANSWERABLE_CASES:
        if page not in pages or normalize(evidence) not in normalize(pages[page]["text"]):
            raise ValueError(f"评测证据未在第 {page} 页找到：{question} / {evidence}")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    validate_evidence()
    rows = []
    for question, page, evidence in ANSWERABLE_CASES:
        hits = retrieve(question, k=3)
        rank = next((i for i, hit in enumerate(hits, 1)
                     if page in hit["pages"] and normalize(evidence) in normalize(hit["text"])), None)
        rows.append({"question": question, "evidence_page": page,
                     "evidence": evidence, "rank": rank,
                     "top3": [{"id": h["id"], "pages": h["pages"]} for h in hits]})
        print(f"[{'PASS' if rank else 'FAIL'}] rank={rank or '-'} p{page} {question}")
    n = len(rows)
    metrics = {
        "hit_at_1": sum(r["rank"] == 1 for r in rows) / n,
        "hit_at_3": sum(r["rank"] is not None for r in rows) / n,
        "mrr_at_3": sum(1 / r["rank"] if r["rank"] else 0 for r in rows) / n,
    }
    report = {"method": "page-and-evidence", "metrics": metrics,
              "answerable_cases": rows, "unanswerable_cases_not_scored": UNANSWERABLE_CASES}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "eval_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Hit@1={metrics['hit_at_1']:.1%} Hit@3={metrics['hit_at_3']:.1%} MRR@3={metrics['mrr_at_3']:.3f}")
    print(f"报告：{path}")
    print(f"另有 {len(UNANSWERABLE_CASES)} 个拒答问题，需单独评测生成答案。")
    return 0 if metrics["hit_at_3"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
