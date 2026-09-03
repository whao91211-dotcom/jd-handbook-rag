"""Step 4 配套：检索冒烟评测（不调用 LLM）。

判定：问题的关键词出现在 Top-3 检索结果的正文中即为 PASS。
结果同时打印并写入 data/eval_report.txt。

用法：python -m ragcore.eval
"""
from __future__ import annotations

import sys

from .config import DATA_DIR
from .retriever import retrieve

# (问题, 期望命中的关键词列表，命中任一即可)
EVAL_CASES = [
    ("员工请事假需要提前几天申请？", ["事假"]),
    ("带薪年假的天数如何计算？", ["年假"]),
    ("病假期间的工资如何发放？", ["病假"]),
    ("加班费如何计算？", ["加班"]),
    ("员工离职需要办理哪些流程？", ["离职"]),
    ("女员工产假能休多少天？", ["产假"]),
    ("新员工试用期多长？", ["试用"]),
    ("违反保密义务有什么后果？", ["保密"]),
    ("公司提供哪些员工福利？", ["福利"]),
    ("员工申诉与投诉的渠道是什么？", ["申诉"]),
    ("差旅费用报销有什么要求？", ["报销"]),
]


def ensure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def main() -> int:
    ensure_utf8()
    lines: list[str] = []
    passed = 0
    for q, keywords in EVAL_CASES:
        hits = retrieve(q, k=3)
        texts = " ".join(h["text"] for h in hits)
        ok = any(kw in texts for kw in keywords)
        passed += int(ok)
        top = hits[0] if hits else None
        topinfo = (
            f"p{top['page_start']} | {top['path']}" if top else "无结果"
        )
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {q}")
        lines.append(f"        期望关键词: {'/'.join(keywords)} | Top1: {topinfo}")
        print(f"[{'PASS' if ok else 'FAIL'}] {q}")
        print(f"        期望关键词: {'/'.join(keywords)} | Top1: {topinfo}")
    summary = f"\n通过 {passed}/{len(EVAL_CASES)}"
    lines.append(summary)
    print(summary)
    report = DATA_DIR / "eval_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {report}")
    return 0 if passed == len(EVAL_CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
