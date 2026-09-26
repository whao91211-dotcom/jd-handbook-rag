"""Step 5：问答 CLI。

用法（InternVL 环境，项目根目录）：
  python ask.py "请事假需要提前几天申请？"     # 单发问答（带引用）
  python ask.py                                  # 交互式 REPL（输入 exit 退出）
  python ask.py "问题" --no-llm                  # 仅检索，展示 Top-K 片段（无需 API）
  python ask.py "问题" --top-k 10                # 调整送入的块数
"""
from __future__ import annotations

import argparse
import sys

from ragcore.config import FINAL_TOP_K, RETRIEVE_MODES, ROOT
from ragcore.llm import api_ready, generate, last_usage
from ragcore.retriever import retrieve
from ragcore.evidence import STRATEGIES

HELP = """可用的示例问题：
  请事假需要提前几天申请？
  带薪年假的天数如何计算？
  加班费怎么计算？
  员工离职需要办理哪些流程？
  违反保密义务有什么后果？
输入 exit / quit 退出。"""


def ensure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def print_sources(chunks: list[dict]) -> None:
    print("\n— 参考来源（按相关度）—")
    for i, c in enumerate(chunks, 1):
        pages = ",".join(str(p) for p in c["pages"])
        head = c["text"].replace("\n", " ")
        if len(head) > 150:
            head = head[:150] + "…"
        print(f"{i}. 第{pages}页 · {c['path'] or '(无章节)'}")
        print(f"   {head}")


def answer_once(question: str, top_k: int, use_llm: bool, mode: str = "rrf", strategy: str = "baseline") -> int:
    print(f"Q: {question}")
    if mode != "rrf":
        print(f"[消融] 检索模式 = {mode}（线上默认为 rrf）")
    print()
    chunks = retrieve(question, k=top_k, mode=mode, strategy=strategy)
    if not chunks:
        print("（未检索到任何相关内容）")
        return 1
    if use_llm:
        if not api_ready():
            print("[提示] .env 未配置 OPENAI_API_KEY / OPENAI_BASE_URL，改用仅检索模式。")
            use_llm = False
        else:
            try:
                answer = generate(question, chunks)
                # 双保险：generate() 已保证不返回空串，这里再兜一层，防止将来改坏
                if not answer.strip():
                    raise RuntimeError("模型返回了空内容")
                print("A:", answer, "\n")
                u = last_usage()
                if u:
                    print("— 本次用量 — %s · 第%s次尝试 · 预算%s · "
                          "prompt %s + completion %s（其中 reasoning %s）· 正文 %s 字\n" % (
                              u.get("model"), u.get("attempt"), u.get("max_tokens"),
                              u.get("prompt_tokens"), u.get("completion_tokens"),
                              u.get("reasoning_tokens"), u.get("content_chars")))
            except Exception as exc:
                print(f"[提示] LLM 调用失败（{type(exc).__name__}: {exc}），降级为仅检索模式：\n")
                use_llm = False
    if not use_llm:
        print_sources(chunks)
    else:
        print("\n— 检索命中的核心片段（供核对）—")
        for i, c in enumerate(chunks[:3], 1):
            pages = ",".join(str(p) for p in c["pages"])
            head = c["text"].replace("\n", " ")
            if len(head) > 150:
                head = head[:150] + "…"
            print(f"{i}. 第{pages}页 · {c['path'] or '(无章节)'}")
            print(f"   {head}")
    return 0


def main() -> int:
    ensure_utf8()
    ap = argparse.ArgumentParser(description="京东员工手册 RAG 问答")
    ap.add_argument("question", nargs="?", default=None, help="问题；缺省进入交互模式")
    ap.add_argument("--top-k", type=int, default=FINAL_TOP_K, help=f"送入的块数（默认 {FINAL_TOP_K}）")
    ap.add_argument("--no-llm", action="store_true", help="仅检索，不调用 LLM")
    ap.add_argument(
        "--mode",
        choices=RETRIEVE_MODES,
        default="rrf",
        help="检索模式（消融实验）：rrf=双路融合（默认）/ dense=仅向量 / bm25=仅词法",
    )
    ap.add_argument('--strategy', choices=STRATEGIES, default='baseline', help='检索实验策略')
    args = ap.parse_args()

    use_llm = not args.no_llm
    if args.question:
        return answer_once(args.question, args.top_k, use_llm, args.mode, args.strategy)

    print("京东员工手册 RAG 问答（交互模式）· exit 退出\n" + HELP)
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            return 0
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("再见")
            return 0
        answer_once(q, args.top_k, use_llm, args.mode, args.strategy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
