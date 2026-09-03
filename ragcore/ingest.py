"""Step 3：入库入口 —— 分块 -> 嵌入 -> 写入 Chroma。

用法（InternVL 环境，项目根目录下）：
  python -m ragcore.ingest            # 默认：重建索引（自动复用已有 chunks.jsonl）
  python -m ragcore.ingest --rechunk  # 先重新分块再入库
  python -m ragcore.ingest --append   # 不清空，按 id 追加/覆盖
"""
from __future__ import annotations

import argparse
import json
import sys

from . import chunker
from .config import CHUNKS_JSONL, STATS_INGEST_JSON
from .embedder import embed_texts
from .store import add_chunks, get_collection, reset_collection


def ensure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def load_chunks() -> list[dict]:
    with CHUNKS_JSONL.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    ensure_utf8()
    ap = argparse.ArgumentParser(description="京东员工手册 RAG 入库")
    ap.add_argument("--rechunk", action="store_true", help="先重新分块")
    ap.add_argument("--append", action="store_true", help="不清空集合，按 id 追加")
    args = ap.parse_args()

    if args.rechunk or not CHUNKS_JSONL.exists():
        print("[ingest] 重新分块 ...")
        chunker.main()

    chunks = load_chunks()
    print(f"[ingest] 读取 {len(chunks)} 个分块")

    print("[ingest] 批量嵌入 ...")
    texts = [c["text"] for c in chunks]
    vecs = embed_texts(texts)
    dim = len(vecs[0]) if vecs else 0
    print(f"[ingest] 嵌入完成 dim={dim}")

    if args.append:
        col = get_collection()
    else:
        print("[ingest] 重建集合 ...")
        col = reset_collection()
    n = add_chunks(col, chunks, vecs)
    stats = {
        "step": "ingest",
        "mode": "append" if args.append else "rebuild",
        "chunks": n,
        "dim": dim,
        "collection": col.name,
        "count_after": col.count(),
    }
    STATS_INGEST_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-" * 60)
    print(f"入库分块 : {n}")
    print(f"向量维度 : {dim}")
    print(f"集合计数 : {col.count()}")
    print(f"统计     : {STATS_INGEST_JSON}")
    print("-" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
