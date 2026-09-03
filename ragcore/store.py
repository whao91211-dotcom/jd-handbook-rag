"""Step 3b：ChromaDB 向量库存取。"""
from __future__ import annotations

from typing import Optional

import chromadb

from .config import CHROMA_DIR, CHROMA_SPACE, COLLECTION

_client: Optional[chromadb.PersistentClient] = None


def client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_collection(create: bool = True):
    c = client()
    if create:
        return c.get_or_create_collection(COLLECTION, metadata={"hnsw:space": CHROMA_SPACE})
    try:
        return c.get_collection(COLLECTION)
    except Exception:
        raise RuntimeError(f"集合 {COLLECTION} 不存在，请先运行 ingest.py")


def reset_collection():
    """删除并重建集合（幂等重建）。"""
    c = client()
    try:
        c.delete_collection(COLLECTION)
    except Exception:
        pass
    return c.create_collection(COLLECTION, metadata={"hnsw:space": CHROMA_SPACE})


def add_chunks(col, chunks: list[dict], embeddings: list[list[float]]) -> int:
    ids = [c["id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [
        {
            "page_start": c["pages"][0],
            "pages": ",".join(str(p) for p in c["pages"]),
            "path": c["path"] or "(无章节)",
            "clause": c["clause_range"],
            "char_len": c["char_len"],
        }
        for c in chunks
    ]
    col.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    return len(ids)
