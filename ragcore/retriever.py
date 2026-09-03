"""Step 4：检索增强 —— dense(chroma/bge) + BM25(jieba) -> RRF 融合。

用法：
  from ragcore.retriever import retrieve
  results = retrieve("员工请事假需要提前几天申请？", k=8)
  # results: [{id,text,rrf,dense_rank,bm25_rank,pages,page_start,path,clause,char_len}]
"""
from __future__ import annotations

from .config import DENSE_TOP_K, FINAL_TOP_K, RRF_K
from .embedder import embed_query
from .store import get_collection

_corpus = None


def _load_corpus() -> tuple[list[str], list[dict], list[dict]]:
    """全量正文 + 元数据（125 块，量小，一次性读入并缓存）。"""
    global _corpus
    if _corpus is None:
        col = get_collection()
        data = col.get(include=["documents", "metadatas"])
        _corpus = (data["ids"], data["documents"], data["metadatas"])
    return _corpus


def _bm25_index(docs: list[str]):
    import jieba
    from rank_bm25 import BM25Okapi

    corpus = [list(jieba.cut(d)) for d in docs]
    return BM25Okapi(corpus), jieba


def retrieve(query: str, k: int = FINAL_TOP_K) -> list[dict]:
    """返回 RRF 融合后的 Top-k 检索结果（从高到低）。"""
    if not query or not query.strip():
        return []
    ids, docs, metas = _load_corpus()
    n = len(ids)
    k = max(1, min(k, n))

    # ---- dense：chroma cosine ----
    vec = embed_query(query.strip())
    col = get_collection()
    dense_res = col.query(
        query_embeddings=[vec],
        n_results=min(DENSE_TOP_K, n),
        include=["distances"],
    )
    dense_ids = dense_res["ids"][0]
    dense_rank = {iid: rank for rank, iid in enumerate(dense_ids, start=1)}

    # ---- BM25：jieba 分词 ----
    bm25, jieba = _bm25_index(docs)
    q_tokens = list(jieba.cut(query.strip()))
    bm_scores = bm25.get_scores(q_tokens)
    bm_order = sorted(range(n), key=lambda i: -bm_scores[i])
    bm_rank = {ids[i]: rank for rank, i in enumerate(bm_order, start=1) if bm_scores[i] > 0}

    # ---- RRF 融合 ----
    rrf: dict[str, float] = {}
    for rank_map in (dense_rank, bm_rank):
        for iid, rank in rank_map.items():
            rrf[iid] = rrf.get(iid, 0.0) + 1.0 / (RRF_K + rank)
    top_ids = sorted(rrf, key=lambda iid: -rrf[iid])[:k]
    pos = {iid: i for i, iid in enumerate(ids)}

    results = []
    for iid in top_ids:
        i = pos[iid]
        meta = metas[i]
        results.append(
            {
                "id": iid,
                "text": docs[i],
                "rrf": round(rrf[iid], 4),
                "dense_rank": dense_rank.get(iid),
                "bm25_rank": bm_rank.get(iid),
                "pages": [int(x) for x in meta["pages"].split(",")] if meta.get("pages") else [int(meta["page_start"])],
                "page_start": int(meta["page_start"]),
                "path": meta.get("path", ""),
                "clause": meta.get("clause", ""),
                "char_len": int(meta.get("char_len", 0)),
            }
        )
    return results
