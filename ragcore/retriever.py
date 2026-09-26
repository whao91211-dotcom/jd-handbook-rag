"""Step 4：检索增强 —— dense(chroma/bge) + BM25(jieba) -> RRF 融合。

支持三档检索模式（供消融实验 A/B 对比，见 METRICS.md §6.4）：
  mode="rrf"    双路 RRF 融合          —— 线上默认口径
  mode="dense"  仅 bge 向量检索
  mode="bm25"   仅 BM25 词法检索（不加载嵌入模型，最快）

用法：
  from ragcore.retriever import retrieve
  results = retrieve("员工请事假需要提前几天申请？", k=8)
  results = retrieve("...", k=8, mode="bm25")     # 消融
  # results: [{id,text,mode,rrf,dense_rank,bm25_rank,pages,page_start,path,clause,char_len}]

实现要点：
  - 单路模式复用同一套 RRF 打分代码：只喂一个 rank_map 时，RRF 分数对 rank 单调，
    排序结果与该路单独排序完全一致 —— 保证三档之间"只有变量不同"。
  - BM25 侧按 BM25_TOP_K 截断（原实现对全量 125 块排序、与 README 声明冲突，见 GAPS A-027）；
    传 bm25_top_k=None 可还原旧行为，用于量化该修复本身的影响。
"""
from __future__ import annotations

from .config import BM25_TOP_K, DENSE_TOP_K, FINAL_TOP_K, RETRIEVE_MODES, RRF_K
from .embedder import embed_query
from .store import get_collection
from .evidence import STRATEGIES, expand_queries, fuse_ranks, select_evidence

_corpus = None
_bm25_cache: tuple | None = None   # (ids_tuple, BM25Okapi, jieba)


def _load_corpus() -> tuple[list[str], list[dict], list[dict]]:
    """全量正文 + 元数据（125 块，量小，一次性读入并缓存）。"""
    global _corpus
    if _corpus is None:
        col = get_collection()
        data = col.get(include=["documents", "metadatas"])
        _corpus = (data["ids"], data["documents"], data["metadatas"])
    return _corpus


def _bm25_index(ids: list[str], docs: list[str]):
    """构建/复用 BM25 索引。

    A-021：原实现每次 retrieve() 都重建一遍 jieba 分词 + BM25 索引。
    此处按语料 id 序列缓存；语料变更（重建索引）时自动失效。
    """
    global _bm25_cache
    if _bm25_cache is None or _bm25_cache[0] != tuple(ids):
        import jieba
        from rank_bm25 import BM25Okapi

        corpus = [list(jieba.cut(d)) for d in docs]
        _bm25_cache = (tuple(ids), BM25Okapi(corpus), jieba)
    _, bm25, jieba = _bm25_cache
    return bm25, jieba


def _dense_ranks(query: str, n: int) -> dict[str, int]:
    """dense 检索 -> {chunk_id: 1-based rank}。"""
    vec = embed_query(query)
    col = get_collection()
    res = col.query(
        query_embeddings=[vec],
        n_results=min(DENSE_TOP_K, n),
        include=["distances"],
    )
    return {iid: rank for rank, iid in enumerate(res["ids"][0], start=1)}


def _bm25_ranks(
    query: str, ids: list[str], docs: list[str], top_k: int | None
) -> dict[str, int]:
    """BM25 检索 -> {chunk_id: 1-based rank}，只保留得分 > 0 的块。"""
    import jieba

    bm25, _ = _bm25_index(ids, docs)
    scores = bm25.get_scores(list(jieba.cut(query)))
    order = sorted(range(len(ids)), key=lambda i: -scores[i])
    if top_k is not None:
        order = order[:top_k]
    return {ids[i]: rank for rank, i in enumerate(order, start=1) if scores[i] > 0}


def retrieve(
    query: str,
    k: int = FINAL_TOP_K,
    mode: str = "rrf",
    bm25_top_k: int | None = BM25_TOP_K,
    strategy: str = "baseline",
) -> list[dict]:
    """返回检索结果 Top-k（从高到低）。

    Args:
        query: 查询文本。
        k: 返回块数。
        mode: "rrf" | "dense" | "bm25"（见模块 docstring）。
        bm25_top_k: BM25 侧候选截断数；None = 不截断（旧行为）。
    """
    if mode not in RETRIEVE_MODES:
        raise ValueError(f"mode 必须是 {RETRIEVE_MODES} 之一，收到 {mode!r}")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy 必须是 {STRATEGIES} 之一")
    if not query or not query.strip():
        return []
    query = query.strip()

    ids, docs, metas = _load_corpus()
    n = len(ids)
    k = max(1, min(k, n))

    # ---- 按 mode 选择参与融合的检索路 ----
    rank_maps: dict[str, dict[str, int]] = {}
    if mode in ("rrf", "dense"):
        rank_maps["dense"] = _dense_ranks(query, n)
    if mode in ("rrf", "bm25"):
        rank_maps["bm25"] = _bm25_ranks(query, ids, docs, bm25_top_k)
    queries = [query] if strategy == "baseline" else expand_queries(query)
    query_groups = [rank_maps.copy()]
    for number, supplement in enumerate(queries[1:], 1):
        group = {}
        if mode in ("rrf", "dense"):
            group["dense"] = _dense_ranks(supplement, n)
        if mode in ("rrf", "bm25"):
            group["bm25"] = _bm25_ranks(supplement, ids, docs, bm25_top_k)
        query_groups.append(group)
        for channel, ranks in group.items():
            rank_maps[f"{channel}:{number}"] = ranks

    # ---- RRF 融合（单路时退化为该路自身排序）----
    rrf = fuse_ranks(list(rank_maps.values()), RRF_K)
    fused_ids = sorted(rrf, key=lambda iid: -rrf[iid])
    facet_orders = []
    for group in query_groups:
        group_scores = fuse_ranks(list(group.values()), RRF_K)
        facet_orders.append(sorted(group_scores, key=lambda iid: -group_scores[iid]))
    top_ids = select_evidence(fused_ids, facet_orders, k) if strategy == "coverage" else fused_ids[:k]

    dense_rank = rank_maps.get("dense", {})
    bm25_rank = rank_maps.get("bm25", {})
    pos = {iid: i for i, iid in enumerate(ids)}

    results = []
    for iid in top_ids:
        i = pos[iid]
        meta = metas[i]
        results.append(
            {
                "id": iid,
                "text": docs[i],
                "mode": mode,
                "rrf": round(rrf[iid], 6),
                "dense_rank": dense_rank.get(iid),
                "bm25_rank": bm25_rank.get(iid),
                "pages": [int(x) for x in meta["pages"].split(",")] if meta.get("pages") else [int(meta["page_start"])],
                "page_start": int(meta["page_start"]),
                "path": meta.get("path", ""),
                "clause": meta.get("clause", ""),
                "char_len": int(meta.get("char_len", 0)),
                "queries": queries,
                "strategy": strategy,
                "candidate_count": len(rrf),
            }
        )
    return results
