"""Step 3a：bge-small-zh 嵌入器（CPU）。

文档侧不做指令前缀；查询侧加 bge 官方指令（见 EMBED_QUERY_PREFIX），
两路均 cosine 归一化。
"""
from __future__ import annotations

import functools
import os
from typing import Iterable

# 必须在导入 sentence_transformers / transformers 之前设置：
# 模型已在本机 HF 缓存，禁止任何联网探测（否则会卡在 huggingface.co 连接超时）。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer

from .config import EMBED_BATCH, EMBED_MODEL, EMBED_QUERY_PREFIX


@functools.lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    print(f"[embed] 加载模型 {EMBED_MODEL} (CPU) ...")
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed_texts(texts: Iterable[str], batch_size: int = EMBED_BATCH) -> list[list[float]]:
    """批量嵌入文档（不添加查询前缀）。"""
    model = get_model()
    vecs = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.tolist()


def embed_query(query: str) -> list[float]:
    """嵌入单个查询（带 bge 检索指令前缀）。"""
    model = get_model()
    vec = model.encode(
        [EMBED_QUERY_PREFIX + query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return vec.tolist()
