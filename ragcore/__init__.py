"""ragcore —— 京东员工手册本地 RAG 可复用核心库。

管线：PDF -> 分块 -> bge 嵌入 -> chroma 入库 -> dense+BM25(RRF) 检索 -> OpenAI 兼容 LLM 生成
"""
__version__ = "0.1.0"
