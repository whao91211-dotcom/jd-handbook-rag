"""全局路径与参数常量。"""
from __future__ import annotations

from pathlib import Path

# ---- 路径 ----
ROOT = Path(__file__).resolve().parent.parent          # 项目根 = 工作区
PDF_PATH = ROOT / "京东集团员工手册.pdf"
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"

EDITABLE_MD = DATA_DIR / "handbook_editable.md"        # Step1 可编辑整册文本
PAGES_JSONL = DATA_DIR / "pages.jsonl"                 # Step1 逐页结构化记录
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"               # Step2 分块产物
STATS_JSON = DATA_DIR / "stats.json"                   # Step1 统计报告
STATS_CHUNK_JSON = DATA_DIR / "stats_chunk.json"       # Step2 统计报告
STATS_INGEST_JSON = DATA_DIR / "stats_ingest.json"     # Step3 统计报告

# ---- Step1: 页眉/页脚噪声剔除 ----
NOISE_MIN_PAGE_FREQ = 30   # 一行出现在 >= N 个不同页，视为页眉/页脚候选
NOISE_MAX_EDGE = 3         # 只处理每页首/尾 <= K 行的位置

# ---- Step2: 分块参数 ----
CHUNK_TARGET = 500         # 目标块字符数（中文）
CHUNK_MAX = 750            # 硬上限：超出后递归切
CHUNK_OVERLAP = 80         # 相邻块重叠字符
CHUNK_MIN = 100            # 小于该值并入前块或丢弃

# ---- Step3: 嵌入 ----
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"   # 本机 HF 缓存已就绪（dim=512）
EMBED_BATCH = 32
EMBED_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："   # bge 查询侧指令
COLLECTION = "jd_handbook"
CHROMA_SPACE = "cosine"

# ---- Step4: 检索 ----
DENSE_TOP_K = 20           # dense / bm25 各自先取
RRF_K = 60                 # RRF 融合常数
FINAL_TOP_K = 8            # 最终送入 LLM 的块数
BM25_NGRAM = True          # 无 jieba 时退化为字符 bigram 兜底

# ---- Step5: 生成 ----
LLM_CONTEXT_BUDGET = 6000  # 送入 LLM 的上下文最大字符数
