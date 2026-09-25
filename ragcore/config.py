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
DENSE_TOP_K = 20           # dense 侧候选数
BM25_TOP_K = 20            # BM25 侧候选数（原实现未截断，与 README 声明不一致 → GAPS A-027）
RRF_K = 60                 # RRF 融合常数
FINAL_TOP_K = 8            # 最终送入 LLM 的块数
RETRIEVE_MODES = ("rrf", "dense", "bm25")   # 消融实验三档；rrf = 线上默认
BM25_NGRAM = True          # 无 jieba 时退化为字符 bigram 兜底

# ---- Step5: 生成 ----
LLM_CONTEXT_BUDGET = 6000  # 送入 LLM 的上下文最大字符数

# ---- Step5: 生成 token 预算 ----
# 本模型带 reasoning，思考过程与正文**共享**同一 token 预算。
# 实测 max_tokens=1024 时 reasoning 恰好吃光全部预算 -> content 为空（GAPS A-043）。
LLM_MAX_TOKENS = 2048      # 首次尝试的预算
LLM_MAX_TOKENS_CAP = 8192  # 重试时的封顶（预算逐次翻倍）
LLM_GEN_ATTEMPTS = 3       # 正文为空时的总尝试次数

# ---- Step6: 评测（L1 检索层 / golden set）----
EVAL_DIR = DATA_DIR / "eval"
GOLDEN_JSONL = EVAL_DIR / "golden.jsonl"            # 评测数据集（gold_chunk_ids 经人工校核）
GOLDEN_REVIEW_MD = EVAL_DIR / "golden_review.md"    # 人工校核清单
STATS_GOLDEN_JSON = EVAL_DIR / "stats_golden.json"  # 生成统计
REPORT_IR_MD = EVAL_DIR / "report_ir.md"            # L1 检索评测报告（人读）
REPORT_IR_JSON = EVAL_DIR / "report_ir.json"        # L1 检索评测报告（机读）
