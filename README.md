# 京东员工手册 · 本地 RAG

基于《京东集团员工手册.pdf》（51 页 / 约 5 万字符 / 7 页表格）的本地检索增强问答项目。
检索与嵌入完全本地（CPU），仅生成环节调用 OpenAI 兼容 API（默认 DeepSeek）。

## 运行环境

- Python：`C:\Users\wuhao\.conda\envs\InternVL\python.exe`（py3.10，CPU）
- 依赖：见 `requirements-internvl.txt`（chromadb 1.5.9 / sentence-transformers / torch-cpu 已装，
  补充安装 pdfplumber / rank_bm25 / jieba / python-dotenv）
- 嵌入模型：`BAAI/bge-small-zh-v1.5`（本机 HF 缓存，离线加载，dim=512）

## 管线与文件

```
京东集团员工手册.pdf
  │  ragcore/extract.py   逐页正文+表格，剔除页眉页脚/页码/正文表格去重
  ▼
data/pages.jsonl (51 页) + data/handbook_editable.md（可编辑整册文本）
  │  ragcore/chunker.py   章节识别 + 短条款合并，目标 ~500 字符/块
  ▼
data/chunks.jsonl (125 块，带【章节路径】前缀与页码)
  │  ragcore/embedder.py + store.py + ingest.py   bge 嵌入 → chroma
  ▼
data/chroma/（集合 jd_handbook，cosine，125 条向量）
  │  ragcore/retriever.py   dense(bge) + BM25(jieba) → RRF 融合
  ▼
ragcore/llm.py + ask.py   OpenAI 兼容 API 生成带引用回答
```

## 使用

```powershell
# 0) 配置 API（一次）
#    复制 .env.example 为 .env 并填写 OPENAI_API_KEY / OPENAI_BASE_URL / RAG_LLM_MODEL

# 1) 重建知识库（可选，分步）
& "C:\Users\wuhao\.conda\envs\InternVL\python.exe" -m ragcore.extract     # PDF→文本
& "C:\Users\wuhao\.conda\envs\InternVL\python.exe" -m ragcore.chunker     # →分块
& "C:\Users\wuhao\.conda\envs\InternVL\python.exe" -m ragcore.ingest      # →嵌入入库(可 --rechunk)

# 2) 问答
& "C:\Users\wuhao\.conda\envs\InternVL\python.exe" ask.py "员工请事假需要提前几天申请？"
& "C:\Users\wuhao\.conda\envs\InternVL\python.exe" ask.py                 # 交互模式
& "C:\Users\wuhao\.conda\envs\InternVL\python.exe" ask.py "加班费怎么计算？" --no-llm  # 仅检索

# 3) 检索评测（离线）
& "C:\Users\wuhao\.conda\envs\InternVL\python.exe" -m ragcore.eval
```

## 评测结果

`ragcore/eval.py` 内置 11 个覆盖各章节的冒烟问题（事假/年假/病假/加班/离职/产假/试用期/
保密/福利/申诉/报销），判定关键词命中 Top-3 检索结果即 PASS：**11/11 通过**。
报告：`data/eval_report.txt`。

示例问答（DeepSeek，真实输出节选）：

> Q: 员工请事假需要提前几天申请？
> A: 手册未明确规定具体提前天数……事假申请有条件：员工只被允许在当年的年休假及
> 调休休完后申请事假。〔第23页·第四章 考勤和休假 > 第二节 休假 > 3. 事假〕

## 关键设计说明

- **分块**：同一节下短条款自动合并（如 `12.1.1~12.2`），块内保留条款标题行；
  每块前缀 `【章节路径】` 保持自含语义；跨页条款合并记录页范围。
- **检索增强**：dense(top20) + BM25(top20) → RRF(k=60) 融合取 top8（`--top-k` 可调）。
- **嵌入**：bge 官方要求查询侧加指令前缀；`HF_HUB_OFFLINE=1` 离线加载避免联网卡死。
- **引用溯源**：System prompt 强制逐条标注〔页码·章节〕；LLM 失败自动降级展示检索片段。
- **表格**：正文提取按表格 bbox 去重，表格仅保留一份 Markdown 入库。

## 已知边界 / 后续

- 文本型 PDF 全量入库；图片/图表内容未做视觉理解（本机无 GPU，InternVL 多模态阶段预留）。
- 目前仅单轮事实问答；若需流式输出/网页界面/重排(bge-reranker)可增量扩展。
- 本机 `.env` 含密钥，勿提交到公开仓库（项目无 git 仓库）。
