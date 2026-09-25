# 京东员工手册 · 本地 RAG

基于《京东集团员工手册.pdf》（51 页 / 约 5 万字符 / 7 页表格）的本地检索增强问答项目。
检索与嵌入完全本地（CPU），仅生成环节调用 OpenAI 兼容 API（默认 DeepSeek）。

## 运行环境

- Python：3.10，CPU；先创建并激活独立环境。下方命令使用 `python`，无需本机绝对路径。
- 依赖：`pip install -r requirements.txt`。已在 Windows / Python 3.10 / CPU 环境验证。
- 嵌入模型：`BAAI/bge-small-zh-v1.5`（首次运行自动下载，dim=512）。
  已缓存模型时可设置 `RAG_OFFLINE=1` 禁止联网探测。

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
python -m ragcore.extract     # PDF→文本
python -m ragcore.chunker     # →分块
python -m ragcore.ingest      # →嵌入入库(可 --rechunk)

# 2) 问答
python ask.py "员工申请事假有哪些前提？"
python ask.py                 # 交互模式
python ask.py "法定节假日加班如何支付加班费？" --no-llm  # 仅检索

# 3) 检索评测（离线）
python -m ragcore.eval
```

## 评测结果

`ragcore/eval.py` 包含 11 个经原文核对的可回答问题，按证据页及片段计算检索指标。
本地重建结果：**Hit@1 81.8%、Hit@3 100%、MRR@3 0.909**；报告写入
`data/eval_report.json`。另列 2 个无具体答案的问题，供后续单独评测生成阶段的拒答能力。
这些数字只反映小规模检索集，不代表真实用户问题的答案准确率。

示例问答（DeepSeek，真实输出节选）：

> Q: 员工请事假需要提前几天申请？
> A: 手册未明确规定具体提前天数……事假申请有条件：员工只被允许在当年的年休假及
> 调休休完后申请事假。〔第23页·第四章 考勤和休假 > 第二节 休假 > 3. 事假〕

## 关键设计说明

- **分块**：同一节下短条款自动合并（如 `12.1.1~12.2`），块内保留条款标题行；
  每块前缀 `【章节路径】` 保持自含语义；跨页条款合并记录页范围。
- **检索增强**：dense(top20) + BM25(top20) → RRF(k=60) 融合取 top8（`--top-k` 可调）。
- **嵌入**：查询侧加 bge 检索指令前缀；可选 `RAG_OFFLINE=1` 离线加载。
- **引用溯源**：System prompt 要求逐条标注〔页码或页码范围·章节〕；LLM 失败自动降级展示检索片段。
- **表格**：正文提取按表格 bbox 去重，表格仅保留一份 Markdown 入库。

## 已知边界 / 后续

- 文本型 PDF 全量入库；图片/图表内容未做视觉理解（本机无 GPU，InternVL 多模态阶段预留）。
- 目前仅单轮事实问答；若需流式输出/网页界面/重排(bge-reranker)可增量扩展。
- 本机 `.env` 含密钥，勿提交到仓库。对外展示前还需确认手册的使用与公开许可。
