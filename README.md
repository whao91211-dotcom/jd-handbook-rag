# 京东员工手册 RAG

以《京东集团员工手册.pdf》为语料的问答实验项目。PDF 提取、分块、嵌入和检索在本地 CPU 上运行；回答生成调用 OpenAI 兼容 API。项目重点是可追溯引用、检索消融，以及对高风险员工问题的失败模式评测。

## 管线

```text
PDF（51 页）→ 逐页正文和表格提取 → 125 个带页码与章节路径的文本块
           → bge-small-zh-v1.5 向量索引 + jieba/BM25
           → dense、BM25 或 RRF 检索 → 带来源引用的回答
```

主要实现位于 `ragcore/`：`extract.py`、`chunker.py`、`ingest.py`、`retriever.py`、`llm.py`。数据与评测报告位于 `data/eval/`。`data/chroma/` 是可重建的本地索引，未提交到 Git。

## 本地运行

已验证的环境为 Python 3.10、CPU、预先安装 ChromaDB、sentence-transformers、PyTorch 和 OpenAI SDK 的 conda 环境；其余补充依赖见 `requirements-internvl.txt`。嵌入模型需要预先缓存到本机：当前实现默认启用 Hugging Face 离线模式。复制 `.env.example` 为 `.env` 并配置生成模型的 API 参数；`.env` 不会提交。

在项目根目录运行：

```powershell
python -m ragcore.extract
python -m ragcore.chunker
python -m ragcore.ingest
python ask.py "员工请事假需要提前几天申请？"
python ask.py "加班费怎么计算？" --no-llm
```

运行评测（红队全量运行会调用生成 API；`--rescore` 只重评已保存的回答）：

```powershell
python -m ragcore.eval_ir
python -m ragcore.eval_robust
python -m ragcore.redteam
python -m ragcore.redteam --rescore
```

## 已有评测

| 层次 | 数据和结果 | 解释 |
|---|---|---|
| 检索 golden set | 46 条已复核标注：35 条正样本、11 条负样本 | 题目主要由语料生成，词面重合可能使 BM25 占优；负样本中的困难样本不足 |
| 检索消融 | 正样本 Recall@3：BM25 97.14%、RRF 91.43%、dense 77.14%；MRR 分别为 0.8762、0.8131、0.7049 | 在当前数据上，BM25 优于融合；不能据此声称融合一定更好 |
| 输入扰动 | 截短问法的 Recall@5：BM25 97.14%、RRF 85.71%、dense 77.14% | 机械扰动测试，尚不能代表真实用户分布 |
| 红队问答 | 21 条风险用例的规则初筛 21/21；空回答 0、生成异常 0 | 规则只检查已知风险模式；**不是人工判定的回答准确率** |

完整结果和逐题输出见 `data/eval/report_ir.md`、`data/eval/robust_report.md`、`data/eval/redteam_report.md`。红队回答的 176 处引用中，页码不存在的引用为 0；有 1 处把两个前言页塞进同一引用标记，触发章节路径告警。规则通过不代表回答事实正确，仍需逐条核对。

另有 16 条待复核的拟真问法与困难负样本，见 `data/eval/challenge_review_v1.md`。它们尚未并入上述正式评测，也不被称为真实员工提问。

## 设计取舍与当前边界

- 检索支持 dense、BM25 和 RRF 对照。现有 golden set 显示 BM25 更强，后续应先补充真实改写与困难负样本，再决定是否调整融合策略。
- 回答附页码及章节引用；对辞职、解雇、赔偿、纪律等高风险问题提示核实正式渠道，不代写辞职申请或协议。文档生效日期不等于其当前仍有效。
- 生成采用自适应输出预算，红队重跑未出现空答案或截断兜底；目前没有独立的端到端人工正确率或引用支持率标注。
- 只处理文本型 PDF 及提取出的表格，没有图像理解、多轮对话或权限控制。公开展示前需确认手册的分享权限。
