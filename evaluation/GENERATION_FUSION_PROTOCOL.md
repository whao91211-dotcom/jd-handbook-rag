# Guard / Guard+BGE 生成对照协议

本轮检验原 Guard（A）与 50/50 Guard/BGE 秩融合（D）是否改变实际生成质量。它是开发/回归实验；参考标注由 Codex 依据冻结来源编写并交叉复核，尚无完整用户人工裁定。回答模型 Flash 和评审模型 Pro 属于同一 DeepSeek 家族，可能共享偏差。

## 固定输入与测量范围

复用 `guard_fusion_experiment_2026-10-13` 中已审计的 A/D、repeat=0 证据选取，123 块语料，最多40候选、8块选取。每题使用现有 `prepare_context` 构建6000字符预算内实际送入模型的资料，完整片段原文不改写。原始46题与模拟开发36题分开报告，两组每题新生成3次，共492次，组内交替执行，每次最多3个问题并发。

生成仍使用现有 evidence 提示词、temperature=0.2、low thinking 和重试策略。生成器只收到问题与证据，评测标签不进入生成。正式输出保留提示词、来源映射、原始 API 返回、应用层请求记录、时间、用量及失败；SDK 内部重试未逐个记录，未返回用量的请求计为未知。所有时间是生成/评审时间，不包含本轮未重新执行的召回及 CPU 重排。

## 指标口径

这些是透明自定义指标，不是直接运行 Ragas 得到的官方分数。

- Faithfulness：答案事实中实际输入上下文支持的比例。审查员不接收参考答案；无事实声明记 NA，失败或截断答案不评分。范围性“当前资料未明确”与“整本手册不存在某制度”的全局断言区别审查。
- 引用支持率：在全部事实声明中，实际引用且所引来源真实支持该声明的比例；未引用也计为未支持。合法来源编号不代表语义支持。
- Context Recall：冻结参考事实条目被实际上下文支持的比例。一些条目包含多项条件，不宣称严格逐原子事实等权。Context Precision：按上下文顺序计算相关片段的平均精度（AP）。只在正题计算，每题每组一次，独立于生成成功率。
- Factual F1：`2TP/(2TP+FP+FN)`；只扣必要事实及条件遗漏，不把参考未列的有来源相关细节机械算错。
- Answer Correctness：正题 `0.75 * Factual F1 + 0.25 * BGE 原始余弦相似度`。它是辅助复合分数；负题和含糊题单独评必要拒答/澄清行为。
- BERTScore：官方 bert-score 0.3.13，中文 BERT 第8层，CPU、batch8、idf=false、rescale=false。移除来源标记并合并空白；保留否定、数字。高相似度不能证明事实正确。

答案指标先平均同题可配对成功重复，再等权平均题目；每项都报告可评分题数和 NA。行为通过率同时报告已评审分母和全部计划运行分母。失败、未执行、评审无效不混入成功结果。上下文指标有独立正题分母。原始题35正题，模拟开发题9正题；其他38题用行为诊断。

## 本地依赖与固定模型

生产依赖不改。将 `requirements-generation-eval.txt` 用 `pip install --target <ignored-directory> --no-deps -r ...` 安装到独立评测目录；重用已有 torch/transformers/sentence-transformers。运行时版本写入 manifest。

- BERT：`google-bert/bert-base-chinese`，revision `8f23c25b06e129b6c986331a13d8d025a92cf0ea`。
- BGE：`BAAI/bge-small-zh-v1.5`，revision `7999e1d3359715c523056ef9478215996d62a620`。

`semantic_model_artifacts_v1.json` 固定权重、配置和词表文件指纹；权重及本地配置不提交。BERT 版本通过官方 Hub API 核对，官方下载网络失败后以 HF CLI 经公共镜像下载，权重匹配固定 LFS ETag；其他文件是此次下载后的本地固定指纹，不宣称逐文件独立官方审计。BGE 重用已有固定缓存。两个编码器原生512 token 上限，逐条记录未截断token数和模型上限，据此报告超过上限的数量；完整回答保留。文本冻结哈希仅规范 CRLF 为 LF，模型文件计算原始字节哈希。

## 复现顺序

在仓库根目录，使用同一 Python 环境；`<env>` 是已有服务配置文件，不输出或提交其内容。

```powershell
python evaluation/run_generation_fusion_smoke.py generate --output <smoke-directory> --env-file <env>
python evaluation/run_generation_fusion_smoke.py judge --output <smoke-directory> --env-file <env>
# 对 smoke-directory 运行下述语义评分，再用 smoke 脚本 summarize。
python evaluation/run_generation_fusion_experiment.py generate --env-file <env>
python evaluation/run_generation_fusion_experiment.py judge --resume --env-file <env>
python evaluation/run_generation_semantic_metrics.py evaluation/generation_fusion_experiment_2026-10-13 --dependency-dir <ignored-directory> --bert-model <local-pinned-BERT> --embed-model <local-pinned-BGE>
python evaluation/run_generation_fusion_experiment.py summarize --resume --env-file <env>
python evaluation/verify_generation_fusion_results.py evaluation/generation_fusion_experiment_2026-10-13
```

冒烟固定 P17、C-P06、C-N01、C-A01，各组一次；只检查真实链路契约，不纳入正式比较。正式生成和评审可用 `--resume` 继续未完成项；已记录失败不静默覆盖。源语料、参考、代码或设置发生变化时必须使用新目录。仅结果提交导致 Git HEAD 改变时允许恢复，并保留测量时版本。

结论必须包含 A/D 同期基线、新值、绝对变化、失败/NA、改善及退化实例、生成与评审用量。不得以本轮数据宣称真实用户准确率、人工审核通过率或在线端到端延迟提升；不因结果自动改变默认模式。
