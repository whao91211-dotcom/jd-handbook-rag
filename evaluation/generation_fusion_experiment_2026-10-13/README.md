# 原始生成与 Pro 评审记录

本目录保留492次正式生成（489成功、3失败）、164份Pro上下文评审、489份Pro忠实度记录和261份正题语义指标。失败仍保留。

Pro正确性分支在40份记录（34有效、6失败）时因HTTP402停止。用户随后指定新的Flash正确性评审；本目录summary.json/scored_answers.json是旧Pro正确性不完整分支的中间汇总，不是最终主报告。不要把40份分数与新Flash剩余分数混合。

最终五项指标对照、统一Flash正确性及结论见 ../generation_fusion_combined_2026-10-14/REPORT.md。该汇总重用本目录的上下文、忠实度和语义指标，全部成功答案使用独立Flash正确性记录。旧Pro正确性只单列用量。

CASE_REVIEW.md及ADDITIONAL_CASE_REVIEW.md为Codex来源诊断，分别说明盲选例复核与事后诊断范围，不是人类金标，不覆盖正式自动分数。context_quote_diagnostics.json仅验证冻结证据引文是否进入实际上下文，不等同语义Context Recall。

输入与代码版本见manifest.json；换行哈希规范为CRLF转LF，模型文件按原始字节。原始响应、推理、用量、失败与全部三次重复均保留。