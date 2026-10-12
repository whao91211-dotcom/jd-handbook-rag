# Approved rerank evaluation plan

Authority: the user-approved plan in this chat, 2026-10-12.

1. Add a local, pinned CPU FP32 BGE cross-encoder (batch 8, pair length 512), stable ranking and a common protected 40-candidate pool. Test limits, ties, non-finite scores, truncation and failure propagation.
2. Add opt-in global/guarded reranking to retrieval. Keep defaults byte-for-byte compatible; preserve source text, quotas and lexical anchors. Test with controlled retrieval inputs.
3. Freeze the current 123-chunk corpus and 46/36 development datasets, verify the actual index, remap P32 by its source quote, then run A/B/C three times each with rotating order. Record rankings, candidate scores, selection reasons, truncation, context coverage and cold/warm timing. No generation calls.
4. Publish per-question artifacts and a bounded report. Review the full change, verify the suite, commit each coherent batch, fast-forward the original feature branch and push it. Preserve its unrelated .gitignore change.

Defaults: BAAI/bge-reranker-base revision 2cfc18c9415c912f9d8155881c133215df768a70; CPU threads fixed to min(4, logical CPUs) for the experiment; no latency ceiling, no score thresholds or tuning. B/C score the original question only, using the same deterministic pool. C sorts within facets and reapplies protected lexical anchors and existing quotas.

Review focus: disabled-path compatibility; no gold reaching runtime; genuinely identical B/C pools; no silent fallback; correct P32 mapping; all failed runs excluded from success aggregates; warmed stage timing distinct from full selection/diagnostic overhead; negative questions are not scored as answer rejection.
