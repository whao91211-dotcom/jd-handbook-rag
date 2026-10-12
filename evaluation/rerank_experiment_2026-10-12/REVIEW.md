# Review and verification record

Independent read-only code review covered 108f844..78357eb, against the approved plan. No Critical, Important or Minor findings were reported. The reviewer checked gold isolation, disabled compatibility, shared candidate pools, quotas/anchors, stable score ties, paired quality denominators, failure propagation and the installed sentence-transformers 6.0.1 API. Its 14 lightweight helper/evaluator tests passed.

Real-model smoke subsequently exposed huggingface-hub 1.29's full-repository offline completeness check. Commit bc14445 limits snapshot validation to the six required inference files. A regression test first reproduced the failure, then the full 76-test suite passed. Verified real A/B/C scoring followed on the local model. All model/tokenizer files were compared to official fixed-revision hashes.

Full live-corpus disabled compatibility compared default retrieval and explicit rerank='none' with the original 108f844 implementation for all 82 questions. Every complete result dictionary matched, including full text and source metadata. See baseline_compatibility.json.

The independent reviewer did not adjudicate the semantic completeness of existing gold annotations, answer quality, rejection behavior or production generalization. This experiment retains those annotation boundaries. The sole missing original reference (P32) is remapped from its original page and exact normalized source quote, independently confirmed by the reviewer. Real latency/quality claims are supported only by the completed frozen runs and report, not by static review.

The six-file snapshot fix was verified with tests and real inference; it was not sent for a second independent code review.

Final actual experiment completed all 738 runs successfully (original46: 138 per variant; development36: 108 per variant). Independent artifact audit recomputed all quality metrics, checked identical pools and exact score/rank stability across repeats, pool/result bounds, context completeness and frozen source hashes. The post-experiment suite passed 76/76 in 29.623 seconds. The additional offline audit tool was verified against the completed artifacts; it does not call retrieval or models.

Both reranked variants improve mean ranking metrics, but lose one of the three required evidence chunks on development case C-P06. Full coverage falls from 9/9 to 8/9, with the same drop after context construction. B/C therefore do not qualify for the next answer experiment under the approved criterion. Runtime parameters and labels were not tuned after observing results.

Packaging audit detected Windows Git newline conversion: measured source files contain a mix of existing CRLF and added LF, while Git stores normalized LF. The manifest remains unchanged. Portable content hashes are derived only after matching the measured byte hashes, normalize CRLF to LF only, and are checked on integration. This is an artifact verification adjustment and does not change the frozen retrieval implementation or results.
