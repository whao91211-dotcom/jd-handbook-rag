import unittest
from unittest.mock import patch

from ragcore import retriever, reranker


class RetrievalRerankTests(unittest.TestCase):
    def setUp(self):
        ids = ['a', 'b', 'c', 'd', 'e', 'f', 'pay', 'lex', 'exit']
        self.docs = ['原文-' + iid for iid in ids]
        self.metas = [{'pages': '1,2', 'page_start': 1, 'path': '章节', 'clause': '条款', 'char_len': 4} for _ in ids]
        def dense(q, n):
            if q == '发薪时间 发薪日 薪酬核算周期':
                return {'pay': 1, 'a': 2, 'b': 3}
            if q == '离职工资 交接 结算':
                return {'exit': 1, 'a': 2}
            return {iid: i + 1 for i, iid in enumerate(ids[:6])}
        def lexical(q, *args):
            if q == '发薪时间 发薪日 薪酬核算周期':
                return {'lex': 1, 'pay': 2, 'a': 3}
            return dense(q, len(ids))
        self.patches = [patch.object(retriever, '_load_corpus', return_value=(ids, self.docs, self.metas)),
                        patch.object(retriever, '_dense_ranks', side_effect=dense),
                        patch.object(retriever, '_bm25_ranks', side_effect=lexical)]
        for p in self.patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])
        self.question = '离职工资什么时候发，如何结算？'
        self.scores = dict(zip(ids, [6., 5., 4., 7., 9., 8., 1., -2., 0.]))

    def fake_score(self, query, candidates):
        if query != self.question:
            raise AssertionError('Scoring must use the original question')
        return ({c['id']: self.scores[c['id']] for c in candidates}, {c['id']: 12 for c in candidates})

    def test_disabled_preserves_historical_selection_and_schema(self):
        with patch.object(reranker, 'score_candidates', side_effect=AssertionError('disabled scorer called')):
            hits = retriever.retrieve(self.question, strategy='guarded', rerank='none')
        self.assertEqual([h['id'] for h in hits], ['a', 'b', 'c', 'lex', 'pay', 'exit', 'd', 'e'])
        self.assertNotIn('rerank_score', hits[0])
        self.assertEqual(hits[0]['pages'], [1, 2])

    def test_global_and_guarded_share_pool_and_preserve_full_sources(self):
        traces = []
        with patch.object(reranker, 'score_candidates', side_effect=self.fake_score):
            for method in ('global', 'guarded'):
                trace = {}
                hits = retriever.retrieve(self.question, strategy='guarded', rerank=method, diagnostics=trace)
                traces.append(trace)
                self.assertEqual(len(hits), 8)
                self.assertEqual(len(set(h['id'] for h in hits)), 8)
                self.assertEqual(hits[0]['text'], '原文-e')
                self.assertEqual(hits[0]['pages'], [1, 2])
                self.assertEqual(hits[0]['path'], '章节')
                self.assertEqual(hits[0]['rerank_score'], 9.)
        self.assertEqual(traces[0]['candidate_ids'], traces[1]['candidate_ids'])
        self.assertEqual(traces[0]['selected_ids'], ['e', 'f', 'd', 'a', 'b', 'c', 'pay', 'exit'])
        self.assertTrue({'lex', 'exit'} <= set(traces[1]['selected_ids']))
        self.assertEqual(traces[1]['selection_reasons']['lex'], 'lexical_anchor:1')

    def test_same_scores_keep_rrf_order_inside_facets(self):
        def equal(query, candidates):
            return ({c['id']: 1. for c in candidates}, {c['id']: 12 for c in candidates})
        with patch.object(reranker, 'score_candidates', side_effect=equal):
            trace = {}
            hits = retriever.retrieve(self.question, strategy='guarded', rerank='global', diagnostics=trace)
        self.assertEqual([h['id'] for h in hits], trace['candidate_ids'][:8])

    def test_scoring_failure_is_not_a_successful_baseline_result(self):
        with patch.object(reranker, 'score_candidates', side_effect=RuntimeError('model failed')):
            with self.assertRaisesRegex(RuntimeError, 'model failed'):
                retriever.retrieve(self.question, strategy='guarded', rerank='global')

    def test_invalid_method_and_oversized_rerank_k_are_rejected(self):
        for kwargs in ({'rerank': 'other'}, {'rerank': 'global', 'k': 41}):
            with self.assertRaises(ValueError):
                retriever.retrieve(self.question, **kwargs)

    def test_fusion_uses_two_complete_orders_before_final_cut(self):
        traces = {}
        with patch.object(reranker, 'score_candidates', side_effect=self.fake_score):
            for method in ('none', 'global', 'guarded', 'fusion'):
                traces[method] = {}
                hits = retriever.retrieve(self.question, strategy='guarded', rerank=method,
                                          diagnostics=traces[method])
        d = traces['fusion']
        self.assertTrue(all(t['candidate_ids'] == d['candidate_ids'] for t in traces.values()))
        self.assertEqual(len(d['guard_order']), 9)
        self.assertEqual(d['guard_order'], ['a', 'b', 'c', 'lex', 'pay', 'exit', 'd', 'e', 'f'])
        self.assertEqual(d['semantic_order'], ['e', 'f', 'd', 'a', 'b', 'c', 'pay', 'exit', 'lex'])
        self.assertEqual(set(d['fusion_order']), set(d['candidate_ids']))
        self.assertEqual(d['selected_ids'], d['fusion_order'][:8])
        self.assertEqual(hits[0]['text'], '原文-' + hits[0]['id'])
        self.assertEqual(hits[0]['pages'], [1, 2])
        self.assertEqual(hits[0]['fusion_score'], d['fusion_scores'][hits[0]['id']])

    def test_fusion_weight_endpoints_and_score_failure(self):
        with patch.object(reranker, 'score_candidates', side_effect=self.fake_score):
            for weight, key in [(0., 'semantic_order'), (1., 'guard_order')]:
                trace = {}
                hits = retriever.retrieve(self.question, strategy='guarded', rerank='fusion',
                                          guard_weight=weight, diagnostics=trace)
                self.assertEqual([h['id'] for h in hits], trace[key][:8])
        with patch.object(reranker, 'score_candidates', side_effect=RuntimeError('failed')):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                retriever.retrieve(self.question, strategy='guarded', rerank='fusion')

    def test_fusion_rejects_missing_guard_strategy_and_bad_weight(self):
        with self.assertRaises(ValueError):
            retriever.retrieve(self.question, rerank='fusion')
        with self.assertRaises(ValueError):
            retriever.retrieve(self.question, strategy='guarded', rerank='fusion', guard_weight=2.)


if __name__ == '__main__':
    unittest.main()
