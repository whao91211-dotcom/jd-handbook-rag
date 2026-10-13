import unittest

from ragcore import reranker


class RankFusionTests(unittest.TestCase):
    def test_endpoints_keep_each_complete_ranking(self):
        pool, guard, semantic = ['a', 'b', 'c'], ['c', 'b', 'a'], ['b', 'a', 'c']
        for weight, expected in [(0., semantic), (1., guard)]:
            order, scores = reranker.fuse_orders(pool, guard, semantic, weight)
            self.assertEqual(order, expected)
            self.assertEqual(set(scores), set(pool))

    def test_rule_priority_can_rescue_low_semantic_evidence(self):
        pool = [str(i) for i in range(20)]
        guard = ['14'] + [iid for iid in pool if iid != '14']
        order, _ = reranker.fuse_orders(pool, guard, pool, .5)
        self.assertIn('14', order[:8])
        self.assertEqual(set(order), set(pool))

    def test_fusion_tie_keeps_input_rrf_order(self):
        order, scores = reranker.fuse_orders(['b', 'a'], ['a', 'b'], ['b', 'a'], .5)
        self.assertEqual(scores['a'], scores['b'])
        self.assertEqual(order, ['b', 'a'])

    def test_partial_or_duplicate_rankings_are_rejected(self):
        for guard, semantic in [(['a'], ['a', 'b']), (['a', 'a'], ['a', 'b']),
                                (['a', 'b'], ['a', 'c'])]:
            with self.assertRaises(ValueError):
                reranker.fuse_orders(['a', 'b'], guard, semantic, .5)

    def test_weight_must_be_finite_and_between_zero_and_one(self):
        for weight in [-.1, 1.1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                reranker.fuse_orders(['a'], ['a'], ['a'], weight)
