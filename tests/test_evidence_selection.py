import unittest
from ragcore import evidence


class SelectionTests(unittest.TestCase):
    def select(self, fused, facets, k):
        function = getattr(evidence, 'select_evidence', None)
        self.assertIsNotNone(function, 'Missing facet evidence selection')
        return function(fused, facets, k)

    def test_distinct_facet_survives_popular_duplicate_topic(self):
        result = self.select(['a','b','c','d','e','f','g','h','salary'],
                             [['a','b','c','d'], ['salary','a']], 8)
        self.assertIn('salary', result)
        self.assertEqual(result[:3], ['a','b','c'])

    def test_duplicates_bounds_and_short_lists(self):
        result = self.select(['a','b','c'], [['a','b'], ['a','c']], 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result), len(set(result)))
        self.assertEqual(self.select([], [[]], 8), [])

    def test_without_supplement_retains_fused_order(self):
        self.assertEqual(self.select(['a','b','c'], [['b','c']], 2), ['a','b'])

    def test_single_channel_first_hit_is_not_buried_by_agreement(self):
        function = getattr(evidence, 'lexical_anchor_order', None)
        self.assertIsNotNone(function)
        self.assertEqual(function(['shared1','shared2','unique'], {'unique':1,'shared2':2}),
                         ['unique','shared1','shared2'])


if __name__ == '__main__':
    unittest.main()
