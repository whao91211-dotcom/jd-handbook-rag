import importlib
import unittest


class QueryTests(unittest.TestCase):
    def expand(self, query):
        try:
            module = importlib.import_module('ragcore.evidence')
        except ModuleNotFoundError:
            self.fail('Missing bounded query expansion feature')
        return module.expand_queries(query)

    def test_original_retained_and_queries_bounded(self):
        query = '电脑弄坏了，离职工资什么时候到账？'
        queries = self.expand(query)
        self.assertEqual(queries[0], query)
        self.assertLessEqual(len(queries), 3)
        self.assertEqual(len(queries), len(set(queries)))

    def test_damage_vocabulary_bridges_colloquial_query(self):
        queries = self.expand('办公设备被我弄坏了，需要赔钱吗？')
        self.assertTrue(any('追偿' in q and '财产' in q for q in queries[1:]))

    def test_payment_question_includes_payroll_facet(self):
        queries = self.expand('辞职交接完成，薪水何时到账？')
        self.assertTrue(any('发薪日' in q for q in queries[1:]))

    def test_unknown_and_empty_queries_are_not_guessed(self):
        self.assertEqual(self.expand('这个能怎么办？'), ['这个能怎么办？'])
        self.assertEqual(self.expand('  '), [])

    def test_scoped_expansion_retains_probation_condition(self):
        from ragcore import evidence
        self.assertTrue(hasattr(evidence, 'scoped_queries'))
        extra = evidence.scoped_queries('试用期迟到早退几次算不符合录用条件？')[1:]
        self.assertTrue(extra)
        self.assertTrue(all('试用期' in q for q in extra))


if __name__ == '__main__':
    unittest.main()
