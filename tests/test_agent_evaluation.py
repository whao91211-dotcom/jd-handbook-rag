import importlib
import importlib.util
import unittest


class AgentEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluation.run_agent_experiment'), 'Missing paired agent evaluation')
        self.module = importlib.import_module('evaluation.run_agent_experiment')

    def test_gold_labels_never_enter_runtime_and_context_coverage_is_separate(self):
        received = []
        def runner(question):
            received.append(question)
            return {'sources': [{'id': 'a', 'in_context': True}, {'id': 'b', 'in_context': False}],
                    'status': 'complete', 'answer': '规则', 'metrics': {'total_seconds': 1, 'total_tokens': 10}}
        case = {'qid': 'private-id', 'question': '病假工资', 'gold_chunk_ids': ['a', 'b'], 'expected': 'secret-gold'}
        row = self.module.run_case(case, 'agent', runner)
        self.assertEqual(received, ['病假工资'])
        self.assertTrue(row['gathered_full_coverage'])
        self.assertFalse(row['context_full_coverage'])
        self.assertEqual(row['context_ids'], ['a'])

    def test_empty_gold_is_not_counted_as_perfect_coverage(self):
        row = self.module.run_case({'qid': 'negative', 'question': '未知制度'}, 'quality',
            lambda question: {'sources': [], 'status': 'no_evidence', 'answer': '', 'metrics': {}})
        self.assertIsNone(row['context_full_coverage'])

    def test_incomplete_pairs_are_excluded_from_paired_success_metrics(self):
        rows = [{'qid': 'ok', 'profile': p, 'status': 'complete', 'metrics': {'total_seconds': t, 'total_tokens': n}}
                for p, t, n in [('quality', 2, 10), ('agent', 4, 30)]]
        rows += [{'qid': 'failed', 'profile': 'quality', 'status': 'complete', 'metrics': {'total_seconds': 1, 'total_tokens': 10}},
                 {'qid': 'failed', 'profile': 'agent', 'status': 'timeout', 'metrics': {'total_seconds': 120, 'total_tokens': None}}]
        report = self.module.summarize(rows)
        self.assertEqual(report['paired_complete_n'], 1)
        self.assertEqual(report['paired']['agent']['total_seconds_p50'], 4)
        self.assertEqual(report['profiles']['agent']['status_counts'], {'complete': 1, 'timeout': 1})
        self.assertEqual(report['profiles']['agent']['known_usage_n'], 1)


if __name__ == '__main__':
    unittest.main()
