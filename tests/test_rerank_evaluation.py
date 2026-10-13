import importlib
import unittest


class EvaluationTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('evaluation.run_rerank_experiment')
        except ModuleNotFoundError:
            self.fail('Reproducible rerank experiment is missing')

    def test_missing_reference_is_mapped_by_source_quote_and_page(self):
        cases = [{'qid': 'changed', 'gold_chunk_ids': ['old'], 'gold_pages': [48], 'evidence_quote': '下属信息：绩效'}]
        chunks = [{'id': 'new', 'pages': [48], 'text': '下属信息： 绩效'},
                  {'id': 'wrong-page', 'pages': [47], 'text': '下属信息：绩效'}]
        mapped, audit = self.module().map_references(cases, chunks)
        self.assertEqual(mapped[0]['gold_chunk_ids'], ['new'])
        self.assertEqual(cases[0]['gold_chunk_ids'], ['old'])
        self.assertEqual(audit[0]['new_ids'], ['new'])

    def test_missing_or_ambiguous_mapping_cannot_be_scored_as_a_miss(self):
        case = {'qid': 'x', 'gold_chunk_ids': ['old'], 'gold_pages': [1], 'evidence_quote': 'rule'}
        for chunks in ([], [{'id': 'a', 'pages': [1], 'text': 'rule'}, {'id': 'b', 'pages': [1], 'text': 'rule'}]):
            with self.assertRaises(ValueError):
                self.module().map_references([case], chunks)

    def test_context_coverage_can_fall_even_when_retrieval_is_complete(self):
        metric = self.module().retrieval_metrics({'a', 'b'}, ['a', 'b', 'x'], ['a'])
        self.assertEqual(metric['recall_at8'], 1.)
        self.assertEqual(metric['full_at8'], 1.)
        self.assertEqual(metric['context_full'], 0.)

    def test_negative_has_no_invented_recall_or_rejection_score(self):
        self.assertEqual(self.module().retrieval_metrics(set(), ['a'], ['a']), {})

    def test_index_verification_rejects_text_drift(self):
        chunks = [{'id': 'a', 'text': 'rule', 'pages': [1], 'path': '章', 'clause_range': '', 'char_len': 4}]
        data = {'ids': ['a'], 'documents': ['changed'], 'metadatas': [{'pages': '1', 'page_start': 1, 'path': '章', 'clause': '', 'char_len': 4}]}
        with self.assertRaises(ValueError):
            self.module().verify_index(chunks, data)

    def test_failed_variant_excludes_question_from_all_quality_averages(self):
        rows = []
        for qid in ('good', 'failed'):
            for variant in ('A', 'B', 'C'):
                row = {'dataset': 'test', 'qid': qid, 'variant': variant, 'status': 'complete',
                       'selected_ids': ['a'], 'metrics': dict.fromkeys(self.module().METRICS, 1.),
                       'recall_seconds': .1, 'rerank_seconds': .2, 'total_seconds': .3, 'pair_tokens': {'a': 513}}
                if qid == 'failed' and variant == 'B':
                    row = {'dataset': 'test', 'qid': qid, 'variant': variant, 'status': 'failed'}
                rows.append(row)
        summary = self.module().summarize(rows, 1)['test']
        self.assertEqual(summary['paired_complete_cases'], 1)
        self.assertEqual(summary['variants']['A']['positive_cases'], 1)
        self.assertEqual(summary['variants']['B']['failed_runs'], 1)

    def test_timing_repetitions_do_not_multiply_quality_sample_size(self):
        module = self.module()
        rows = [{'dataset': 'test', 'qid': 'one', 'variant': variant, 'status': 'complete',
                 'selected_ids': ['a'], 'metrics': dict.fromkeys(module.METRICS, 1.),
                 'recall_seconds': .1, 'rerank_seconds': .2, 'total_seconds': .3, 'pair_tokens': {}}
                for _ in range(3) for variant in ('A', 'B', 'C')]
        summary = module.summarize(rows, 3)['test']
        self.assertEqual(summary['variants']['A']['positive_cases'], 1)
        self.assertEqual(summary['variants']['A']['successful_runs'], 3)

    def test_four_way_pairing_requires_fusion_success(self):
        module = self.module()
        rows = [{'dataset': 'test', 'qid': 'one', 'variant': variant, 'status': 'complete',
                 'selected_ids': ['a'], 'metrics': dict.fromkeys(module.METRICS, 1.),
                 'recall_seconds': .1, 'rerank_seconds': .2, 'total_seconds': .3, 'pair_tokens': {}}
                for variant in ('A', 'B', 'C', 'D')]
        summary = module.summarize(rows, 1, variants=module.FUSION_VARIANTS)['test']
        self.assertEqual(summary['paired_complete_cases'], 1)
        self.assertEqual(summary['variants']['D']['positive_cases'], 1)
        rows[-1]['status'] = 'failed'
        summary = module.summarize(rows, 1, variants=module.FUSION_VARIANTS)['test']
        self.assertEqual(summary['paired_complete_cases'], 0)
        self.assertEqual(summary['variants']['A']['positive_cases'], 0)


if __name__ == '__main__':
    unittest.main()
