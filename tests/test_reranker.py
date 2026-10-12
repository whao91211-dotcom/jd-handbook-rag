import importlib
import math
import unittest
from unittest.mock import patch


class RerankerTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('ragcore.reranker')
        except ModuleNotFoundError:
            self.fail('Local reranker module is missing')

    def test_pool_keeps_low_rank_baseline_and_anchors_within_cap(self):
        pool = self.module().candidate_pool([str(i) for i in range(70)], ['69', '68'], ['67', '69'])
        self.assertEqual(len(pool), 40)
        self.assertEqual(len(set(pool)), 40)
        self.assertTrue({'69', '68', '67'} <= set(pool))
        self.assertEqual(pool[-3:], ['67', '68', '69'])

    def test_small_pool_is_deduplicated_in_rrf_order(self):
        self.assertEqual(self.module().candidate_pool(['a', 'b', 'c'], ['c'], ['c']), ['a', 'b', 'c'])

    def test_same_score_uses_rrf_tie_order(self):
        self.assertEqual(self.module().score_order(['a', 'b', 'c'], {'a': 1., 'b': 2., 'c': 2.}), ['b', 'c', 'a'])

    def test_invalid_scores_fail_instead_of_silently_falling_back(self):
        module = self.module()
        for scores in ({'a': math.nan}, {'a': math.inf}, {}):
            with self.assertRaises(ValueError):
                module.score_order(['a'], scores)

    def test_scoring_empty_pool_does_not_load_model(self):
        module = self.module()
        with patch.object(module, 'get_model', side_effect=AssertionError('unnecessary model load')):
            self.assertEqual(module.score_candidates('q', []), ({}, {}))

    def test_scoring_records_untruncated_pair_length_and_keeps_text(self):
        module = self.module()
        class FakeTokenizer:
            def __call__(self, queries, texts, **kwargs):
                self.asserted = (queries, texts, kwargs)
                return {'input_ids': [[0] * 513, [0] * 12]}
        class FakeModel:
            tokenizer = FakeTokenizer()
            def predict(self, pairs, **kwargs):
                self.received = (pairs, kwargs)
                return [2., -1.]
        model = FakeModel()
        docs = [{'id': 'a', 'text': '完整原文'}, {'id': 'b', 'text': '补充原文'}]
        with patch.object(module, 'get_model', return_value=model):
            scores, lengths = module.score_candidates('原问题', docs)
        self.assertEqual(scores, {'a': 2., 'b': -1.})
        self.assertEqual(lengths, {'a': 513, 'b': 12})
        self.assertEqual(docs[0]['text'], '完整原文')
        self.assertEqual(model.received[0][0], ('原问题', '完整原文'))

    def test_model_failure_propagates(self):
        module = self.module()
        with patch.object(module, 'get_model', side_effect=RuntimeError('missing local weights')):
            with self.assertRaisesRegex(RuntimeError, 'missing local weights'):
                module.score_candidates('q', [{'id': 'a', 'text': 'text'}])

    def test_local_snapshot_needs_only_the_downloaded_inference_files(self):
        module = self.module()
        required = {'config.json', 'model.safetensors', 'sentencepiece.bpe.model',
                    'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json'}
        def snapshot(*args, **kwargs):
            if not kwargs.get('local_files_only') or set(kwargs.get('allow_patterns', [])) != required:
                raise RuntimeError('Unused repository artifacts requested from incomplete offline snapshot')
            return 'local-verified-snapshot'
        model = object()
        module.get_model.cache_clear()
        self.addCleanup(module.get_model.cache_clear)
        with patch('huggingface_hub.snapshot_download', side_effect=snapshot), \
             patch('sentence_transformers.CrossEncoder', return_value=model):
            self.assertIs(module.get_model(), model)


if __name__ == '__main__':
    unittest.main()
