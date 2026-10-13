import unittest
from pathlib import Path
import tempfile
import json
from evaluation.run_generation_semantic_metrics import assert_unchanged, clean_text, model_hashes, read_snapshot, select_pairs, validate_model_hashes


class SemanticInputTests(unittest.TestCase):
    def setUp(self):
        self.refs = {('development36', 'q'): {'positive': True, 'reference_answer': '事假无餐补。'},
                     ('development36', 'n'): {'positive': False, 'reference_answer': '需确认。'}}
        self.answer = dict(dataset='development36', qid='q', variant='D', repeat=0,
                           status='complete', answer='事假不发餐补。〔来源1〕')

    def test_only_completed_positive_answers_receive_similarity(self):
        failed = dict(self.answer, repeat=1, status='failed', answer=None)
        negative = dict(self.answer, qid='n')
        pairs = select_pairs([self.answer, failed, negative], self.refs)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['reference'], '事假无餐补。')
        self.assertEqual(pairs[0]['candidate'], '事假不发餐补。')

    def test_duplicate_run_keys_cannot_be_scored_twice(self):
        with self.assertRaises(ValueError):
            select_pairs([self.answer, self.answer], self.refs)

    def test_complete_empty_answer_is_invalid(self):
        with self.assertRaises(ValueError):
            select_pairs([dict(self.answer, answer='')], self.refs)

    def test_unknown_reference_cannot_silently_disappear(self):
        with self.assertRaises(ValueError):
            select_pairs([dict(self.answer, qid='unknown')], self.refs)

    def test_citation_cleanup_retains_negation_and_numbers(self):
        self.assertEqual(clean_text('不发15元餐补。〔来源1〕\n  需审批。'), '不发15元餐补。 需审批。')

    def test_cache_parent_does_not_hide_model_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)/'.cache'/'model'
            model.mkdir(parents=True)
            (model/'config.json').write_text('{}')
            (model/'.cache').mkdir()
            (model/'.cache'/'download-lock').write_text('internal')
            self.assertEqual(set(model_hashes(model)), {'config.json'})

    def test_wrong_local_model_cannot_be_reported_as_pinned(self):
        with self.assertRaises(ValueError):
            validate_model_hashes({'bert': {'model.safetensors': 'wrong'}, 'bge': {}})

    def test_tokenizer_tampering_is_not_hidden_by_correct_weights(self):
        spec = Path(__file__).resolve().parents[1]/'evaluation/semantic_model_artifacts_v1.json'
        artifacts = json.loads(spec.read_text())
        hashes = {k: v['files_sha256'] for k, v in artifacts['models'].items()}
        hashes['bert']['tokenizer.json'] = 'changed'
        with self.assertRaises(ValueError):
            validate_model_hashes(hashes)

    def test_input_mutation_cannot_be_attached_to_old_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'answers.jsonl'
            path.write_bytes(b'{"answer":"original"}\n')
            text, expected = read_snapshot(path)
            path.write_bytes(b'{"answer":"changed"}\n')
            self.assertIn('original', text)
            with self.assertRaises(ValueError):
                assert_unchanged(path, expected)


if __name__ == '__main__':
    unittest.main()
