"""Offline mixed-judge contracts; fixtures are synthetic Flash copies of smoke data."""
import copy
import importlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evaluation import run_generation_fusion_experiment as runner


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT/'evaluation/generation_fusion_smoke_2026-10-13'


class MixedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)/'parent'
        self.flash = Path(self.temp.name)/'flash'
        self.output = Path(self.temp.name)/'combined'
        shutil.copytree(SMOKE, self.parent)
        self.flash.mkdir()
        self.parent_manifest = json.loads((self.parent/'manifest.json').read_text(encoding='utf8'))
        self.flash_manifest = copy.deepcopy(self.parent_manifest)
        self.flash_manifest['judge']['model'] = 'deepseek-flash'
        self.flash_manifest['judge']['prompt_sha256'] = self.parent_manifest['judge']['prompt_sha256']['correctness']
        self.flash_manifest['parent_manifest'] = copy.deepcopy(self.parent_manifest)
        self.flash_manifest['parent_manifest_sha256'] = runner.file_hash(self.parent/'manifest.json')
        self.flash_manifest['parent_answers_sha256'] = runner.file_hash(self.parent/'answers.jsonl')
        self.flash_manifest['parent_files_sha256'] = {
            name: runner.file_hash(self.parent/name) for name in
            ('manifest.json','answers.jsonl','context_reviews.jsonl','faithfulness_reviews.jsonl','correctness_reviews.jsonl')}
        self.save(self.flash/'manifest.json', self.flash_manifest)
        self.save(self.flash/'parent_manifest.json', self.parent_manifest)
        self.correctness = runner.read_jsonl(self.parent/'correctness_reviews.jsonl')
        for row in self.correctness:
            row['model'] = 'deepseek-flash'
            for attempt in row['attempts']:
                if 'response_model' in attempt:
                    attempt['response_model'] = 'deepseek-flash'
        self.write_reviews()

    @staticmethod
    def save(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')

    def write_reviews(self):
        (self.flash/'correctness_reviews.jsonl').write_text(
            ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in self.correctness), encoding='utf8')

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluation.compose_generation_mixed_summary'),
                             'Offline mixed composer is missing')
        return importlib.import_module('evaluation.compose_generation_mixed_summary')

    def compose(self):
        return self.module().compose(self.parent, self.flash, self.output)

    def target(self):
        return next(r for r in self.correctness if r['dataset']=='original46' and r['variant']=='A')

    def mark_failed(self, row):
        row.update(status='failed',review=None,scores=None)
        row['attempts'][-1].update(status='failed',finish_reason='length')

    def test_all_models_are_named_and_parent_fingerprints_preserved(self):
        result = self.compose()
        manifest = json.loads((self.output/'manifest.json').read_text(encoding='utf8'))
        self.assertEqual(manifest['judge']['models_by_metric'], {
            'context':'deepseek-v4-pro','faithfulness':'deepseek-v4-pro','correctness':'deepseek-flash'})
        self.assertEqual(manifest['code_sha256'], self.parent_manifest['code_sha256'])
        self.assertEqual(manifest['reference_sha256'], self.parent_manifest['reference_sha256'])
        self.assertEqual(result['judge']['correctness']['model'], 'deepseek-flash')
        self.assertEqual(result['judge']['faithfulness']['model'], 'deepseek-v4-pro')
        self.assertEqual(result['supplemental_records'], 4)
        self.assertEqual(result['legacy_pro_correctness']['recorded'], 8)
        self.assertFalse(result['legacy_pro_correctness']['included_in_main_scores'])
        self.assertIn('composition_code_sha256', manifest)
        self.assertEqual(len(json.loads((self.output/'scored_answers.json').read_text(encoding='utf8'))), 8)

    def test_actual_flash_runner_manifest_interface_is_compatible(self):
        from evaluation import run_correctness_flash_review as flash_runner
        args=SimpleNamespace(parent=self.parent,
            source_dir=ROOT/'evaluation/guard_fusion_experiment_2026-10-13',
            references_dir=ROOT/'evaluation/generation_fusion_references_v1',workers=3,max_attempts=2)
        bundle=flash_runner.load_parent(args)
        manifest=flash_runner.build_manifest(args,bundle)
        self.save(self.flash/'manifest.json',manifest)
        result=self.compose()
        self.assertEqual(result['judge']['correctness']['model'],'deepseek-flash')

    def test_missing_flash_review_never_falls_back_to_old_pro_grade(self):
        target = self.target(); self.correctness.remove(target); self.write_reviews()
        result = self.compose()
        scored = json.loads((self.output/'scored_answers.json').read_text(encoding='utf8'))
        row = next(r for r in scored if runner.metrics.row_key(r)==runner.metrics.row_key(target))
        self.assertNotIn('grade', row)
        self.assertIsNone(row['answer_correctness'])
        self.assertEqual(result['judge']['correctness']['unreviewed_eligible'], 1)
        self.assertEqual(result['datasets']['original46']['variants']['A']['grade_denominator'], 0)
        self.assertEqual(result['datasets']['original46']['paired']['positive_metrics']['factual_f1']['A']['na_pair_count'], 1)

    def test_failed_flash_review_keeps_cost_without_reusing_pro_score(self):
        self.mark_failed(self.target()); self.write_reviews()
        result = self.compose()
        self.assertEqual(result['judge']['correctness']['failed'], 1)
        self.assertEqual(result['judge']['correctness']['tokens']['attempt_count'], 8)
        self.assertEqual(result['datasets']['original46']['variants']['A']['rubric']['all_case_pass_rate'], 0)

    def test_parent_pro_correctness_cost_is_separate_not_added_to_flash(self):
        target = self.target()
        target['review']['grade'] = 'partial'
        target['review']['FP'] = ['synthetic erroneous assertion']
        target['scores'] = runner.metrics.correctness_scores(target['review'], {
            'required_points':target['review']['required_checks'],
            'required_clarifications':target['review']['clarification_checks'],
            'forbidden_inferences':target['review']['forbidden_checks']})
        target['attempts'][-1]['raw_response'] = json.dumps(target['review'],ensure_ascii=False)
        self.write_reviews()
        result = self.compose()
        scored = json.loads((self.output/'scored_answers.json').read_text(encoding='utf8'))
        row = next(r for r in scored if runner.metrics.row_key(r)==runner.metrics.row_key(target))
        self.assertEqual(row['grade'], 'partial')
        self.assertEqual(row['factual_f1'], target['scores']['factual_f1'])
        self.assertEqual(result['judge']['correctness']['tokens']['attempt_count'], len(self.correctness))
        self.assertEqual(result['legacy_pro_correctness']['tokens']['attempt_count'], 8)

    def test_pro_row_in_flash_journal_is_rejected(self):
        self.target()['model']='deepseek-v4-pro'; self.write_reviews()
        with self.assertRaises(ValueError): self.compose()

    def test_wrong_response_model_is_rejected_even_with_flash_row_tag(self):
        self.target()['attempts'][-1]['response_model']='deepseek-v4-pro'; self.write_reviews()
        with self.assertRaises(ValueError): self.compose()

    def test_changed_flash_payload_or_raw_score_is_rejected(self):
        self.target()['scores']['factual_f1']=.123456; self.write_reviews()
        with self.assertRaises(ValueError): self.compose()

    def test_changed_flash_judge_payload_is_rejected(self):
        self.target()['messages'][1]['content']='{"question":"different"}'
        self.write_reviews()
        with self.assertRaises(ValueError): self.compose()

    def test_wrong_pro_context_model_is_rejected(self):
        path=self.parent/'context_reviews.jsonl'
        rows=runner.read_jsonl(path)
        rows[0]['model']='deepseek-flash'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
        # Refresh only the fixture identity so the actual model check is exercised.
        self.flash_manifest['parent_files_sha256']['context_reviews.jsonl']=runner.file_hash(path)
        self.save(self.flash/'manifest.json',self.flash_manifest)
        with self.assertRaises(ValueError): self.compose()

    def test_flash_key_without_a_completed_generation_is_rejected(self):
        self.target()['qid']='unknown-question'; self.write_reviews()
        with self.assertRaises(ValueError): self.compose()

    def test_partial_paired_flash_completion_keeps_independent_context_denominator(self):
        self.mark_failed(self.target()); self.write_reviews()
        result=self.compose()
        dataset=result['datasets']['original46']
        self.assertEqual(dataset['paired']['positive_metrics']['factual_f1']['A']['paired_run_count'],0)
        self.assertEqual(dataset['context_evaluation']['paired']['metrics']['context_recall']['A']['question_count'],1)

    def test_duplicate_flash_review_is_rejected(self):
        self.correctness.append(copy.deepcopy(self.target()));self.write_reviews()
        with self.assertRaises(ValueError): self.compose()

    def test_both_case_groups_use_question_weighted_existing_formulas(self):
        result=self.compose()
        for dataset in result['datasets'].values():
            for metric in dataset['paired']['positive_metrics'].values():
                if metric['A']['mean'] is not None:
                    self.assertAlmostEqual(metric['delta_D_minus_A'],metric['D']['mean']-metric['A']['mean'])
        manifest=json.loads((self.output/'manifest.json').read_text(encoding='utf8'))
        hashes=manifest['source_file_sha256']
        self.assertEqual(hashes[str((self.parent/'answers.jsonl').resolve())],runner.file_hash(self.parent/'answers.jsonl'))
        self.assertEqual(hashes[str((self.flash/'correctness_reviews.jsonl').resolve())],runner.file_hash(self.flash/'correctness_reviews.jsonl'))

    def test_changed_parent_manifest_identity_is_rejected(self):
        self.flash_manifest['reference_sha256']['original46']='changed'
        self.save(self.flash/'manifest.json', self.flash_manifest)
        with self.assertRaises(ValueError): self.compose()

    def test_changed_parent_journal_is_rejected_before_composition(self):
        with (self.parent/'answers.jsonl').open('a',encoding='utf8') as stream:
            stream.write('{}\n')
        with self.assertRaises(ValueError): self.compose()

    def test_existing_result_is_never_silently_overwritten(self):
        self.compose()
        with self.assertRaises(ValueError): self.compose()

    def test_reference_mutation_during_aggregation_is_rejected_before_write(self):
        refs=Path(self.temp.name)/'references'
        refs.mkdir()
        for ds in runner.DATASETS:
            shutil.copyfile(ROOT/f'evaluation/generation_fusion_references_v1/{ds}.json',refs/f'{ds}.json')
        summarize=runner.metrics.summarize_dataset
        def mutate(*args,**kwargs):
            result=summarize(*args,**kwargs)
            with (refs/'original46.json').open('a',encoding='utf8') as stream: stream.write(' ')
            return result
        with patch.object(runner.metrics,'summarize_dataset',side_effect=mutate):
            with self.assertRaises(ValueError):
                self.module().compose(self.parent,self.flash,self.output,references_dir=refs)
        self.assertFalse(self.output.exists())

    def test_core_code_identity_is_rechecked_after_aggregation(self):
        file_hash=runner.file_hash
        summarize=runner.metrics.summarize_dataset
        changed={'value':False}
        def mutate(*args,**kwargs):
            result=summarize(*args,**kwargs);changed['value']=True;return result
        def changed_hash(path):
            if changed['value'] and Path(path).resolve()==ROOT/'evaluation/generation_metrics.py':
                return 'changed-during-composition'
            return file_hash(path)
        with patch.object(runner.metrics,'summarize_dataset',side_effect=mutate),patch.object(runner,'file_hash',side_effect=changed_hash):
            with self.assertRaises(ValueError): self.compose()
        self.assertFalse(self.output.exists())


if __name__=='__main__': unittest.main()
