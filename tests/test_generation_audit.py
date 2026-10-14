"""Audit real frozen smoke records without calling any model or API."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from evaluation import verify_generation_fusion_results as audit

SMOKE = audit.ROOT/'evaluation/generation_fusion_smoke_2026-10-13'


class GenerationAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)/'smoke'
        shutil.copytree(SMOKE, self.folder)

    def verify(self):
        with redirect_stdout(io.StringIO()):
            return audit.verify(self.folder)

    def mutate(self, filename, change):
        path=self.folder/filename
        data=json.loads(path.read_text(encoding='utf8'))
        change(data)
        path.write_text(json.dumps(data,ensure_ascii=False),encoding='utf8')

    def test_real_smoke_is_valid_without_api(self):
        self.assertTrue(self.verify()['audit_passed'])

    def test_duplicate_scored_record_is_rejected(self):
        self.mutate('scored_answers.json',lambda rows:rows.append(dict(rows[0])))
        with self.assertRaises(AssertionError): self.verify()

    def test_scored_answer_cannot_replace_original_journal(self):
        self.mutate('scored_answers.json',lambda rows:rows[0].update(answer='不同的答案'))
        with self.assertRaises(AssertionError): self.verify()

    def test_generation_denominator_cannot_hide_failure(self):
        self.mutate('summary.json',lambda data:data['datasets']['original46']['variants']['A'].update(failed=99999))
        with self.assertRaises(AssertionError): self.verify()

    def test_na_and_judge_completion_denominators_are_audited(self):
        self.mutate('summary.json',lambda data:data['judge']['faithfulness'].update(completion_rate_all_planned=.123456))
        with self.assertRaises(AssertionError): self.verify()

    def test_token_totals_are_audited(self):
        self.mutate('summary.json',lambda data:data['datasets']['original46']['variants']['A']['tokens'].update(prompt_tokens=99999))
        with self.assertRaises(AssertionError): self.verify()

    def test_removed_metric_cannot_skip_audit(self):
        self.mutate('summary.json',lambda data:data['datasets']['original46']['paired']['positive_metrics'].pop('faithfulness'))
        with self.assertRaises(AssertionError): self.verify()

    def test_context_na_counts_cannot_hide_unscored_evidence(self):
        self.mutate('summary.json',lambda data:data['datasets']['original46']['context_evaluation']['variants']['A']['positive_na_or_unreviewed'].update(context_recall=99999))
        with self.assertRaises(AssertionError): self.verify()

    def test_missing_or_failed_review_cannot_retain_stale_quality_scores(self):
        row=json.loads((self.folder/'scored_answers.json').read_text(encoding='utf8'))[0]
        key=audit.runner.metrics.row_key(row)
        records={kind:audit.runner.read_keyed(self.folder/f'{kind}_reviews.jsonl',include_repeat=kind!='context')
                 for kind in ('context','faithfulness','correctness')}
        supplements=audit.runner.read_keyed(self.folder/'supplemental_metrics.jsonl')
        for kind in records:
            with self.subTest(kind=kind):
                changed={name:dict(rows) for name,rows in records.items()}
                changed[kind].pop(key[:3] if kind=='context' else key)
                with self.assertRaises(AssertionError): audit.validate_scored_sources(row,key,changed,supplements)
                changed[kind][key[:3] if kind=='context' else key]={'status':'failed'}
                with self.assertRaises(AssertionError): audit.validate_scored_sources(row,key,changed,supplements)
        with self.assertRaises(AssertionError): audit.validate_scored_sources(row,key,records,{})

    def test_single_model_sources_remain_unchanged(self):
        manifest=json.loads((self.folder/'manifest.json').read_text(encoding='utf8'))
        source=audit.resolve_sources(self.folder,manifest)
        self.assertEqual(source['generation'],self.folder)
        self.assertEqual(source['models']['correctness'],'deepseek-v4-pro')

    def test_mixed_metrics_require_distinct_verified_sources(self):
        with tempfile.TemporaryDirectory(dir=audit.ROOT/'.superpowers') as location:
            base=Path(location)
            parent=base/'parent'
            flash=base/'flash'
            shutil.copytree(SMOKE,parent)
            flash.mkdir()
            shutil.copy2(parent/'correctness_reviews.jsonl',flash/'correctness_reviews.jsonl')
            (flash/'manifest.json').write_text('{}',encoding='utf8')
            manifest=json.loads((parent/'manifest.json').read_text(encoding='utf8'))
            manifest['source_directories']={'generation':str(parent.relative_to(audit.ROOT)),
                'correctness':str(flash.relative_to(audit.ROOT))}
            manifest['judge']['models_by_metric']={'context':'deepseek-v4-pro',
                'faithfulness':'deepseek-v4-pro','correctness':'deepseek-flash'}
            filenames=[parent/name for name in ('manifest.json','answers.jsonl','context_reviews.jsonl',
                'faithfulness_reviews.jsonl','supplemental_metrics.jsonl','semantic_manifest.json','correctness_reviews.jsonl')]
            filenames += [flash/'manifest.json',flash/'correctness_reviews.jsonl']
            manifest['source_file_sha256']={str(path.relative_to(audit.ROOT)):audit.runner.file_hash(path) for path in filenames}
            sources=audit.resolve_sources(base,manifest)
            self.assertEqual(sources['review_paths']['context'],parent/'context_reviews.jsonl')
            self.assertEqual(sources['review_paths']['correctness'],flash/'correctness_reviews.jsonl')
            self.assertEqual(sources['models']['correctness'],'deepseek-flash')
            with (flash/'correctness_reviews.jsonl').open('a',encoding='utf8') as stream:stream.write('\n')
            with self.assertRaises(AssertionError):audit.resolve_sources(base,manifest)


if __name__=='__main__': unittest.main()
