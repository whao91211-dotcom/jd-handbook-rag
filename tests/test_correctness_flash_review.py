"""Offline correctness-only continuation contracts; no service calls."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation import run_generation_fusion_experiment as runner


class CorrectnessFlashReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.parent = self.folder/'parent'; self.parent.mkdir()
        self.output = self.folder/'flash'
        self.refs_dir = self.folder/'refs'; self.refs_dir.mkdir()
        self.source = self.folder/'source'; self.source.mkdir()
        self.chunk_map = {'c1':dict(id='c1',text='需要诊断证明。',pages=[2],path='病假')}
        self.references = {}
        self.cases = []
        for qid,kind,positive in [('q1','fact',True),('q2','negative',False)]:
            self.references[('original46',qid)] = dict(qid=qid,question='需要什么材料？',kind=kind,
                positive=positive,reference_answer='需要诊断证明。' if positive else '不得推断其他规则。',
                reference_claims=[dict(text='需要诊断证明',evidence_ids=['c1'],quotes=['诊断证明'])] if positive else [],
                required_points=['说明材料'] if positive else ['不推断'],required_clarifications=[],
                forbidden_inferences=['不需证明'],expected_behavior='依据资料回答',annotation_status='reviewed')
            for variant in ('A','D'):
                self.cases.append(dict(dataset='original46',qid=qid,question='需要什么材料？',kind=kind,
                    variant=variant,repeat=0,candidate_ids=['c1'],ranked_candidates=['c1'],selected_ids=['c1'],status='complete'))
        for ds in runner.DATASETS:
            runner.save_json(self.refs_dir/f'{ds}.json',[r for k,r in self.references.items() if k[0]==ds])
        self.parent_manifest = dict(schema_version=runner.SCHEMA_VERSION,code_revision='measured-pro-head',
            code_sha256={'ragcore/llm.py':runner.file_hash(runner.ROOT/'ragcore/llm.py')},source_sha256={},
            reference_sha256={ds:runner.file_hash(self.refs_dir/f'{ds}.json') for ds in runner.DATASETS},
            selected_cases=[['original46','q1'],['original46','q2']],case_counts={'original46':2,'development36':0},
            repeats=2,generator={'model':'deepseek-v4-flash'},judge={'model':'deepseek-v4-pro'},
            service_destination_sha256='configured-service-hash')
        runner.save_json(self.parent/'manifest.json',self.parent_manifest)
        self.answers = []
        for c in self.cases:
            self.answers.append(runner.generate_once(c,self.chunk_map,repeat=0,model='deepseek-v4-flash',
                                generate_fn=lambda *a,**kw:'需要诊断证明。〔来源1〕'))
        def fail(*a,**kw): raise RuntimeError('failed generation')
        self.answers.append(runner.generate_once(self.cases[0],self.chunk_map,repeat=1,
                            model='deepseek-v4-flash',generate_fn=fail))
        for row in self.answers: runner.Journal(self.parent/'answers.jsonl').append(row)
        (self.parent/'correctness_reviews.jsonl').write_text('{"preserve":"old Pro journal"}\n',encoding='utf8')
        self.args = SimpleNamespace(parent=self.parent,output=self.output,references_dir=self.refs_dir,
            source_dir=self.source,workers=2,max_attempts=2,resume=False)

    def module(self):
        name = 'evaluation.run_correctness_flash_review'
        self.assertIsNotNone(importlib.util.find_spec(name),'Independent Flash correctness runner is missing')
        return importlib.import_module(name)

    def bundle(self, module):
        with patch.object(runner,'load_frozen',return_value=(self.cases,self.chunk_map)):
            return module.load_parent(self.args)

    def response(self, raw):
        return SimpleNamespace(model='deepseek-flash',choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=raw))],usage=SimpleNamespace(prompt_tokens=11,completion_tokens=7))

    def test_only_correctness_of_all_successful_answers_is_reviewed_and_pro_files_preserved(self):
        module = self.module(); bundle = self.bundle(module)
        manifest = module.build_manifest(self.args,bundle)
        frozen = runner.initialize_output(self.output,manifest,resume=False)
        parent_before = {p.name:p.read_bytes() for p in self.parent.iterdir()}
        seen = []
        original_judge = runner.judge_once
        def judge(kind,row,ref,**kwargs):
            seen.append((kind,row['qid'],row['variant'],row['repeat'],kwargs['model']))
            raw=json.dumps(dict(grade='correct',TP=['材料要求'],FP=[],FN=[],required_checks=[True],
                clarification_checks=[],forbidden_checks=[False],reason='通过'))
            return original_judge(kind,row,ref,call_fn=lambda **kw:self.response(raw),**kwargs)
        with patch.object(runner,'judge_once',side_effect=judge):
            summary = module.run_reviews(self.args,frozen,bundle)
        assert len(seen)==4
        assert all(x[0]=='correctness' and x[4]=='deepseek-flash' for x in seen)
        assert {x[1] for x in seen}=={'q1','q2'}
        assert {p.name:p.read_bytes() for p in self.parent.iterdir()}==parent_before
        assert len(runner.read_jsonl(self.output/'correctness_reviews.jsonl'))==4
        assert summary['planned']==4 and summary['complete']==4 and summary['failed']==0
        assert not (self.output/'context_reviews.jsonl').exists()
        assert not (self.output/'faithfulness_reviews.jsonl').exists()
        assert (self.output/'parent_manifest.json').read_bytes()==(self.parent/'manifest.json').read_bytes()

    def test_resume_never_replaces_success_or_failure_and_rejects_other_judge_model(self):
        module = self.module(); bundle = self.bundle(module)
        manifest = module.build_manifest(self.args,bundle)
        runner.initialize_output(self.output,manifest,resume=False)
        calls = []
        original_judge = runner.judge_once
        def judge(kind,row,ref,**kwargs):
            calls.append(row['qid'])
            if row['variant']=='A':
                return original_judge(kind,row,ref,call_fn=lambda **kw:self.response('{not valid JSON'),**kwargs)
            raw=json.dumps(dict(grade='correct',TP=['材料'],FP=[],FN=[],required_checks=[True],
                clarification_checks=[],forbidden_checks=[False],reason='通过'))
            return original_judge(kind,row,ref,call_fn=lambda **kw:self.response(raw),**kwargs)
        with patch.object(runner,'judge_once',side_effect=judge):
            module.run_reviews(self.args,manifest,bundle)
            first=(self.output/'correctness_reviews.jsonl').read_bytes()
            module.run_reviews(self.args,manifest,bundle)
        assert len(calls)==4
        assert (self.output/'correctness_reviews.jsonl').read_bytes()==first
        failed = [row for row in runner.read_jsonl(self.output/'correctness_reviews.jsonl') if row['status']=='failed']
        assert len(failed)==2 and all(len(row['attempts'])==2 for row in failed)
        assert all(row['attempts'][0]['raw_response']=='{not valid JSON' for row in failed)
        rows=runner.read_jsonl(self.output/'correctness_reviews.jsonl');rows[0]['model']='deepseek-v4-pro'
        (self.output/'correctness_reviews.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'model'):
            module.run_reviews(self.args,manifest,bundle)

    def test_parent_answer_reference_and_measured_code_identity_are_required(self):
        module = self.module(); bundle = self.bundle(module)
        manifest = module.build_manifest(self.args,bundle)
        (self.parent/'answers.jsonl').write_text((self.parent/'answers.jsonl').read_text(encoding='utf8')+'\n',encoding='utf8')
        with self.assertRaisesRegex(ValueError,'answer'):
            module.assert_identity(self.args,manifest)
        # Original measured source hash validation occurs before any judge scheduling.
        changed = copy.deepcopy(self.parent_manifest)
        changed['code_sha256']['ragcore/llm.py']='different-measured-code'
        runner.save_json(self.parent/'manifest.json',changed)
        with patch.object(runner,'load_frozen',return_value=(self.cases,self.chunk_map)), self.assertRaisesRegex(ValueError,'code'):
            module.load_parent(self.args)

    def test_reference_change_and_frozen_body_tamper_reject_before_scheduling(self):
        module=self.module();bundle=self.bundle(module)
        manifest=module.build_manifest(self.args,bundle)
        path=self.refs_dir/'original46.json';before=path.read_bytes()
        path.write_bytes(before+b' ')
        with self.assertRaisesRegex(ValueError,'reference'):
            module.assert_identity(self.args,manifest)
        path.write_bytes(before)
        row=copy.deepcopy(self.answers[0]);row['admitted_contexts'][0]['text']='另一个上下文'
        answer_path=self.parent/'answers.jsonl'
        answer_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in [row,*self.answers[1:]]),encoding='utf8')
        with patch.object(runner,'load_frozen',return_value=(self.cases,self.chunk_map)),self.assertRaisesRegex(ValueError,'frozen input'):
            module.load_parent(self.args)

    def test_fatal_balance_status_stops_queued_calls_and_keeps_failure_usage(self):
        module = self.module(); bundle = self.bundle(module)
        self.args.workers=1
        manifest=module.build_manifest(self.args,bundle)
        runner.initialize_output(self.output,manifest,resume=False)
        calls=[]; original_judge=runner.judge_once
        class BalanceError(RuntimeError): status_code=402
        def call(**kw):
            calls.append(kw);raise BalanceError('provider may echo a secret')
        with patch.object(runner,'judge_once',side_effect=lambda kind,row,ref,**kw: original_judge(kind,row,ref,call_fn=call,**kw)):
            summary=module.run_reviews(self.args,manifest,bundle)
        assert len(calls)==1
        assert summary['recorded']==1 and summary['failed']==1 and summary['missing']==3
        review=runner.read_jsonl(self.output/'correctness_reviews.jsonl')[0]
        assert review['attempts'][0]['error']['status_code']==402
        assert review['attempts'][0]['prompt_tokens'] is None
        assert 'provider may echo' not in json.dumps(review)


if __name__=='__main__': unittest.main()
