"""Offline contracts for auditable generation evaluation (no service calls)."""
import importlib
import json
import hashlib
import copy
from types import SimpleNamespace
from unittest.mock import patch

import unittest
import tempfile
from pathlib import Path


def reference(positive=True):
    return dict(qid='secret-qid', question='病假需要什么材料？', kind='fact' if positive else 'negative',
                positive=positive, reference_answer='诊断证明。',
                reference_claims=[dict(text='需要诊断证明', evidence_ids=['c1'], quotes=['诊断证明'])] if positive else [],
                required_points=['材料'], required_clarifications=[], forbidden_inferences=['不需要证明'],
                expected_behavior='回答并标注证据', annotation_status='source_grounded_codex_reviewed')


def case():
    return dict(dataset='original46', qid='secret-qid', question='病假需要什么材料？', kind='fact',
                variant='A', repeat=0, candidate_ids=['c1', 'c2'], ranked_candidates=['c1', 'c2'],
                selected_ids=['c1', 'c2'], context_ids=['c1', 'c2'], context_chars=0,
                status='complete', gold_ids=['secret-gold'])


def chunks():
    return {'c1': dict(id='c1', text='需要诊断证明。', pages=[2], path='病假'),
            'c2': dict(id='c2', text='支付福利。', pages=[3], path='福利')}


class GenerationFusionEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)

    def module(self, name):
        self.assertIsNotNone(importlib.util.find_spec(name), 'Generation evaluator is not implemented')
        return importlib.import_module(name)

    def test_generator_gets_only_question_and_actual_chunk_context(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')

        seen = {}
        def fake_generate(question, hits, **kwargs):
            seen.update(question=question, hits=hits, kwargs=kwargs)
            kwargs['usage_sink'].append(dict(prompt_tokens=10, completion_tokens=3, finish_reason='stop'))
            return '需要诊断证明。〔来源1〕'
        result = runner.generate_once(case(), chunks(), repeat=2, model='flash', generate_fn=fake_generate)
        assert result['status'] == 'complete'
        assert result['repeat'] == 2
        assert seen['question'] == '病假需要什么材料？'
        assert seen['hits'][0] == dict(id='c1', text='需要诊断证明。', pages=[2], page_start=2, path='病假')
        assert set(seen['kwargs']) == {'prompt_version', 'thinking_mode', 'usage_sink'}
        assert seen['kwargs']['prompt_version'] == 'evidence'
        assert seen['kwargs']['thinking_mode'] == 'low'
        assert result['source_map'] == {'来源1': 'c1', '来源2': 'c2'}
        assert 'secret-gold' not in json.dumps(seen, ensure_ascii=False)
        assert 'secret-qid' not in json.dumps(seen, ensure_ascii=False)
        assert result['context_sha256'] and result['messages_sha256']
        assert 'started_at_utc' in result, 'Generation audit must record its actual UTC start'
        assert result['started_at_utc'].endswith('+00:00')
        assert result['completed_at_utc'] >= result['started_at_utc']


    def test_failure_keeps_attempt_usage_and_is_never_complete(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')

        def fail(question, hits, **kwargs):
            kwargs['usage_sink'].append(dict(prompt_tokens=20, completion_tokens=200, finish_reason='length'))
            raise RuntimeError('network provider may contain credentials')
        row = runner.generate_once(case(), chunks(), repeat=0, model='flash', generate_fn=fail)
        assert row['status'] == 'failed'
        assert row['answer'] is None
        assert row['usage_attempts'][0]['completion_tokens'] == 200
        assert row['error'] == {'type': 'RuntimeError', 'status_code': None}

    def test_content_filter_completion_does_not_count_as_complete(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        def filtered(question, hits, **kwargs):
            kwargs['usage_sink'].append(dict(finish_reason='content_filter', prompt_tokens=1, completion_tokens=1))
            return '部分文本'
        row = runner.generate_once(case(), chunks(), repeat=0, model='flash', generate_fn=filtered)
        assert row['status'] == 'failed'



    def test_context_budget_uses_actual_admitted_chunks(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')

        cs = chunks()
        cs['c1']['text'] = 'a' * 6100
        row = runner.prepare_case(case(), cs)
        assert row['source_map'] == {'来源1': 'c2'}
        assert [x['id'] for x in row['admitted_contexts']] == ['c2']
        assert row['admitted_contexts'][0]['label'] == '来源1'

    def test_reconstruction_preserves_index_empty_path_display(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        cs = chunks(); cs['c1']['path'] = ''
        row = runner.prepare_case(case(), cs)
        assert '第2页·(无章节)' in row['context']


    def test_corrupt_frozen_ranking_rejected(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')

        for field,value in [('candidate_ids',['c1','c1']),('selected_ids',['absent']),('ranked_candidates',['c2']),('status','failed')]:
            c = case(); c[field] = value
            with self.assertRaises(ValueError):
                runner.validate_frozen_case(c, chunks())


    def test_judge_payloads_enforce_information_boundaries(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')

        row = runner.prepare_case(case(), chunks()); row['answer'] = '需要诊断证明。〔来源1〕'
        ref = reference()
        faith = runner.judge_payload('faithfulness', row, ref)
        assert set(faith) == {'question', 'answer', 'contexts'}
        assert 'reference' not in json.dumps(faith)
        context = runner.judge_payload('context', row, ref)
        assert 'answer' not in context
        assert context['reference_claims'] == ['需要诊断证明']
        correctness = runner.judge_payload('correctness', row, ref)
        assert correctness['required_points'] == ['材料']
        assert 'contexts' in correctness, 'Correctness judge needs admitted evidence to allow supported extra details'
        assert correctness['contexts'][0]['text'] == '需要诊断证明。'
        assert correctness['contexts'][0]['label'] == '来源1'
        for payload in (faith, context, correctness):
            text = json.dumps(payload, ensure_ascii=False)
            assert 'secret-qid' not in text and 'secret-gold' not in text
            assert 'variant' not in payload and 'evidence_ids' not in text


    def test_no_policy_claims_are_na_not_perfect(self):
        metrics = self.module('evaluation.generation_metrics')

        assert metrics.faithfulness_scores({'claims': []}, ['来源1']) == {
            'faithfulness': None, 'citation_support_rate': None, 'claim_count': 0}


    def test_unsupported_citations_reduce_support_not_just_syntax(self):
        metrics = self.module('evaluation.generation_metrics')

        review = {'claims': [dict(claim='事实一', support='supported', evidence_labels=['来源1'],
                                 cited_labels=['来源2'], citation_supported=False, reason='引用错证据'),
                             dict(claim='事实二', support='unsupported', evidence_labels=[],
                                  cited_labels=[], citation_supported=False, reason='无证据')]}
        assert metrics.faithfulness_scores(review, ['来源1', '来源2'])['faithfulness'] == .5
        assert metrics.faithfulness_scores(review, ['来源1', '来源2'])['citation_support_rate'] == 0
        review['claims'][0]['cited_labels'] = ['来源999']
        with self.assertRaises(ValueError):
            metrics.faithfulness_scores(review, ['来源1', '来源2'])


    def test_judge_schema_rejects_wrong_checks_and_fake_bools(self):
        metrics = self.module('evaluation.generation_metrics')

        ref = reference()
        valid = dict(grade='partial', TP=['诊断证明'], FP=[], FN=['其他材料'], required_checks=[True],
                     clarification_checks=[], forbidden_checks=[False], reason='缺少材料')
        self.assertAlmostEqual(metrics.correctness_scores(valid, ref)['factual_f1'], 2/3)
        for changed in (dict(required_checks=[]), dict(forbidden_checks=['false']), dict(TP=[1])):
            with self.assertRaises(ValueError):
                metrics.correctness_scores({**valid, **changed}, ref)

    def test_correct_grade_rejects_failed_required_or_forbidden_checks(self):
        metrics = self.module('evaluation.generation_metrics')
        valid = dict(grade='correct', TP=['診断证明'], FP=[], FN=[], required_checks=[True],
                     clarification_checks=[], forbidden_checks=[False], reason='正确')
        for checks in (dict(required_checks=[False]), dict(forbidden_checks=[True])):
            with self.assertRaises(ValueError):
                metrics.correctness_scores({**valid, **checks}, reference())

    def test_correct_grade_cannot_report_erroneous_or_missing_core_facts(self):
        metrics = self.module('evaluation.generation_metrics')
        review = dict(grade='correct',TP=['核心事实'],FP=[],FN=[],required_checks=[True],
            clarification_checks=[],forbidden_checks=[False],reason='错误地报告正确')
        for contradiction in (dict(FP=['错误制度断言']),dict(FN=['必要核心条件'])):
            with self.assertRaises(ValueError):
                metrics.correctness_scores({**review,**contradiction},reference())

    def test_stored_answer_inputs_must_match_rebuilt_content_not_only_old_hashes(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        row = runner.generate_once(case(),chunks(),repeat=0,model='flash',generate_fn=lambda *a,**kw:'需要材料。〔来源1〕')
        key = ('original46','secret-qid','A',0)
        runner.validate_existing_answers({key:row},[case()],chunks(),1,'flash')
        replacements = dict(dataset='different-dataset',qid='another-qid',kind='negative',variant='D',
            question='改成另一个问题',context='无关上下文',selected_ids=['c2','c1'],
            messages=[dict(role='user',content='保存了另一条提示')],source_map={'来源1':'c2'},
            admitted_contexts=[dict(id='c1',label='来源1',text='无关材料',page_start=2,pages=[2],path='病假')])
        for field,value in replacements.items():
            tampered = copy.deepcopy(row); tampered[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                runner.validate_existing_answers({key:tampered},[case()],chunks(),1,'flash')


    def test_context_average_precision_and_positive_scope(self):
        metrics = self.module('evaluation.generation_metrics')

        review = dict(claim_supported=[True, False], context_relevant=[False, True, True], reasons=['解释'])
        positive = dict(positive=True, reference_claims=[{}, {}])
        score = metrics.context_scores(review, positive, 3)
        assert score['context_recall'] == .5
        self.assertAlmostEqual(score['context_precision'], (1/2 + 2/3)/2)
        assert metrics.context_scores(review, dict(positive=False, reference_claims=[{}, {}]), 3)['context_recall'] is None

    def test_context_summary_is_once_per_question_and_independent_of_answers(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        cases = [{**case(),'qid':qid,'variant':v} for qid in ('q1','q2') for v in ('A','D')]
        refs = {('original46',qid):reference() for qid in ('q1','q2')}
        manifest = dict(generator={'model':'flash'},case_counts={'original46':2,'development36':0},
                        judge={'bias_note':'same family'},quality_note='regression')
        runner.save_json(self.tmp_path/'manifest.json',manifest)
        args = SimpleNamespace(output=self.tmp_path,repeats=3,judge_model='pro',supplemental=None)
        for c in cases:
            prepared = runner.prepare_case(c,chunks())
            present = c['qid']=='q1'
            raw = json.dumps(dict(claim_supported=[present],context_relevant=[present,False],reasons=['证据判定']))
            def call(raw=raw, **kwargs):
                return SimpleNamespace(model='pro', choices=[SimpleNamespace(finish_reason='stop',
                    message=SimpleNamespace(content=raw))],usage=None)
            reviewed = runner.judge_once('context',prepared,refs[('original46',c['qid'])],model='pro',call_fn=call)
            runner.Journal(self.tmp_path/'context_reviews.jsonl').append(reviewed)
        # No answers have been generated, yet all four independent contexts are reviewed.
        empty = runner.run_summary(args,manifest,refs,cases,chunks())['datasets']['original46']
        assert 'context_evaluation' in empty, 'Context quality must exist before any answer generation'
        context = empty['context_evaluation']
        assert context['paired']['eligible_questions']==2
        assert context['paired']['metrics']['context_recall']['A']['mean']==.5
        assert context['paired']['metrics']['context_precision']['D']['mean']==.5
        assert context['paired']['metrics']['context_recall']['A']['question_count']==2
        assert 'context_recall' not in empty['paired']['positive_metrics']
        def failed(*args,**kwargs): raise RuntimeError('generation failure')
        for c in cases:
            for repeat in range(3):
                generate = (lambda *a,**kw:'诊断证明〔来源1〕') if c['qid']=='q1' else failed
                row = runner.generate_once(c,chunks(),repeat=repeat,model='flash',generate_fn=generate)
                runner.Journal(self.tmp_path/'answers.jsonl').append(row)
        partial = runner.run_summary(args,manifest,refs,cases,chunks())['datasets']['original46']
        assert partial['context_evaluation']==context
        assert partial['paired']['eligible_questions']==1


    def test_paired_summary_excludes_failed_counterparts_and_averages_questions(self):
        metrics = self.module('evaluation.generation_metrics')

        runs = []
        for qid, repeat, a, d in [('q1',0,1.,0.), ('q1',1,0.,1.), ('q2',0,1.,1.)]:
            for variant, score in [('A',a), ('D',d)]:
                runs.append(dict(dataset='original46', qid=qid, variant=variant, repeat=repeat,
                                 status='complete', positive=True, faithfulness=score, generation_seconds=1,
                                 grade='correct', required_pass=True, clarification_pass=True, forbidden_pass=True,
                                 usage_attempts=[dict(prompt_tokens=10, completion_tokens=3)]))
        runs.append(dict(dataset='original46', qid='q3', variant='A', repeat=0, status='failed', positive=True,
                         faithfulness=1., generation_seconds=2, usage_attempts=[dict(prompt_tokens=20,completion_tokens=9)]))
        runs.append(dict(dataset='original46', qid='q3', variant='D', repeat=0, status='complete', positive=True,
                         faithfulness=1., generation_seconds=2, usage_attempts=[]))
        s = metrics.summarize_dataset(runs, expected_cases=3, repeats=2)
        assert s['paired']['eligible_pairs'] == 3
        assert s['paired']['eligible_questions'] == 2
        assert s['paired']['positive_metrics']['faithfulness']['A']['mean'] == .75
        assert s['paired']['positive_metrics']['faithfulness']['D']['mean'] == .75
        assert s['variants']['A']['planned_runs'] == 6
        assert s['variants']['A']['complete'] == 3
        assert s['variants']['A']['tokens']['prompt_tokens'] == 50
        assert s['variants']['A']['rubric']['all_case_pass_rate'] == .5


    def test_resume_rejects_incompatible_manifest_and_duplicate_rows(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        tmp_path = self.tmp_path

        runner.initialize_output(tmp_path, {'source': 'one'}, resume=False)
        with self.assertRaisesRegex(ValueError, 'manifest'):
            runner.initialize_output(tmp_path, {'source': 'two'}, resume=True)
        path = tmp_path / 'runs.jsonl'
        row = dict(dataset='x', qid='q', variant='A', repeat=0)
        path.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            runner.read_keyed(path)

    def test_text_hashes_ignore_only_crlf_and_binary_hashes_preserve_bytes(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        lf = self.tmp_path/'lf.txt'; crlf = self.tmp_path/'crlf.txt'
        lf.write_bytes(b'line1\nline2\n'); crlf.write_bytes(b'line1\r\nline2\r\n')
        assert runner.file_hash(lf)==runner.file_hash(crlf), 'Git newline conversion must not change text identity'
        assert runner.file_hash(lf, normalize_newlines=False)!=runner.file_hash(crlf, normalize_newlines=False)
        crlf.write_bytes(b'line1 \r\nline2\r\n')
        assert runner.file_hash(lf)!=runner.file_hash(crlf)

    def test_frozen_loader_accepts_git_newlines_with_normalized_artifact_hash(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        source = runner.ROOT/'evaluation/guard_fusion_experiment_2026-10-13'
        for filename in ('chunks.jsonl','runs.jsonl','content_hashes.json'):
            content = (source/filename).read_bytes().replace(b'\r\n',b'\n')
            (self.tmp_path/filename).write_bytes(content.replace(b'\n',b'\r\n'))
        # Legacy manifest raw hashes intentionally differ; content_hashes owns text identity.
        runner.save_json(self.tmp_path/'manifest.json', {'chunks_sha256':'obsolete_raw_hash'})
        cases, corpus = runner.load_frozen(self.tmp_path)
        assert len(cases)==164 and len(corpus)==123
        hashes = json.loads((self.tmp_path/'content_hashes.json').read_text(encoding='utf8'))
        hashes['chunks_sha256']='genuinely_changed_content'
        runner.save_json(self.tmp_path/'content_hashes.json',hashes)
        with self.assertRaisesRegex(ValueError,'Frozen chunk hash'):
            runner.load_frozen(self.tmp_path)

    def test_resume_preserves_measured_revision_when_only_head_changes(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        frozen = dict(code_revision='measured-head',code_sha256={'generator':'same'},reference_sha256={'original46':'same'})
        runner.initialize_output(self.tmp_path,frozen,resume=False)
        actual = runner.initialize_output(self.tmp_path,{**frozen,'code_revision':'results-commit'},resume=True)
        assert actual==frozen
        assert json.loads((self.tmp_path/'manifest.json').read_text(encoding='utf8'))==frozen
        with self.assertRaisesRegex(ValueError,'manifest'):
            runner.initialize_output(self.tmp_path,{**frozen,'code_revision':'another','code_sha256':{'generator':'changed'}},resume=True)

    def test_fatal_authentication_error_stops_queued_generation_requests(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        args = SimpleNamespace(output=self.tmp_path,references_dir=self.tmp_path,source_dir=self.tmp_path,
                               workers=1,repeats=3)
        manifest = dict(generator={'model':'flash'},reference_sha256={},source_sha256={})
        refs = {('original46','secret-qid'):reference()}
        cases = [case(), {**case(),'variant':'D'}]
        calls = []
        class AuthenticationError(RuntimeError): status_code=401
        def create(**kwargs):
            calls.append(kwargs)
            raise AuthenticationError('Do not retain this possible secret-bearing provider message')
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(runner.llm,'get_client',return_value=client),patch.object(runner.llm,'env_config',return_value={'model':'flash'}):
            runner.run_generate(args,manifest,refs,cases,chunks())
        rows = runner.read_jsonl(self.tmp_path/'answers.jsonl')
        assert len(calls)==1, 'A fatal credential response must stop later queued requests'
        assert len(rows)==1 and rows[0]['status']=='failed'
        assert rows[0]['error']['status_code']==401
        assert rows[0]['usage_attempts'][0]['prompt_tokens'] is None

    def test_fatal_balance_error_stops_queued_judge_requests(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        args = SimpleNamespace(output=self.tmp_path,references_dir=self.tmp_path,source_dir=self.tmp_path,
                               workers=1,repeats=1,judge_model='pro',judge_attempts=2)
        manifest = dict(generator={'model':'flash'},reference_sha256={},source_sha256={})
        refs = {('original46','secret-qid'):reference()}
        cases = [case(), {**case(),'variant':'D'}]
        for c in cases:
            row = runner.generate_once(c,chunks(),repeat=0,model='flash',generate_fn=lambda *a,**kw:'材料〔来源1〕')
            runner.Journal(self.tmp_path/'answers.jsonl').append(row)
        calls = []
        class BalanceError(RuntimeError): status_code=402
        def create(**kwargs):
            calls.append(kwargs)
            raise BalanceError('possible provider secret')
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(runner.llm,'get_client',return_value=client):
            runner.run_judge(args,manifest,refs,cases,chunks())
        recorded = sum(len(runner.read_jsonl(self.tmp_path/f'{kind}_reviews.jsonl')) for kind in ('context','faithfulness','correctness'))
        assert len(calls)==1, 'A fatal balance response must stop later queued requests'
        assert recorded==1


    def test_judge_retry_records_invalid_and_truncated_attempts(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')

        raw = ['{"claims": []}', '{"claims": []}', '{"claims": []}']
        finishes = ['length', 'stop', 'stop']
        count = [0]
        def call(**kwargs):
            i=count[0]; count[0]+=1
            return SimpleNamespace(model='pro-revision', choices=[SimpleNamespace(finish_reason=finishes[i],
                message=SimpleNamespace(content=raw[i]))], usage=SimpleNamespace(prompt_tokens=100,completion_tokens=10))
        row = runner.prepare_case(case(), chunks()); row.update(answer='无事实', repeat=0)
        result = runner.judge_once('faithfulness', row, reference(), model='pro', call_fn=call, max_attempts=3)
        assert result['status'] == 'complete'
        assert len(result['attempts']) == 2
        assert result['attempts'][0]['status'] == 'failed'
        assert result['attempts'][0]['raw_response'] == '{"claims": []}'
        assert result['scores']['faithfulness'] is None

    def test_resumed_judge_rejects_tampered_score(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        row = runner.prepare_case(case(), chunks()); row.update(answer='无事实', repeat=0)
        def call(**kwargs):
            return SimpleNamespace(model='pro', choices=[SimpleNamespace(finish_reason='stop',
                message=SimpleNamespace(content='{"claims":[]}'))], usage=None)
        review = runner.judge_once('faithfulness', row, reference(), model='pro', call_fn=call)
        review['scores']['faithfulness'] = 1.
        refs = {('original46', 'secret-qid'): reference()}
        with self.assertRaises(ValueError):
            runner.validate_existing_reviews({('original46','secret-qid','A',0):review},
                'faithfulness', [row], refs, SimpleNamespace(judge_model='pro'))

    def test_judge_invalid_json_attempts_are_preserved(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        row = runner.prepare_case(case(), chunks()); row.update(answer='无事实', repeat=0)
        counter = [0]
        def call(**kwargs):
            counter[0] += 1
            raw = '{broken' if counter[0]==1 else '{"claims":[]}'
            return SimpleNamespace(model='pro', choices=[SimpleNamespace(finish_reason='stop',
                message=SimpleNamespace(content=raw))], usage=None)
        result = runner.judge_once('faithfulness', row, reference(), model='pro', call_fn=call)
        assert result['status'] == 'complete'
        assert result['attempts'][0]['raw_response'] == '{broken'
        assert result['attempts'][0]['error']['type'] == 'JSONDecodeError'
        assert result['attempts'][0]['prompt_tokens'] is None


    def test_pair_schedule_alternates_case_and_repeat(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')

        cases = [{**case(), 'qid': q, 'variant': v} for q in ['q1','q2'] for v in ['A','D']]
        tasks = runner.schedule_generations(cases, repeats=2)
        assert [(x['qid'], r, x['variant']) for x,r,order in tasks] == [
            ('q1',0,'A'),('q1',0,'D'),('q2',0,'D'),('q2',0,'A'),
            ('q1',1,'D'),('q1',1,'A'),('q2',1,'A'),('q2',1,'D')]

    def test_generation_workers_execute_each_pair_sequentially(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        refs = {('original46','secret-qid'):reference()}
        cases = [case(), {**case(),'variant':'D'}]
        args = SimpleNamespace(output=self.tmp_path,references_dir=self.tmp_path,source_dir=self.tmp_path,
                               workers=3,repeats=3)
        manifest = dict(generator={'model':'flash'},reference_sha256={},source_sha256={})
        events = []
        from threading import Lock
        from time import sleep
        lock = Lock()
        def generate(c, chunk_map, *, repeat, model):
            with lock: events.append((repeat,c['variant'],'start'))
            sleep(.01)
            with lock: events.append((repeat,c['variant'],'end'))
            return dict(dataset=c['dataset'],qid=c['qid'],variant=c['variant'],repeat=repeat,status='complete')
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:None)))
        with patch.object(runner,'generate_once',side_effect=generate), patch.object(runner.llm,'get_client',return_value=client):
            runner.run_generate(args,manifest,refs,cases,chunks())
        for rep in range(3):
            pair = [event[1:] for event in events if event[0]==rep]
            order = ['A','D'] if rep%2==0 else ['D','A']
            assert pair==[(order[0],'start'),(order[0],'end'),(order[1],'start'),(order[1],'end')]


    def test_missing_semantic_similarity_never_becomes_zero(self):
        metrics = self.module('evaluation.generation_metrics')

        assert metrics.answer_correctness(.8, None, positive=True) is None
        self.assertAlmostEqual(metrics.answer_correctness(.8, .4, positive=True), .7)
        assert metrics.answer_correctness(.8, .4, positive=False) is None

    def supplement_fixture(self, runner):
        from evaluation.run_generation_semantic_metrics import clean_text
        row = runner.generate_once(case(),chunks(),repeat=0,model='flash',generate_fn=lambda *a,**kw:'需要诊断证明。〔来源1〕')
        path = self.tmp_path/'answers.jsonl'
        runner.Journal(path).append(row)
        records = runner.read_keyed(path)
        ref = reference()
        refs = {('original46','secret-qid'):ref}
        manifest = dict(reference_sha256={'original46':'frozen-ref-file','development36':'frozen-ref-file2'},
            code_sha256={'evaluation/run_generation_semantic_metrics.py':runner.file_hash(runner.ROOT/'evaluation/run_generation_semantic_metrics.py')})
        supplemental = dict(dataset='original46',qid='secret-qid',variant='A',repeat=0,status='complete',
            answer_sha256=hashlib.sha256(clean_text(row['answer']).encode('utf8')).hexdigest(),
            reference_sha256=hashlib.sha256(clean_text(ref['reference_answer']).encode('utf8')).hexdigest(),
            semantic_similarity=.7,bertscore=dict(precision=.8,recall=.7,f1=.75))
        runner.Journal(self.tmp_path/'supplemental_metrics.jsonl').append(supplemental)
        semantic_manifest = dict(answers_sha256=runner.file_hash(path),reference_sha256=manifest['reference_sha256'],
            source_sha256=manifest['code_sha256']['evaluation/run_generation_semantic_metrics.py'],scored_pairs=1)
        runner.save_json(self.tmp_path/'semantic_manifest.json',semantic_manifest)
        return records,refs,manifest,path,supplemental

    def test_supplements_reject_same_key_scores_from_a_different_answer_file(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        self.assertTrue(hasattr(runner,'load_supplemental'), 'Supplemental scores need frozen input validation')
        records,refs,manifest,path,supp = self.supplement_fixture(runner)
        metrics_path = self.tmp_path/'supplemental_metrics.jsonl'
        accepted, metadata = runner.load_supplemental(metrics_path,records,refs,manifest,path)
        assert len(accepted)==1
        # Same legal run key; generation answer and its ordinary answer hash both updated.
        row = next(iter(records.values())); row['answer']='需要其他凭证。〔来源1〕'
        row['answer_sha256']=runner.value_hash(row['answer'])
        path.write_text(json.dumps(row,ensure_ascii=False)+'\n',encoding='utf8')
        with self.assertRaisesRegex(ValueError,'answer'):
            runner.load_supplemental(metrics_path,records,refs,manifest,path)
        # Updating only the whole-journal hash must not bypass the per-answer identity guard.
        metadata['answers_sha256']=runner.file_hash(path)
        runner.save_json(self.tmp_path/'semantic_manifest.json',metadata)
        with self.assertRaisesRegex(ValueError,'answer'):
            runner.load_supplemental(metrics_path,records,refs,manifest,path)

    def test_summary_cannot_attach_old_scores_to_a_legally_changed_answer(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        records,refs,manifest,path,supp = self.supplement_fixture(runner)
        manifest.update(generator={'model':'flash'},case_counts={'original46':1,'development36':0},
                        judge={'bias_note':'same family'},quality_note='regression')
        runner.save_json(self.tmp_path/'manifest.json',manifest)
        row = next(iter(records.values()))
        row['answer']='完全不同的材料要求。〔来源1〕'
        row['answer_sha256']=runner.value_hash(row['answer'])
        path.write_text(json.dumps(row,ensure_ascii=False)+'\n',encoding='utf8')
        args = SimpleNamespace(output=self.tmp_path,repeats=1,judge_model='pro',supplemental=None)
        with self.assertRaisesRegex(ValueError,'answer'):
            runner.run_summary(args,manifest,refs,[case(),{**case(),'variant':'D'}],chunks())

    def test_supplements_require_metadata_and_reject_mismatched_reference(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        self.assertTrue(hasattr(runner,'load_supplemental'), 'Supplemental scores need frozen input validation')
        records,refs,manifest,path,supp = self.supplement_fixture(runner)
        metrics_path = self.tmp_path/'supplemental_metrics.jsonl'
        metadata = json.loads((self.tmp_path/'semantic_manifest.json').read_text(encoding='utf8'))
        metadata['reference_sha256']={'original46':'different'}
        runner.save_json(self.tmp_path/'semantic_manifest.json',metadata)
        with self.assertRaisesRegex(ValueError,'reference'):
            runner.load_supplemental(metrics_path,records,refs,manifest,path)
        (self.tmp_path/'semantic_manifest.json').unlink()
        with self.assertRaisesRegex(ValueError,'semantic_manifest'):
            runner.load_supplemental(metrics_path,records,refs,manifest,path)

    def test_supplements_reject_extra_keys_nonpositive_and_nonfinite_scores(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        self.assertTrue(hasattr(runner,'load_supplemental'), 'Supplemental scores need frozen input validation')
        records,refs,manifest,path,supp = self.supplement_fixture(runner)
        metrics_path = self.tmp_path/'supplemental_metrics.jsonl'
        for change in (dict(qid='unplanned'),dict(semantic_similarity=float('nan')),
                       dict(bertscore={'precision':.8,'recall':float('inf'),'f1':.7}),
                       dict(reference_sha256='a-different-clean-reference')):
            metrics_path.write_text(json.dumps({**supp,**change})+'\n',encoding='utf8')
            with self.assertRaises(ValueError):
                runner.load_supplemental(metrics_path,records,refs,manifest,path)
        metrics_path.write_text(json.dumps(supp)+'\n',encoding='utf8')
        refs[('original46','secret-qid')]['positive']=False
        with self.assertRaises(ValueError):
            runner.load_supplemental(metrics_path,records,refs,manifest,path)
        refs[('original46','secret-qid')]['positive']=True
        next(iter(records.values()))['status']='failed'
        with self.assertRaises(ValueError):
            runner.load_supplemental(metrics_path,records,refs,manifest,path)

    def test_missing_supplements_remain_unscored_without_metadata(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        self.assertTrue(hasattr(runner,'load_supplemental'), 'Supplemental scores need frozen input validation')
        scores, metadata = runner.load_supplemental(self.tmp_path/'absent.jsonl',{}, {}, {},self.tmp_path/'answers.jsonl')
        assert scores=={} and metadata is None

    def test_offline_stages_persist_audits_and_resume_without_replacing_records(self):
        runner = self.module('evaluation.run_generation_fusion_experiment')
        args = SimpleNamespace(output=self.tmp_path, source_dir=self.tmp_path, references_dir=self.tmp_path,
            workers=2, repeats=2, judge_model='pro', judge_attempts=2, supplemental=None)
        cases = [case(), {**case(), 'variant':'D'}]
        refs = {('original46','secret-qid'):reference()}
        manifest = dict(reference_sha256={}, source_sha256={}, generator={'model':'flash'},
            case_counts={'original46':1,'development36':0}, judge={'bias_note':'same family'}, quality_note='regression')
        runner.save_json(self.tmp_path/'manifest.json', manifest)
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            if kwargs['model']=='flash':
                raw = '需要诊断证明。〔来源1〕'
            elif kwargs['messages'][0]['content']==runner.JUDGE_INSTRUCTIONS['faithfulness']:
                raw = json.dumps({'claims':[dict(claim='需要诊断证明',support='supported',evidence_labels=['来源1'],
                    cited_labels=['来源1'],citation_supported=True,reason='资料明确')]},ensure_ascii=False)
            elif kwargs['messages'][0]['content']==runner.JUDGE_INSTRUCTIONS['context']:
                raw = json.dumps(dict(claim_supported=[True],context_relevant=[True,False],reasons=['资料一支持']))
            else:
                raw = json.dumps(dict(grade='correct',TP=['需要诊断证明'],FP=[],FN=[],required_checks=[True],
                    clarification_checks=[],forbidden_checks=[False],reason='正确'))
            return SimpleNamespace(model=kwargs['model'],choices=[SimpleNamespace(finish_reason='stop',
                message=SimpleNamespace(content=raw))],usage=SimpleNamespace(prompt_tokens=10,completion_tokens=5))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(runner.llm,'get_client',return_value=client), patch.object(runner.llm,'env_config',return_value={'model':'flash'}):
            runner.run_generate(args,manifest,refs,cases,chunks())
            runner.run_judge(args,manifest,refs,cases,chunks())
            first_calls = len(calls)
            runner.run_generate(args,manifest,refs,cases,chunks())
            runner.run_judge(args,manifest,refs,cases,chunks())
        assert len(calls)==first_calls==14
        answers = runner.read_jsonl(self.tmp_path/'answers.jsonl')
        assert len(answers)==4
        assert all(len(row['api_attempts'])==1 for row in answers)
        assert all(row['api_attempts'][0]['raw_response']=='需要诊断证明。〔来源1〕' for row in answers)
        assert len(runner.read_jsonl(self.tmp_path/'context_reviews.jsonl'))==2
        summary = runner.run_summary(args,manifest,refs,cases,chunks())
        original = summary['datasets']['original46']
        assert 'planned' in summary['judge']['faithfulness'], 'Judge completion needs an explicit planned denominator'
        assert summary['judge']['faithfulness']['planned']==4
        assert summary['judge']['faithfulness']['eligible']==4
        assert summary['judge']['context']['planned']==2
        assert summary['judge']['faithfulness']['completion_rate_all_planned']==1
        assert original['paired']['eligible_pairs']==2
        assert original['paired']['positive_metrics']['faithfulness']['A']['mean']==1
        assert original['variants']['A']['rubric']['all_case_pass_rate']==1
        assert original['paired']['positive_metrics']['answer_correctness']['A']['mean'] is None

if __name__ == "__main__":
    unittest.main()
