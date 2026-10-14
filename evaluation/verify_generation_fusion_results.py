"""Model-free audit of frozen inputs, judge isolation and independently recomputed scores."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import run_generation_fusion_experiment as runner


def close(left, right):
    if left is None or right is None:
        assert left is right, (left, right)
    else:
        assert math.isclose(left, right, abs_tol=1e-10), (left, right)


def tokens(stored, attempts):
    assert stored['attempt_count']==len(attempts)
    for name in ('prompt_tokens','completion_tokens','reasoning_tokens'):
        known=[a[name] for a in attempts if isinstance(a.get(name),int)]
        assert stored[name]==sum(known)
        assert stored[name+'_unknown_attempts']==len(attempts)-len(known)


def timing(stored, values):
    ordered=sorted(values)
    assert stored['count']==len(values)
    close(stored['total'],sum(values))
    close(stored['mean'],statistics.mean(values) if values else None)
    for name, percentile in (('p50',.5),('p95',.95)):
        if values:
            index=(len(values)-1)*percentile
            lower=int(index)
            expected=ordered[lower]+(ordered[min(lower+1,len(values)-1)]-ordered[lower])*(index-lower)
        else: expected=None
        close(stored[name],expected)


def counts(stored, **expected):
    for name,value in expected.items():
        assert stored[name]==value, (name,stored[name],value)


def rubric_pass(row):
    return row.get('grade')=='correct' and all(row.get(name) is True for name in
        ('required_pass','clarification_pass','forbidden_pass'))


def validate_scored_sources(row, key, reviews, supplements):
    fields = {
        'faithfulness': ('faithfulness','citation_support_rate','claim_count'),
        'context': ('context_recall','context_precision'),
        'correctness': ('grade','factual_f1','TP_count','FP_count','FN_count','required_pass',
            'clarification_pass','forbidden_pass','required_checks','clarification_checks','forbidden_checks')}
    for kind, names in fields.items():
        review=reviews[kind].get(key[:3] if kind=='context' else key)
        if review and review['status']=='complete':
            for name in names:
                assert row[name]==review['scores'][name], f'Scored row differs from review: {name}'
        else:
            for name in names:
                assert row.get(name) is None, f'Score without a successful review: {name}'
    supplement=supplements.get(key)
    if supplement:
        close(row['semantic_similarity'],supplement['semantic_similarity'])
        close(row['bertscore_f1'],supplement['bertscore']['f1'])
    else:
        assert row.get('semantic_similarity') is None and row.get('bertscore_f1') is None


def resolve_sources(folder, manifest):
    model_map=manifest['judge'].get('models_by_metric')
    if not model_map:
        return dict(generation=folder,models={name:manifest['judge']['model'] for name in
            ('context','faithfulness','correctness')},review_paths={name:folder/f'{name}_reviews.jsonl'
            for name in ('context','faithfulness','correctness')})
    assert model_map=={'context':'deepseek-v4-pro','faithfulness':'deepseek-v4-pro',
                      'correctness':'deepseek-flash'}, 'Unexpected per-metric model protocol'
    directories=manifest['source_directories']
    assert set(directories)=={'generation','correctness'}
    generation=(ROOT/directories['generation']).resolve()
    correctness=(ROOT/directories['correctness']).resolve()
    parent=json.loads((generation/'manifest.json').read_text(encoding='utf8'))
    for name in ('code_sha256','reference_sha256','source_sha256','selected_cases','repeats',
                 'case_counts','generator'):
        assert manifest[name]==parent[name], f'Inherited generation identity changed: {name}'
    required={generation/name for name in ('answers.jsonl','manifest.json','context_reviews.jsonl',
        'faithfulness_reviews.jsonl','supplemental_metrics.jsonl','semantic_manifest.json','correctness_reviews.jsonl')}
    required|={correctness/'manifest.json',correctness/'correctness_reviews.jsonl'}
    hashes=manifest['source_file_sha256']
    assert required<={(ROOT/name).resolve() for name in hashes}, 'Incomplete metric provenance'
    for name,expected in hashes.items():
        assert runner.file_hash(ROOT/name)==expected, f'Metric source changed: {name}'
    return dict(generation=generation,models=model_map,review_paths={
        'context':generation/'context_reviews.jsonl','faithfulness':generation/'faithfulness_reviews.jsonl',
        'correctness':correctness/'correctness_reviews.jsonl'})


def verify(folder):
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    sources=resolve_sources(folder,manifest)
    generation=sources['generation']
    args = argparse.Namespace(source_dir=ROOT/'evaluation/guard_fusion_experiment_2026-10-13',
        references_dir=ROOT/'evaluation/generation_fusion_references_v1',
        judge_model=sources['models']['context'])
    runner.assert_frozen_inputs(manifest, args)
    for name, expected in manifest['code_sha256'].items():
        assert runner.file_hash(ROOT/name) == expected, f'Code changed: {name}'
    all_cases, chunks = runner.load_frozen(args.source_dir)
    refs, hashes = runner.load_references(args.references_dir, chunks)
    selected = {tuple(k) for k in manifest['selected_cases']}
    cases = [r for r in all_cases if (r['dataset'], r['qid']) in selected]
    answers = runner.read_keyed(generation/'answers.jsonl')
    runner.validate_existing_answers(answers, cases, chunks, manifest['repeats'], manifest['generator']['model'])
    scheduled = {(r['dataset'], r['qid'], r['variant'], repeat)
                 for r in cases for repeat in range(manifest['repeats'])}
    assert set(answers) <= scheduled
    prepared = {runner.metrics.row_key(r, False): runner.prepare_case(r, chunks) for r in cases}
    order_counts = Counter()
    for key, row in answers.items():
        order_counts[''.join(row['planned_pair_order'])] += 1
        if row['status'] == 'complete':
            assert row['api_attempts'], 'Missing real API attempts'
            assert row['usage_attempts'][-1]['finish_reason'] == 'stop'
        for attempt in row['api_attempts']:
            request = attempt['request']
            assert request['messages'] == row['messages'], 'Actual generator prompt changed'
            assert request['model'] == manifest['generator']['model']
            assert request['temperature'] == .2 and request['reasoning_effort']=='low'
            assert request['extra_body']=={'thinking':{'type':'enabled'}}
            assert runner.value_hash(request)==attempt['request_sha256']
        peer = answers.get((key[0], key[1], 'D' if key[2]=='A' else 'A', key[3]))
        if peer and len(row['execution_order'])==2 and key[2] == row['execution_order'][0]:
            assert row['completed_at_utc'] <= peer['started_at_utc'], 'Pair did not run sequentially'
    reviews = {}
    complete = [r for r in answers.values() if r['status']=='complete']
    for kind in ('context', 'faithfulness', 'correctness'):
        records = runner.read_keyed(sources['review_paths'][kind], include_repeat=kind!='context')
        kind_args=argparse.Namespace(judge_model=sources['models'][kind])
        runner.validate_existing_reviews(records, kind,
            list(prepared.values()) if kind=='context' else complete, refs, kind_args)
        for key, row in records.items():
            payload = json.loads(row['messages'][1]['content'])
            allowed = {'question', 'answer', 'contexts'} if kind=='faithfulness' else (
                {'question', 'reference_answer', 'reference_claims', 'contexts'} if kind=='context' else
                {'question', 'answer', 'reference_answer', 'required_points', 'expected_behavior',
                 'forbidden_inferences', 'required_clarifications', 'contexts'})
            assert set(payload)==allowed, f'Unexpected judge fields: {kind}'
            if row['status'] != 'complete':
                assert row['scores'] is None
                continue
            raw, score = row['review'], row['scores']
            if kind=='faithfulness':
                n = len(raw['claims'])
                close(score['faithfulness'], sum(c['support']=='supported' for c in raw['claims'])/n if n else None)
                close(score['citation_support_rate'], sum(c['support']=='supported' and bool(c['cited_labels']) and
                      c['citation_supported'] for c in raw['claims'])/n if n else None)
            elif kind=='correctness':
                tp, fp, fn = (len(raw[k]) for k in ('TP','FP','FN'))
                close(score['factual_f1'], 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
                assert raw['grade']!='correct' or (fp==fn==0 and all(raw['required_checks']) and
                    all(raw['clarification_checks']) and not any(raw['forbidden_checks']))
            elif refs[key[:2]]['positive']:
                supported, relevant = raw['claim_supported'], raw['context_relevant']
                close(score['context_recall'], sum(supported)/len(supported) if supported else None)
                positions = [i+1 for i,v in enumerate(relevant) if v]
                close(score['context_precision'], sum(j/i for j,i in enumerate(positions,1))/len(positions) if positions else 0.)
        reviews[kind] = records
    supplements, semantic = runner.load_supplemental(generation/'supplemental_metrics.jsonl', answers, refs, manifest,
                                                      generation/'answers.jsonl')
    scored = json.loads((folder/'scored_answers.json').read_text(encoding='utf8'))
    summary = json.loads((folder/'summary.json').read_text(encoding='utf8'))
    assert len(scored)==len({runner.metrics.row_key(r) for r in scored}), 'Duplicate scored keys'
    assert {runner.metrics.row_key(r) for r in scored} == set(answers)
    for row in scored:
        key = runner.metrics.row_key(row)
        for field,value in answers[key].items():
            assert row[field]==value, f'Scored original field changed: {field}'
        assert row['positive'] == refs[key[:2]]['positive']
        validate_scored_sources(row,key,reviews,supplements)
        expected = .75*row['factual_f1']+.25*row['semantic_similarity'] if row['positive'] and (
            row.get('factual_f1') is not None and row.get('semantic_similarity') is not None) else None
        close(row['answer_correctness'], expected)
    by_key = {runner.metrics.row_key(r):r for r in scored}
    assert set(summary['datasets'])=={'original46','development36'}
    assert set(summary['judge'])=={'context','faithfulness','correctness'}
    assert summary['manifest_sha256']==runner.file_hash(folder/'manifest.json')
    assert summary['supplemental_records']==len(supplements)
    assert summary['semantic_manifest']==semantic
    for kind, records in reviews.items():
        stored=summary['judge'][kind]
        if manifest['judge'].get('models_by_metric'):
            assert stored['model']==sources['models'][kind]
        planned=len(cases)*(1 if kind=='context' else manifest['repeats'])
        eligible=len(cases) if kind=='context' else len(complete)
        completed=sum(r['status']=='complete' for r in records.values())
        counts(stored,planned=planned,eligible=eligible,recorded=len(records),complete=completed,
            failed=len(records)-completed,unreviewed_eligible=eligible-len(records),
            ineligible_generation_runs=planned-eligible)
        close(stored['completion_rate_all_planned'],completed/planned if planned else None)
        close(stored['completion_rate_eligible'],completed/eligible if eligible else None)
        tokens(stored['tokens'],[a for r in records.values() for a in r['attempts']])
        timing(stored['timing'],[r['judge_seconds'] for r in records.values()])
    for dataset, result in summary['datasets'].items():
        assert set(result['variants'])=={'A','D'}
        assert set(result['paired']['positive_metrics'])=={'faithfulness','citation_support_rate','factual_f1',
            'semantic_similarity','answer_correctness','bertscore_f1'}
        expected_cases=sum(k[0]==dataset for k in selected)
        expected_positive=sum(k[0]==dataset and refs[k]['positive'] for k in selected)
        counts(result,expected_cases=expected_cases,repeats=manifest['repeats'])
        for variant, stored in result['variants'].items():
            group=[r for r in scored if r['dataset']==dataset and r['variant']==variant]
            completed=[r for r in group if r['status']=='complete']
            reviewed=[r for r in completed if r.get('grade') in ('correct','partial','incorrect')]
            passed=sum(rubric_pass(r) for r in reviewed)
            planned=expected_cases*manifest['repeats']
            counts(stored,planned_runs=planned,recorded=len(group),complete=len(completed),
                failed=len(group)-len(completed),missing=planned-len(group),grade_denominator=len(reviewed))
            close(stored['completion_rate'],len(completed)/planned if planned else None)
            assert stored['grades']=={g:sum(r.get('grade')==g for r in reviewed) for g in ('correct','partial','incorrect')}
            counts(stored['rubric'],reviewed=len(reviewed),unreviewed=planned-len(reviewed),passed=passed)
            close(stored['rubric']['reviewed_pass_rate'],passed/len(reviewed) if reviewed else None)
            close(stored['rubric']['all_case_pass_rate'],passed/planned if planned else None)
            assert set(stored['by_kind'])=={r['kind'] for r in group}
            for kind, item in stored['by_kind'].items():
                subset=[r for r in group if r['kind']==kind]
                subset_reviewed=[r for r in subset if r['status']=='complete' and r.get('grade')]
                counts(item,recorded_runs=len(subset),completed=sum(r['status']=='complete' for r in subset),reviewed=len(subset_reviewed))
                assert item['grades']=={g:sum(r['grade']==g for r in subset_reviewed) for g in ('correct','partial','incorrect')}
            tokens(stored['tokens'],[a for r in group for a in r['usage_attempts']])
            timing(stored['generation_timing_all'],[r['generation_seconds'] for r in group if r.get('generation_seconds') is not None])
            timing(stored['generation_timing_complete'],[r['generation_seconds'] for r in completed if r.get('generation_seconds') is not None])
        evidence = result['context_evaluation']
        assert set(evidence['variants'])=={'A','D'}
        assert set(evidence['paired']['metrics'])=={'context_recall','context_precision'}
        counts(evidence,expected_cases=expected_cases,expected_positive_cases=expected_positive)
        for variant, stored in evidence['variants'].items():
            group=[r for key,r in reviews['context'].items() if key[0]==dataset and key[2]==variant]
            completed=[r for r in group if r['status']=='complete']
            positive=[r for r in completed if refs[(r['dataset'],r['qid'])]['positive']]
            counts(stored,planned_questions=expected_cases,recorded=len(group),complete=len(completed),
                failed=len(group)-len(completed),missing=expected_cases-len(group),
                planned_positive_questions=expected_positive,positive_complete=len(positive),
                negative_or_ambiguous_questions=expected_cases-expected_positive)
            close(stored['completion_rate'],len(completed)/expected_cases if expected_cases else None)
            assert stored['positive_na_or_unreviewed']=={name:expected_positive-sum(r['scores'].get(name) is not None for r in positive)
                for name in ('context_recall','context_precision')}
        context_pairs=[]
        for key,a in reviews['context'].items():
            d=reviews['context'].get((key[0],key[1],'D'))
            if key[0]==dataset and key[2]=='A' and refs[key[:2]]['positive'] and d and a['status']==d['status']=='complete':
                context_pairs.append((a,d))
        counts(evidence['paired'],eligible_questions=len(context_pairs),unavailable_positive_questions=expected_positive-len(context_pairs))
        for name, stored in evidence['paired']['metrics'].items():
            pairs = []
            for key, a in reviews['context'].items():
                d = reviews['context'].get((key[0],key[1],'D'))
                if key[0]==dataset and key[2]=='A' and refs[key[:2]]['positive'] and d and (
                    a['status']==d['status']=='complete') and a['scores'].get(name) is not None and d['scores'].get(name) is not None:
                    pairs.append((a['scores'][name], d['scores'][name]))
            for index, variant in enumerate(('A','D')):
                close(stored[variant]['mean'], statistics.mean(p[index] for p in pairs) if pairs else None)
                assert stored[variant]['question_count']==len(pairs)
                assert stored[variant]['na_or_unreviewed_positive_questions']==expected_positive-len(pairs)
            close(stored['delta_D_minus_A'], statistics.mean(d-a for a,d in pairs) if pairs else None)
        generation_pairs=[]
        for key,a in by_key.items():
            d=by_key.get((key[0],key[1],'D',key[3]))
            if key[0]==dataset and key[2]=='A' and d and a['status']==d['status']=='complete':
                generation_pairs.append((a,d))
        positive_pairs=[(a,d) for a,d in generation_pairs if a['positive'] and d['positive']]
        counts(result['paired'],eligible_pairs=len(generation_pairs),eligible_questions=len({a['qid'] for a,d in generation_pairs}),positive_pairs=len(positive_pairs))
        for metric, stored in result['paired']['positive_metrics'].items():
            values = defaultdict(lambda: {'A':[], 'D':[]})
            pairs = 0
            for key, a in by_key.items():
                d = by_key.get((key[0], key[1], 'D', key[3]))
                if key[0]!=dataset or key[2]!='A' or not a['positive'] or not d or (
                    a['status']!='complete' or d['status']!='complete'):
                    continue
                if a.get(metric) is None or d.get(metric) is None:
                    continue
                pairs += 1
                for r in (a,d):
                    values[r['qid']][r['variant']].append(r[metric])
            means = {v: statistics.mean([statistics.mean(x[v]) for x in values.values()]) if values else None
                     for v in ('A','D')}
            for v in ('A','D'):
                close(stored[v]['mean'], means[v])
                assert stored[v]['question_count']==len(values) and stored[v]['paired_run_count']==pairs
                assert stored[v]['na_pair_count']==len(positive_pairs)-pairs
            close(stored['delta_D_minus_A'], means['D']-means['A'] if values else None)
    report = dict(audit_passed=True, code_sha256=runner.file_hash(__file__),
        planned_runs=len(scheduled), recorded_runs=len(answers),
        generation_status=dict(Counter(r['status'] for r in answers.values())), missing_runs=len(scheduled-set(answers)),
        planned_order_counts=dict(order_counts),
        judges={k:dict(Counter(r['status'] for r in rows.values())) for k,rows in reviews.items()},
        supplemental_pairs=len(supplements),
        models_by_metric=sources['models'],
        note='Input/schema/formula audit only; model judgments still need source-grounded case review.')
    runner.save_json(folder/'verification.json', report)
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    verify(parser.parse_args().folder)
