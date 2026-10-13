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


def verify(folder):
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    args = argparse.Namespace(source_dir=ROOT/'evaluation/guard_fusion_experiment_2026-10-13',
        references_dir=ROOT/'evaluation/generation_fusion_references_v1',
        judge_model=manifest['judge']['model'])
    runner.assert_frozen_inputs(manifest, args)
    for name, expected in manifest['code_sha256'].items():
        assert runner.file_hash(ROOT/name) == expected, f'Code changed: {name}'
    all_cases, chunks = runner.load_frozen(args.source_dir)
    refs, hashes = runner.load_references(args.references_dir, chunks)
    selected = {tuple(k) for k in manifest['selected_cases']}
    cases = [r for r in all_cases if (r['dataset'], r['qid']) in selected]
    answers = runner.read_keyed(folder/'answers.jsonl')
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
        if peer and key[2] == row['execution_order'][0]:
            assert row['completed_at_utc'] <= peer['started_at_utc'], 'Pair did not run sequentially'
    reviews = {}
    complete = [r for r in answers.values() if r['status']=='complete']
    for kind in ('context', 'faithfulness', 'correctness'):
        records = runner.read_keyed(folder/f'{kind}_reviews.jsonl', include_repeat=kind!='context')
        runner.validate_existing_reviews(records, kind,
            list(prepared.values()) if kind=='context' else complete, refs, args)
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
    supplements, semantic = runner.load_supplemental(folder/'supplemental_metrics.jsonl', answers, refs, manifest,
                                                      folder/'answers.jsonl')
    scored = json.loads((folder/'scored_answers.json').read_text(encoding='utf8'))
    summary = json.loads((folder/'summary.json').read_text(encoding='utf8'))
    assert {runner.metrics.row_key(r) for r in scored} == set(answers)
    for row in scored:
        key = runner.metrics.row_key(row)
        assert row['positive'] == refs[key[:2]]['positive']
        for kind in ('context', 'faithfulness', 'correctness'):
            review = reviews[kind].get(key[:3] if kind=='context' else key)
            if review and review['status']=='complete':
                for field, value in review['scores'].items():
                    assert row[field]==value, f'Scored row differs from review: {field}'
        supplement = supplements.get(key)
        if supplement:
            close(row['semantic_similarity'], supplement['semantic_similarity'])
            close(row['bertscore_f1'], supplement['bertscore']['f1'])
        expected = .75*row['factual_f1']+.25*row['semantic_similarity'] if row['positive'] and (
            row.get('factual_f1') is not None and row.get('semantic_similarity') is not None) else None
        close(row['answer_correctness'], expected)
    by_key = {runner.metrics.row_key(r):r for r in scored}
    for dataset, result in summary['datasets'].items():
        evidence = result['context_evaluation']
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
            close(stored['delta_D_minus_A'], statistics.mean(d-a for a,d in pairs) if pairs else None)
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
            close(stored['delta_D_minus_A'], means['D']-means['A'] if values else None)
    report = dict(audit_passed=True, code_sha256=runner.file_hash(__file__),
        planned_runs=len(scheduled), recorded_runs=len(answers),
        generation_status=dict(Counter(r['status'] for r in answers.values())), missing_runs=len(scheduled-set(answers)),
        planned_order_counts=dict(order_counts),
        judges={k:dict(Counter(r['status'] for r in rows.values())) for k,rows in reviews.items()},
        supplemental_pairs=len(supplements),
        note='Input/schema/formula audit only; model judgments still need source-grounded case review.')
    runner.save_json(folder/'verification.json', report)
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    verify(parser.parse_args().folder)
