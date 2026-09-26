"""Validate saved ablations and summarize explicit reviewer notes, never auto-grade facts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evaluation/evidence_optimization_summary_2026-09-26'

def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]

def main():
    OUT.mkdir(exist_ok=True)
    notes = json.loads((ROOT/'evaluation/primary_review_notes.json').read_text(encoding='utf8'))
    cases = read(ROOT/'evaluation/baseline_v1_2026-09-26/cases.jsonl')
    case_ids = {x['qid'] for x in cases}
    latest, all_rows = {}, []
    for folder in ('evidence_experiment_2026-09-26','guarded_experiment_2026-09-26','fast_experiment_2026-09-26','low_experiment_2026-09-26'):
        path = ROOT/'evaluation'/folder/'answers.jsonl'
        if not path.exists():
            continue
        records = read(path)
        for row in records:
            assert row['qid'] in case_ids
            assert hashlib.sha256(row['context'].encode()).hexdigest() == row['context_sha256']
            assert hashlib.sha256(row['system_prompt'].encode()).hexdigest() == row['system_sha256']
            latest[(row['variant'],row['qid'])] = row
        all_rows.extend(records)
    reviewed = []
    summary = {'reviewer': 'Codex model review; not a full human gold evaluation',
               'rubric': 'baseline_v1_2026-09-26/RUBRIC.md', 'dataset': '36 simulated development/regression cases'}
    for variant in 'ABCDEFGH':
        rows = [latest[(variant,qid)] for qid in case_ids if (variant,qid) in latest]
        complete = [x for x in rows if not x['error'] and x['usage'] and x['usage'][-1].get('finish_reason')=='stop']
        counts = Counter()
        for row in rows:
            note = notes.get(variant+'/'+row['qid'])
            if note:
                grade, reason = note
                assert grade in ('correct','partial','incorrect')
                counts[grade] += 1
                reviewed.append({'variant':variant,'qid':row['qid'],'answer_grade':grade,'reason':reason,
                    'answer_sha256':hashlib.sha256(row['answer'].encode()).hexdigest(),
                    'reviewer_type':'Codex_model_review','citation_semantic_review':'not_scored_in_this_round'})
        known = lambda key: sum(u.get(key) or 0 for x in rows for u in x['usage'])
        clean_timed = [x for x in rows if x['usage'] and all(not u.get('error_type') for u in x['usage'])]
        summary[variant] = {'saved_cases':len(rows),'complete_answers':len(complete),
            'error_answers':sum(bool(x['error']) for x in rows), 'grades':dict(counts),
            'unreviewed':len(rows)-sum(counts.values()),
            'strict_accuracy':counts['correct']/36 if sum(counts.values())==36 else None,
            'first_attempt_complete':sum(len(x['usage'])==1 and x['usage'][0].get('finish_reason')=='stop' for x in rows),
            'total_application_attempts_latest_runs':sum(len(x['usage']) for x in rows),
            'reported_prompt_tokens_latest_runs':known('prompt_tokens'),
            'reported_completion_tokens_latest_runs':known('completion_tokens'),
            'unknown_usage_attempts':sum(u.get('prompt_tokens') is None or u.get('completion_tokens') is None for x in rows for u in x['usage']),
            'clean_timing_n':len(clean_timed),
            'generation_p50_seconds':statistics.median(x['generation_seconds'] for x in clean_timed) if clean_timed else None,
            'generation_p95_seconds_nearest_rank': sorted(x['generation_seconds'] for x in clean_timed)[max(0,__import__('math').ceil(.95*len(clean_timed))-1)] if clean_timed else None}
    summary['all_saved_attempts_reported_tokens'] = sum((u.get('prompt_tokens') or 0)+(u.get('completion_tokens') or 0) for r in all_rows for u in r['usage'])
    summary['timing_note'] = 'Generation wall time includes application retries and completed or failed outputs; excludes runs with API errors. Retrieval was precomputed; not end-to-end latency.'
    summary['cost_note'] = 'Reported response tokens include reasoning inside completion; no monetary price or missing API usage inferred. Superseded failed runs are retained separately.'
    (OUT/'answer_reviews.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in reviewed),encoding='utf8')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
