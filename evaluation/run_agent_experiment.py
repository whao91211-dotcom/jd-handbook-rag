"""Serial paired development evaluation. Labels are used only after runtime returns."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_case(case, profile, runner):
    result = runner(case['question'])  # The only input to runtime: never labels or IDs.
    gathered = [s['id'] for s in result.get('sources', [])]
    context = [s['id'] for s in result.get('sources', []) if s.get('in_context')]
    gold = set(case.get('gold_chunk_ids', []))
    return {**result, 'qid': case['qid'], 'profile': profile, 'gold_ids': sorted(gold),
            'gathered_ids': gathered, 'context_ids': context,
            'gathered_full_coverage': gold <= set(gathered) if gold else None,
            'context_full_coverage': gold <= set(context) if gold else None,
            'review_points': case.get('review_points', []),
            'model_review': None, 'human_adjudication': None}


def distribution(rows):
    times = sorted(r['metrics']['total_seconds'] for r in rows if r.get('metrics', {}).get('total_seconds') is not None)
    tokens = [r['metrics']['total_tokens'] for r in rows if r.get('metrics', {}).get('total_tokens') is not None]
    gold_rows = [r for r in rows if r.get('context_full_coverage') is not None]
    return {'n': len(rows), 'status_counts': dict(Counter(r['status'] for r in rows)),
            'total_seconds_p50': statistics.median(times) if times else None,
            'total_seconds_p95': times[max(0, __import__('math').ceil(.95*len(times))-1)] if times else None,
            'known_usage_n': len(tokens), 'tokens_mean': statistics.mean(tokens) if tokens else None,
            'gold_n': len(gold_rows), 'context_full_coverage_n': sum(r['context_full_coverage'] for r in gold_rows),
            'gathered_full_coverage_n': sum(r['gathered_full_coverage'] for r in gold_rows)}


def summarize(rows):
    profiles = sorted({r['profile'] for r in rows})
    groups = {profile: [r for r in rows if r['profile'] == profile] for profile in profiles}
    eligible = set.intersection(*(set(r['qid'] for r in group if r['status'] == 'complete')
                                 for group in groups.values())) if groups else set()
    return {'profiles': {p: distribution(rs) for p, rs in groups.items()},
            'paired_complete_n': len(eligible), 'paired_complete_ids': sorted(eligible),
            'paired': {p: distribution([r for r in rs if r['qid'] in eligible]) for p, rs in groups.items()},
            'correctness_note': 'No answer correctness score until independent model review or human adjudication. Coverage is not correctness.'}


def quality_answer(question):
    from ragcore.retriever import retrieve
    from ragcore.llm import generate, prepare_context
    started = perf_counter()
    chunks = retrieve(question, k=8, strategy='guarded')
    retrieval_seconds = perf_counter()-started
    context, mapping = prepare_context(chunks, 'evidence')
    reverse = {i: label for label, i in mapping.items()}
    usage = []
    tick = perf_counter()
    answer, status, error_type = '', 'complete', None
    try:
        answer = generate(question, chunks, prompt_version='evidence', thinking_mode='low', usage_sink=usage)
        if not context:
            status = 'no_evidence'
    except Exception as exc:
        status, error_type = 'generation_failed', type(exc).__name__
    generation_seconds = perf_counter()-tick
    known = bool(usage) and all(u.get('prompt_tokens') is not None and u.get('completion_tokens') is not None for u in usage)
    return {'question': question, 'profile': 'quality', 'answer': answer, 'status': status, 'error_type': error_type,
        'sources': [{'id': c['id'], 'label': reverse.get(c['id']), 'text': c['text'], 'pages': c['pages'],
                     'path': c['path'], 'in_context': c['id'] in reverse} for c in chunks],
        'source_map': mapping, 'context': context, 'usage': usage, 'trace': [],
        'metrics': {'retrieval_seconds': retrieval_seconds, 'generation_seconds': generation_seconds,
                    'total_seconds': perf_counter()-started, 'attempts': len(usage),
                    'total_tokens': sum(u['prompt_tokens']+u['completion_tokens'] for u in usage) if known else None}}


def main():
    parser = argparse.ArgumentParser(description='Paired quality/agent development evaluation')
    parser.add_argument('--cases', type=Path, default=ROOT/'evaluation/complex_cases_v1.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT/'evaluation/agent_experiment_2026-10-04')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--profiles', nargs='+', choices=['quality', 'agent'], default=['quality', 'agent'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [json.loads(s) for s in args.cases.read_text(encoding='utf-8').splitlines() if s.strip()]
    if args.limit is not None:
        cases = cases[:args.limit]
    from ragcore.agent import answer_question
    from ragcore.retriever import retrieve
    from ragcore.llm import env_config
    retrieve('病假', k=8, strategy='guarded')  # Warm-up excluded from request timings.
    metadata = {'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'corpus_sha256': hashlib.sha256((ROOT/'data/chunks.jsonl').read_bytes()).hexdigest(),
                'cases_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                'runtime_sha256': hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'ragcore').glob('*.py')))).hexdigest(),
                'configured_model': env_config()['model'], 'profiles': args.profiles,
                'timing_note': 'Serial paired requests; warmed retrieval; includes orchestration and synthesis; first LlamaIndex import is warmed separately.',
                'dataset_note': 'Simulated development probes. Gold references checked against local source chunks; not human answer adjudication or held-out real-user benchmark.'}
    # Warm only imports, without making a model call.
    from llama_index.llms.openai_like import OpenAILike
    (args.output/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    raw_path = args.output/'answers.jsonl'
    with raw_path.open('w', encoding='utf-8') as stream:
        for n, case in enumerate(cases):
            # Alternate order to reduce a systematic provider/timing-order bias.
            order = args.profiles if n % 2 == 0 else list(reversed(args.profiles))
            for profile in order:
                row = run_case(case, profile, quality_answer if profile == 'quality' else answer_question)
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                stream.flush()
                print(json.dumps({'qid': case['qid'], 'profile': profile, 'status': row['status'],
                                  'metrics': row['metrics'], 'context_full_coverage': row['context_full_coverage']}, ensure_ascii=False), flush=True)
                if any(u.get('status_code') in (401, 402, 403, 429) for u in row.get('usage', [])):
                    metadata['stopped_for_api_status'] = True
                    break
            if metadata.get('stopped_for_api_status'):
                break
    (args.output/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output/'summary.json').write_text(json.dumps(summarize(rows), ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Answer review worksheet', '', 'Model review and human adjudication are separate; no scores have been filled.', '']
    for case in cases:
        lines += [f"## {case['qid']}: {case['question']}", '', *('- '+point for point in case.get('review_points', [])),
                  '', 'Model review: unreviewed. Human adjudication: unreviewed.', '']
    (args.output/'review.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
