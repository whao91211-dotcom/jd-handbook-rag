"""Reproducible development-set ablations; never passes gold labels to retrieval."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'evaluation' / 'evidence_experiment_2026-09-26'
    out.mkdir(exist_ok=True)
    print('Loading retrieval runtime...', flush=True)
    from ragcore.retriever import retrieve, _load_corpus
    from ragcore.llm import prepare_context, generate, SYSTEM_PROMPT, EVIDENCE_PROMPT
    cases = read(ROOT / 'evaluation/baseline_v1_2026-09-26/cases.jsonl')
    old = [x for x in read(ROOT / 'data/eval/golden.jsonl') if x.get('gold_chunk_ids')]
    frozen = read(ROOT / 'evaluation/baseline_v1_2026-09-26/chunks.jsonl')
    ids, docs, _ = _load_corpus()
    actual = dict(zip(ids, docs))
    assert all(actual[x['id']] == x['text'] for x in frozen) and len(actual) == len(frozen)
    saved = json.loads((ROOT / 'evaluation/baseline_v1_2026-09-26/retrieval_saved.json').read_text(encoding='utf8'))
    expected = {x['qid']: x['retrieved_ids'] for x in saved['modes']['rrf']['rows']}
    retrieve(cases[0]['question'])  # warm-up excluded from request timing
    print('Runtime warm; corpus verified.', flush=True)
    rows = []
    prepared = {}
    for dataset, questions in [('development36', cases), ('original35', old)]:
        for case in questions:
            for strategy in ('baseline', 'expanded', 'coverage'):
                start = time.perf_counter()
                chunks = retrieve(case['question'], strategy=strategy)
                elapsed = time.perf_counter() - start
                hit_ids = [x['id'] for x in chunks]
                if dataset == 'development36' and strategy == 'baseline':
                    assert hit_ids == expected[case['qid']], case['qid']
                gold = set(case.get('gold_chunk_ids', []))
                row = {'dataset': dataset, 'qid': case['qid'], 'kind': case.get('kind'),
                       'strategy': strategy, 'retrieved_ids': hit_ids, 'gold_ids': sorted(gold),
                       'queries': chunks[0]['queries'] if chunks else [], 'retrieval_seconds': elapsed,
                       'any_at3': bool(gold & set(hit_ids[:3])), 'any_at8': bool(gold & set(hit_ids)),
                       'full_at8': bool(gold) and gold <= set(hit_ids)}
                rows.append(row)
                if dataset == 'development36':
                    prepared[(case['qid'], strategy)] = (chunks, row)
        print(f'{dataset}: completed {len(questions)} questions x 3 strategies', flush=True)
    save(out / 'retrieval.json', rows)
    summary = {}
    for dataset in ('development36', 'original35'):
        for strategy in ('baseline', 'expanded', 'coverage'):
            eligible = [x for x in rows if x['dataset'] == dataset and x['strategy'] == strategy
                        and (dataset != 'development36' or x['kind'] == 'naturalistic_positive')]
            summary[f'{dataset}/{strategy}'] = {'n': len(eligible), **{
                metric: sum(x[metric] for x in eligible) for metric in ('any_at3', 'any_at8', 'full_at8')}}
    save(out / 'retrieval_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not args.generate:
        return
    raw = out / 'answers.jsonl'
    completed = {(x['variant'], x['qid']) for x in read(raw)} if raw.exists() else set()
    variants = [('A', 'baseline', 'baseline'), ('B', 'expanded', 'baseline'),
                ('C', 'coverage', 'baseline'), ('D', 'coverage', 'evidence')]
    def run(case, variant, strategy, prompt):
        chunks, retrieval = prepared[(case['qid'], strategy)]
        context, source_map = prepare_context(chunks, prompt)
        system = SYSTEM_PROMPT if prompt == 'baseline' else EVIDENCE_PROMPT
        usage = []
        start = time.perf_counter()
        error = None
        try:
            answer = generate(case['question'], chunks, prompt_version=prompt, usage_sink=usage)
        except Exception as exc:
            answer, error = '', type(exc).__name__
        return {'variant': variant, 'qid': case['qid'], 'question': case['question'],
                'strategy': strategy, 'prompt_version': prompt, 'context': context,
                'source_map': source_map, 'retrieved_ids': retrieval['retrieved_ids'],
                'system_prompt': system, 'system_sha256': digest(system), 'context_sha256': digest(context),
                'answer': answer, 'error': error, 'usage': usage,
                'generation_seconds': time.perf_counter()-start,
                'retrieval_seconds': retrieval['retrieval_seconds'],
                'timing_note': 'retrieval warm and precomputed; not a live end-to-end measurement'}
    tasks = [(case, *v) for case in cases for v in variants if (v[0], case['qid']) not in completed]
    print(f'Generating {len(tasks)} remaining answers with 3 workers', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool, raw.open('a', encoding='utf8') as stream:
        futures = [pool.submit(run, *task) for task in tasks]
        for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            stream.write(json.dumps(result, ensure_ascii=False)+'\n')
            stream.flush()
            print(f"{number}/{len(tasks)} {result['variant']} {result['qid']} error={result['error']}", flush=True)

if __name__ == '__main__':
    main()
