"""Reproduce per-case fusion diagnostics and timing drift from frozen runs only."""
import argparse
import json
import statistics
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--baseline', type=Path,
                        default=Path(__file__).resolve().parent/'rerank_experiment_2026-10-12')
    args = parser.parse_args()
    rows = read_rows(args.folder/'runs.jsonl')
    old = read_rows(args.baseline/'runs.jsonl')
    key = lambda r: (r['dataset'], r['qid'], r['variant'], r['repeat'])
    current = {key(r): r for r in rows}
    assert len(current) == len(rows) == 984
    assert all(r['status'] == 'complete' for r in rows)
    # Exact historical quality, candidate, score and selection compatibility;
    # timing is deliberately excluded because it is freshly measured.
    checked = ('candidate_ids', 'scores', 'pair_tokens', 'selected_ids', 'context_ids', 'metrics')
    for previous in old:
        assert all(current[key(previous)][f] == previous[f] for f in checked), key(previous)
    result = {'historical_abc_runs_compared': len(old), 'historical_fields_match': list(checked),
              'datasets': {}}
    for ds in ('original46', 'development36'):
        subset = [r for r in rows if r['dataset'] == ds]
        positives = sorted({r['qid'] for r in subset if r['gold_ids']})
        entry = {'positive_cases': len(positives), 'diagnostic_cases': len({r['qid'] for r in subset})-len(positives),
                 'timings_by_repeat': {}, 'fusion_overhead_ms': {}, 'cases': []}
        for v in 'ABCD':
            entry['timings_by_repeat'][v] = [
                {'repeat': rep+1, 'total_p50_seconds': statistics.median(
                    r['total_seconds'] for r in subset if r['variant'] == v and r['repeat'] == rep)}
                for rep in range(3)]
        for stage in ('guard_seconds', 'fusion_seconds'):
            samples = [1000*r[stage] for r in subset if r['variant'] == 'D']
            entry['fusion_overhead_ms'][stage] = {
                'p50': statistics.median(samples),
                'p95': statistics.quantiles(samples, n=100, method='inclusive')[94]}
        for qid in positives:
            variants = {v: current[(ds, qid, v, 0)] for v in 'ABCD'}
            d = variants['D']
            case = {'qid': qid, 'question': d['question'],
                    'metrics': {v: r['metrics'] for v, r in variants.items()},
                    'delta_D_vs_A': {m: d['metrics'][m]-variants['A']['metrics'][m] for m in d['metrics']},
                    'delta_D_vs_B': {m: d['metrics'][m]-variants['B']['metrics'][m] for m in d['metrics']},
                    'gold_positions': []}
            for iid in d['gold_ids']:
                case['gold_positions'].append({
                    'id': iid, 'guard_rank': d['guard_ranks'][iid],
                    'semantic_rank': d['semantic_ranks'][iid], 'fusion_score': d['fusion_scores'][iid],
                    'pair_tokens': d['pair_tokens'][iid],
                    'selected_rank': {v: r['selected_ids'].index(iid)+1 if iid in r['selected_ids'] else None
                                      for v, r in variants.items()}})
            entry['cases'].append(case)
        result['datasets'][ds] = entry
    (args.folder/'case_analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({ds: {k: v for k, v in e.items() if k != 'cases'}
                      for ds, e in result['datasets'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
