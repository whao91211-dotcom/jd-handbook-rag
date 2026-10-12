"""Audit frozen experiment artifacts without loading models or calling retrieval."""
import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = args.folder
    rows = [json.loads(line) for line in (folder/'runs.jsonl').read_text(encoding='utf8').splitlines()]
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    summary = json.loads((folder/'summary.json').read_text(encoding='utf8'))
    assert manifest['repeats'] == 3 and manifest['smoke_limit'] is None
    content_path = folder/'content_hashes.json'
    normalized_hash = lambda data: hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()
    if not content_path.exists():
        # First audit must match the actual measured bytes before deriving portable hashes.
        for name, expected in manifest['source_sha256'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
        assert hashlib.sha256((folder/'chunks.jsonl').read_bytes()).hexdigest() == manifest['chunks_sha256']
        content = {'normalization': 'CRLF to LF only; no other byte changes',
                   'measured_code_revision': manifest['code_revision'],
                   'source_sha256': {name: normalized_hash((root/name).read_bytes())
                                     for name in manifest['source_sha256']},
                   'chunks_sha256': normalized_hash((folder/'chunks.jsonl').read_bytes())}
        content_path.write_text(json.dumps(content, indent=2)+'\n', encoding='utf8')
    content = json.loads(content_path.read_text(encoding='utf8'))
    for name, expected in content['source_sha256'].items():
        assert normalized_hash((root/name).read_bytes()) == expected, name
    assert normalized_hash((folder/'chunks.jsonl').read_bytes()) == content['chunks_sha256']
    cases = {ds: json.loads((folder/f'{ds}.json').read_text(encoding='utf8')) for ds in summary}
    assert {ds: len(cs) for ds, cs in cases.items()} == {'original46': 46, 'development36': 36}
    expected_keys = {(ds, c['qid'], v, rep) for ds, cs in cases.items()
                     for c in cs for v in 'ABC' for rep in range(3)}
    keys = [(r['dataset'], r['qid'], r['variant'], r['repeat']) for r in rows]
    assert len(keys) == 738 and len(set(keys)) == 738 and set(keys) == expected_keys
    groups = defaultdict(list)
    for r in rows:
        assert r['status'] == 'complete', (r['qid'], r.get('error'))
        pool, selected = r['candidate_ids'], r['selected_ids']
        assert len(pool) == len(set(pool)) <= 40
        assert len(selected) == len(set(selected)) == 8 and set(selected) <= set(pool)
        assert pool == [iid for iid in r['fused_ids'] if iid in set(pool)]
        assert set(a for a in r['lexical_anchors'] if a) <= set(pool)
        assert set(r['context_ids']) <= set(selected) and r['context_chars'] <= 6000
        assert set(r['selection_reasons']) == set(selected)
        assert all(math.isfinite(r[s]) and r[s] >= 0
                   for s in ('recall_seconds', 'rerank_seconds', 'total_seconds'))
        assert r['total_seconds'] >= r['recall_seconds'] + r['rerank_seconds']
        if r['variant'] == 'A':
            assert not r['scores'] and not r['pair_tokens'] and r['rerank_seconds'] == 0
        else:
            assert set(r['scores']) == set(pool) == set(r['pair_tokens'])
            assert all(math.isfinite(s) for s in r['scores'].values())
            order = sorted(pool, key=lambda iid: -r['scores'][iid])
            assert order == r['ranked_candidates']
            if r['variant'] == 'B':
                assert selected == order[:8]
        groups[(r['dataset'], r['qid'])].append(r)
    for records in groups.values():
        assert len({tuple(r['candidate_ids']) for r in records}) == 1
        for v in 'ABC':
            assert len({tuple(r['selected_ids']) for r in records if r['variant'] == v}) == 1
        scored = [r for r in records if r['variant'] != 'A']
        assert all(r['scores'] == scored[0]['scores'] for r in scored)
    # Independently recompute quality from final IDs; repeated measurements do not increase n.
    recalculated = {}
    for ds in cases:
        recalculated[ds] = {}
        for v in 'ABC':
            eligible = [r for r in rows if r['dataset'] == ds and r['variant'] == v and r['repeat'] == 0 and r['gold_ids']]
            values = defaultdict(list)
            for r in eligible:
                gold = set(r['gold_ids'])
                flags = [iid in gold for iid in r['selected_ids']]
                values['ndcg_at3'].append(sum(flag/math.log2(i+2) for i, flag in enumerate(flags[:3])) /
                    sum(1/math.log2(i+2) for i in range(min(3, len(gold)))))
                values['mrr'].append(next((1/(i+1) for i, flag in enumerate(flags) if flag), 0))
                values['recall_at8'].append(sum(flags)/len(gold))
                values['full_at8'].append(float(gold <= set(r['selected_ids'])))
                values['context_full'].append(float(gold <= set(r['context_ids'])))
            recalculated[ds][v] = {m: statistics.mean(vs) for m, vs in values.items()}
            assert len(eligible) == summary[ds]['variants'][v]['positive_cases']
            for m, value in recalculated[ds][v].items():
                assert math.isclose(value, summary[ds]['variants'][v]['metrics'][m], abs_tol=1e-12)
            measured = [r for r in rows if r['dataset'] == ds and r['variant'] == v]
            for stage in ('recall_seconds', 'rerank_seconds', 'total_seconds'):
                samples = [r[stage] for r in measured]
                timings = summary[ds]['variants'][v]['timings'][stage]
                assert math.isclose(statistics.median(samples), timings['p50'], abs_tol=1e-12)
                assert math.isclose(statistics.quantiles(samples, n=100, method='inclusive')[94],
                                    timings['p95'], abs_tol=1e-12)
    result = {'successful_runs': len(rows), 'failed_runs': 0, 'paired_cases': len(groups),
              'run_counts': {ds: dict(Counter(r['variant'] for r in rows if r['dataset'] == ds)) for ds in cases},
              'candidate_pools_identical': True, 'rankings_and_scores_identical_across_repeats': True,
              'source_content_hashes_match': True, 'hash_normalization': content['normalization'],
              'pool_and_result_bounds_verified': True,
              'timing_percentiles_recomputed': True,
              'context_dropped_chunks': sum(len(r['selected_ids'])-len(r['context_ids']) for r in rows),
              'recomputed_metrics': recalculated}
    (folder/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
