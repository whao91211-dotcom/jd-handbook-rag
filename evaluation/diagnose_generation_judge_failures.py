"""Explain recorded judge failures offline, without repairing responses or scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import generation_metrics as metrics


def fingerprint(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def diagnose(folder):
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    expected_metrics = manifest['code_sha256']['evaluation/generation_metrics.py']
    if fingerprint(ROOT/'evaluation/generation_metrics.py') != expected_metrics:
        raise ValueError('Frozen metric validation code changed')
    sources = manifest['source_directories']
    hashes = manifest['source_file_sha256']
    for name, expected in hashes.items():
        if fingerprint(ROOT/name) != expected:
            raise ValueError('Frozen result source changed')
    result = {'scope': 'Offline schema/logic diagnostics only; no model calls, response repair or score replacement.',
              'metrics_code_sha256': fingerprint(ROOT/'evaluation/generation_metrics.py'), 'sources': hashes, 'judges': {}}
    for kind in ('context', 'faithfulness', 'correctness'):
        source = ROOT/sources['correctness' if kind=='correctness' else 'generation']/f'{kind}_reviews.jsonl'
        rows = [json.loads(line) for line in source.read_text(encoding='utf8').splitlines() if line.strip()]
        details = []
        for row in rows:
            payload = json.loads(row['messages'][-1]['content'])
            for attempt in row['attempts']:
                if attempt['status']=='complete':
                    continue
                reason = None
                if attempt.get('failure_class')=='invalid_response':
                    try:
                        if attempt.get('finish_reason')!='stop':
                            raise ValueError('Judge did not complete normally')
                        review = json.loads(attempt['raw_response'])
                        if kind=='correctness':
                            metrics.correctness_scores(review, payload)
                        elif kind=='faithfulness':
                            metrics.faithfulness_scores(review, [x['label'] for x in payload['contexts']])
                        else:
                            reference = {'positive': True, 'reference_claims': payload['reference_claims']}
                            metrics.context_scores(review, reference, len(payload['contexts']))
                        reason = 'Validation no longer reproduces failure; inspect frozen contract'
                    except (ValueError, TypeError, KeyError) as exc:
                        reason = str(exc)
                else:
                    error = attempt.get('error') or {}
                    reason = f"API/transport error: {error.get('type')}; HTTP {error.get('status_code')}"
                details.append({k: row[k] for k in ('dataset','qid','variant','status')} |
                    {'repeat': row.get('repeat'), 'attempt': attempt['attempt'], 'reason': reason})
        terminal = {tuple(d[k] for k in ('dataset','qid','variant','repeat')):d
                    for d in details if d['status']=='failed'}
        result['judges'][kind] = {'record_status': dict(Counter(r['status'] for r in rows)),
            'terminal_failure_reasons': dict(Counter(d['reason'] for d in terminal.values())),
            'failed_attempt_reasons': dict(Counter(d['reason'] for d in details)), 'failed_attempts': details}
    for name, expected in hashes.items():
        if fingerprint(ROOT/name) != expected:
            raise ValueError('Frozen result source changed during diagnostics')
    if fingerprint(ROOT/'evaluation/generation_metrics.py') != expected_metrics:
        raise ValueError('Frozen metric validation code changed during diagnostics')
    (folder/'judge_failure_diagnostics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({kind: value['failed_attempt_reasons'] for kind,value in result['judges'].items()}, ensure_ascii=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    diagnose(parser.parse_args().folder)
