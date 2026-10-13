"""Offline auxiliary similarity metrics for completed positive generation cases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
BERT_ID = 'google-bert/bert-base-chinese'
BERT_REVISION = '8f23c25b06e129b6c986331a13d8d025a92cf0ea'
BGE_ID = 'BAAI/bge-small-zh-v1.5'
BGE_REVISION = '7999e1d3359715c523056ef9478215996d62a620'
WEIGHTS_SHA256 = {'bert': '3404a1ffd8da507042e8161013ba2a4fc49858b4e3f8fbf5ce5724f94883aec3',
                  'bge': '354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026'}


def clean_text(text):
    # Citation correctness is scored separately; similarity compares answer prose.
    return re.sub(r'\s+', ' ', re.sub(r'〔[^〕]+〕', '', text)).strip()


def select_pairs(answers, references):
    seen, pairs = set(), []
    for row in answers:
        key = (row['dataset'], row['qid'], row['variant'], row['repeat'])
        if key in seen:
            raise ValueError('Duplicate generation key')
        seen.add(key)
        ref = references.get(key[:2])
        if ref is None:
            raise ValueError('Missing reference case')
        if row['status'] != 'complete' or not ref['positive']:
            continue
        if not isinstance(row['answer'], str) or not clean_text(row['answer']):
            raise ValueError('Complete generation has no answer prose')
        if not isinstance(ref['reference_answer'], str) or not clean_text(ref['reference_answer']):
            raise ValueError('Positive reference has no prose')
        pairs.append({'dataset': key[0], 'qid': key[1], 'variant': key[2], 'repeat': key[3],
                      'candidate': clean_text(row['answer']), 'reference': clean_text(ref['reference_answer'])})
    return sorted(pairs, key=lambda r: (r['dataset'], r['qid'], r['repeat'], r['variant']))


def file_hash(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(1024*1024):
            value.update(block)
    return value.hexdigest()


def portable_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def read_snapshot(path):
    data = path.read_bytes()
    return data.decode('utf8'), hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()


def assert_unchanged(path, expected):
    if portable_hash(path) != expected:
        raise ValueError('Semantic input changed during scoring')


def model_hashes(folder):
    return {str(p.relative_to(folder)).replace('\\', '/'): file_hash(p)
            for p in sorted(folder.rglob('*')) if p.is_file() and '.cache' not in p.relative_to(folder).parts}


def validate_model_hashes(hashes):
    for name, expected in WEIGHTS_SHA256.items():
        if hashes.get(name, {}).get('model.safetensors') != expected:
            raise ValueError(f'{name} weights do not match the pinned evaluation model')
    spec = json.loads((ROOT/'evaluation/semantic_model_artifacts_v1.json').read_text(encoding='utf8'))
    expected = {k: v['files_sha256'] for k, v in spec['models'].items()}
    if hashes != expected:
        raise ValueError('Local model configuration/tokenizer artifacts differ from frozen fingerprints')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--references-dir', type=Path, default=ROOT/'evaluation/generation_fusion_references_v1')
    parser.add_argument('--dependency-dir', type=Path, help='Optional isolated bert-score install directory')
    parser.add_argument('--bert-model', type=Path, required=True, help='Pinned local Chinese BERT directory')
    parser.add_argument('--embed-model', type=Path, required=True, help='Pinned local BGE directory')
    args = parser.parse_args()
    out = args.folder/'supplemental_metrics.jsonl'
    if out.exists():
        raise ValueError('Supplemental output already exists; do not overwrite measured data')
    answers_path = args.folder/'answers.jsonl'
    answer_text, answer_hash = read_snapshot(answers_path)
    answers = [json.loads(s) for s in answer_text.splitlines()]
    references, ref_hashes = {}, {}
    for ds in ('original46', 'development36'):
        text, ref_hashes[ds] = read_snapshot(args.references_dir/f'{ds}.json')
        references.update({(ds, r['qid']): r for r in json.loads(text)})
    pairs = select_pairs(answers, references)
    if not pairs:
        raise ValueError('No completed positive answers to score')
    hashes = {'bert': model_hashes(args.bert_model), 'bge': model_hashes(args.embed_model)}
    validate_model_hashes(hashes)
    if args.dependency_dir:
        sys.path.insert(0, str(args.dependency_dir.resolve()))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import torch
    from bert_score import BERTScorer
    from importlib.metadata import version
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    tick = perf_counter()
    scorer = BERTScorer(model_type=str(args.bert_model.resolve()), num_layers=8,
                       device='cpu', batch_size=8, idf=False, rescale_with_baseline=False, lang='zh')
    embedder = SentenceTransformer(str(args.embed_model.resolve()), device='cpu', local_files_only=True)
    load_seconds = perf_counter()-tick
    candidates, refs = [r['candidate'] for r in pairs], [r['reference'] for r in pairs]
    token_counts = lambda tokenizer, strings: [len(tokenizer(s, truncation=False)['input_ids']) for s in strings]
    bert_lengths = token_counts(scorer._tokenizer, candidates+refs)
    bge_lengths = token_counts(embedder.tokenizer, candidates+refs)
    tick = perf_counter()
    precision, recall, f1 = scorer.score(candidates, refs, batch_size=8)
    vectors = embedder.encode(candidates+refs, batch_size=8, normalize_embeddings=True, show_progress_bar=False)
    elapsed = perf_counter()-tick
    cosine = (vectors[:len(pairs)]*vectors[len(pairs):]).sum(axis=1)
    assert_unchanged(answers_path, answer_hash)
    for ds, expected in ref_hashes.items():
        assert_unchanged(args.references_dir/f'{ds}.json', expected)
    if hashes != {'bert': model_hashes(args.bert_model), 'bge': model_hashes(args.embed_model)}:
        raise ValueError('Model artifacts changed during scoring')
    with out.open('w', encoding='utf8') as stream:
        for i, row in enumerate(pairs):
            record = {k: row[k] for k in ('dataset', 'qid', 'variant', 'repeat')}
            record.update(status='complete', answer_sha256=hashlib.sha256(row['candidate'].encode()).hexdigest(),
                          reference_sha256=hashlib.sha256(row['reference'].encode()).hexdigest(),
                          semantic_similarity=float(cosine[i]),
                          bertscore={'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i])},
                          bert_tokens={'answer': bert_lengths[i], 'reference': bert_lengths[i+len(pairs)],
                                       'max_length': int(scorer._tokenizer.model_max_length)},
                          bge_tokens={'answer': bge_lengths[i], 'reference': bge_lengths[i+len(pairs)],
                                      'max_length': int(embedder.max_seq_length)})
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
    manifest = {'bert_model_id': BERT_ID, 'bert_model_revision': BERT_REVISION,
                'bert_score_version': version('bert-score'), 'bert_score_hash': scorer.hash,
                'num_layers': 8, 'idf': False, 'rescale_with_baseline': False,
                'embed_model_id': BGE_ID, 'embed_model_revision': BGE_REVISION,
                'model_files_sha256': hashes,
                'answers_sha256': answer_hash,
                'reference_sha256': ref_hashes,
                'model_artifact_spec_sha256': portable_hash(ROOT/'evaluation/semantic_model_artifacts_v1.json'),
                'versions': {p: version(p) for p in ('torch', 'transformers', 'sentence-transformers')},
                'device': 'cpu', 'cpu_threads': 4, 'batch_size': 8, 'scored_pairs': len(pairs),
                'model_load_seconds': load_seconds, 'scoring_seconds': elapsed,
                'scope': 'Completed positive cases only; no score inferred for failed or diagnostic cases',
                'preprocessing': 'Remove source markers and collapse whitespace; preserve negation and numbers',
                'truncation': 'Tokenizer/model native limits; full original answers retained in answers.jsonl',
                'source_sha256': portable_hash(Path(__file__)),
                'data_code_hash_normalization': 'CRLF to LF only; model file hashes remain raw SHA256'}
    (args.folder/'semantic_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'scored_pairs': len(pairs), 'scoring_seconds': elapsed}, ensure_ascii=False))


if __name__ == '__main__':
    main()
