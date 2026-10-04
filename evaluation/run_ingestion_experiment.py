"""Local chunking ablation; no OCR/LLM calls and no production index writes."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_parents(rows):
    groups, references = {}, {}
    for row in rows:
        key = (row['chapter'], row['section'])
        if key not in groups:
            groups[key] = {'parent': f'section-{len(groups)}', 'path': ' > '.join(x for x in key if x),
                           'content': '', 'page_spans': []}
        parent = groups[key]
        if parent['content']:
            parent['content'] += '\n\n'
        start = len(parent['content'])
        parent['content'] += row['content']
        span = {'parent': parent['parent'], 'start': start, 'end': len(parent['content'])}
        references[row['id']] = span
        parent['page_spans'].append({**span, 'pages': row['pages']})
    return list(groups.values()), references


def covered_fraction(reference, hits):
    intervals = sorted((max(reference['start'], h['start']), min(reference['end'], h['end']))
                       for h in hits if h['parent'] == reference['parent']
                       and h['end'] > reference['start'] and h['start'] < reference['end'])
    covered, end = 0, reference['start']
    for left, right in intervals:
        covered += max(0, right-max(left, end))
        end = max(end, right)
    return covered / (reference['end']-reference['start'])


def make_chunks(parents, chunk_size, overlap):
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap,
                                include_metadata=False, include_prev_next_rel=False)
    chunks, spans = [], {}
    for parent in parents:
        nodes = splitter.get_nodes_from_documents([Document(text=parent['content'])])
        cursor = 0
        for node in nodes:
            content = node.text
            start = parent['content'].find(content, cursor)
            if start < 0:
                raise ValueError('Splitter output cannot be located exactly in parent text')
            cursor = start+1
            span = {'parent': parent['parent'], 'start': start, 'end': start+len(content)}
            pages = sorted({p for s in parent['page_spans'] if s['end'] > span['start']
                            and s['start'] < span['end'] for p in s['pages']})
            iid = 'sentence-'+hashlib.sha256(f"{chunk_size}:{span}".encode()).hexdigest()[:16]
            spans[iid] = span
            chunks.append({'id': iid, 'content': content, 'text': f"【{parent['path']}】\n{content}",
                           'path': parent['path'], 'pages': pages, 'clause_range': '', 'char_len': len(content)})
    return chunks, spans


def score(case, hits, spans, references):
    # Labels are used only here, after retrieval has returned.
    gold = case['gold_chunk_ids']
    if not gold:
        return None
    fractions = [covered_fraction(references[iid], [spans[h['id']] for h in hits]) for iid in gold]
    return {'reference_char_coverage': statistics.mean(fractions),
            'reference_block_recall': sum(f >= 1-1e-12 for f in fractions)/len(fractions),
            'all_reference_blocks_covered': all(f >= 1-1e-12 for f in fractions),
            'fractions': dict(zip(gold, fractions))}


def summarize(rows):
    summary = {}
    for variant in sorted({r['variant'] for r in rows}):
        for strategy in ('baseline', 'guarded'):
            for dataset in sorted({r['dataset'] for r in rows}):
                group = [r for r in rows if (r['variant'], r['strategy'], r['dataset']) == (variant, strategy, dataset)]
                positives = [r for r in group if r['at8'] is not None]
                entry = {'n': len(group), 'positive_n': len(positives),
                         'retrieval_seconds_p50': statistics.median(r['seconds'] for r in group),
                         'top8_text_chars_mean': statistics.mean(r['top8_chars'] for r in group)}
                for stage in ('at3', 'at8', 'context8'):
                    entry[stage] = {metric: statistics.mean(r[stage][metric] for r in positives)
                        for metric in ('reference_char_coverage', 'reference_block_recall', 'all_reference_blocks_covered')}
                summary[f'{variant}/{strategy}/{dataset}'] = entry
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'evaluation/ingestion_experiment_2026-10-04')
    parser.add_argument('--datasets', nargs='+', choices=['old36', 'complex6', 'retrieval46'],
                        default=['old36', 'complex6'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from ragcore import retriever
    from ragcore.store import add_chunks, get_collection
    from ragcore.embedder import embed_texts, get_model
    from ragcore.llm import prepare_context
    import chromadb

    original = [json.loads(s) for s in (ROOT/'data/chunks.jsonl').read_text(encoding='utf8').splitlines()]
    parents, references = build_parents(original)
    cases = []
    paths = {'old36': ROOT/'evaluation/baseline_v1_2026-09-26/cases.jsonl',
             'complex6': ROOT/'evaluation/complex_cases_v1.jsonl',
             'retrieval46': ROOT/'data/eval/golden.jsonl'}
    for name in args.datasets:
        for line in paths[name].read_text(encoding='utf8').splitlines():
            case = json.loads(line)
            case['dataset_group'] = name
            assert all(i in references for i in case['gold_chunk_ids'])
            cases.append(case)
    get_model()  # Exclude model cold load from measured ingestion/retrieval.
    original_collection = get_collection(create=False)
    before_ids = sorted(original_collection.get()['ids'])
    metadata = {'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'chunks_sha256': hashlib.sha256((ROOT/'data/chunks.jsonl').read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'llama_index_core': __import__('importlib.metadata', fromlist=['version']).version('llama-index-core'),
        'cases_sha256': {name: hashlib.sha256(paths[name].read_bytes()).hexdigest() for name in args.datasets},
        'embedding_model': 'BAAI/bge-small-zh-v1.5', 'embedding_max_sequence_length': get_model().max_seq_length,
        'note': 'Same extracted text, rebuilt section parents from existing contents; NOT a PDF parsing/OCR comparison. '
                'Token-based splitter with default tokenizer; BGE uses another tokenizer. '
                'Reference interval metrics are not answer correctness. Simulated development sets, one retrieval per condition.',
        'variants': {}}
    all_rows = []
    (args.output/'cases.jsonl').write_text(''.join(json.dumps({k: c[k] for k in
        ('qid', 'question', 'gold_chunk_ids', 'dataset_group')}, ensure_ascii=False)+'\n' for c in cases), encoding='utf8')
    for variant, size, overlap in [('current', None, None), ('sentence256', 256, 32), ('sentence512', 512, 64)]:
        tick = perf_counter()
        if size is None:
            chunks, spans, collection = original, references, original_collection
            build_seconds = None  # Existing index: no fabricated comparable build time.
        else:
            chunks, spans = make_chunks(parents, size, overlap)
            collection = chromadb.EphemeralClient().create_collection('ingestion-'+variant,
                            metadata={'hnsw:space': 'cosine'})
            add_chunks(collection, chunks, embed_texts(c['text'] for c in chunks))
            build_seconds = perf_counter()-tick
        metadata['variants'][variant] = {'chunk_size_tokens': size, 'overlap_tokens': overlap,
            'n_chunks': len(chunks), 'build_seconds_warm': build_seconds,
            'char_len_mean': statistics.mean(c['char_len'] for c in chunks),
            'bge_truncated_chunk_n': sum(len(get_model().tokenizer.encode(c['text'], truncation=False)) > get_model().max_seq_length for c in chunks)}
        (args.output/f'{variant}_chunks.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in chunks), encoding='utf8')
        data = collection.get(include=['documents', 'metadatas'])
        with patch.object(retriever, 'get_collection', return_value=collection), \
             patch.object(retriever, '_corpus', (data['ids'], data['documents'], data['metadatas'])), \
             patch.object(retriever, '_bm25_cache', None):
            retriever.retrieve('病假', k=8)  # Warm BM25 index before timing.
            for strategy in ('baseline', 'guarded'):
                for case in cases:
                    tick = perf_counter()
                    hits = retriever.retrieve(case['question'], k=8, strategy=strategy)
                    seconds = perf_counter()-tick
                    context, mapping = prepare_context(hits, 'evidence')
                    admitted = [h for h in hits if h['id'] in mapping.values()]
                    all_rows.append({'variant': variant, 'strategy': strategy, 'dataset': case['dataset_group'],
                        'qid': case['qid'], 'question': case['question'], 'seconds': seconds,
                        'hit_ids': [h['id'] for h in hits], 'top8_chars': sum(len(h['text']) for h in hits),
                        'context_chars': len(context), 'context_ids': [h['id'] for h in admitted],
                        'at3': score(case, hits[:3], spans, references),
                        'at8': score(case, hits, spans, references),
                        'context8': score(case, admitted, spans, references)})
        print(json.dumps({'variant': variant, **metadata['variants'][variant]}, ensure_ascii=False), flush=True)
    assert before_ids == sorted(original_collection.get()['ids'])
    metadata['production_index_ids_unchanged'] = True
    for filename, value in [('metadata.json', metadata), ('summary.json', summarize(all_rows))]:
        (args.output/filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    (args.output/'retrievals.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in all_rows), encoding='utf8')


if __name__ == '__main__':
    main()
