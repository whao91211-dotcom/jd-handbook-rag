"""PDF parser ablation with source-page probes; never writes production data/index."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import unicodedata
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text) if not c.isspace() and c != '|')


def audit_page(text, anchors, table_tuples=()):
    normalized = normalize(text)
    positions = [normalized.find(normalize(a)) for a in anchors]
    pairs = [(positions[i], positions[i+1]) for i in range(len(positions)-1)]
    tuples = [''.join(normalize(s) for s in row) in normalized for row in table_tuples]
    return {'anchor_n': len(anchors), 'present_n': sum(p >= 0 for p in positions),
            'missing': [a for a, p in zip(anchors, positions) if p < 0], 'positions': positions,
            'order_complete': all(p >= 0 for p in positions) and positions == sorted(positions),
            'order_pair_n': len(pairs), 'correct_order_pair_n': sum(0 <= a < b for a, b in pairs),
            'table_tuple_total': len(tuples), 'table_tuple_n': sum(tuples),
            'table_tuple_matches': tuples}


def evidence_coverage(probes, hits):
    satisfied = []
    for probe in probes:
        text = normalize('\n'.join(h['text'] for h in hits if probe['page'] in h['pages']))
        satisfied.append(all(normalize(f) in text for f in probe['fragments']))
    return statistics.mean(satisfied) if satisfied else None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'evaluation/pdf_parser_2026-10-04')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from ragcore import extract, chunker, retriever
    from ragcore.store import add_chunks, get_collection
    from ragcore.embedder import embed_texts, get_model
    from ragcore.llm import prepare_context
    from llama_index.readers.file import PDFReader
    import pdfplumber
    import chromadb

    pdf = ROOT/'京东集团员工手册.pdf'
    probes_path = ROOT/'evaluation/pdf_parser_probes_v1.json'
    probes = json.loads(probes_path.read_text(encoding='utf8'))
    production_files = [ROOT/'data/pages.jsonl', ROOT/'data/chunks.jsonl']
    before = {str(p.relative_to(ROOT)): digest(p) for p in production_files}
    original_ids = sorted(get_collection(create=False).get()['ids'])
    metadata = {'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'script_sha256': digest(Path(__file__)), 'pdf_sha256': digest(pdf), 'probes_sha256': digest(probes_path),
                'versions': {p: version(p) for p in ['llama-index-core', 'llama-index-readers-file', 'pypdf', 'pdfplumber', 'pypdfium2']},
                'note': 'Native text PDF. No OCR/cloud/answer generation. Source visual spot checks, not full transcription. '
                        'Same cleaning/chunker/BGE/Chroma/BM25/RRF; parser outputs retain their own layout/structured-table differences.'}
    tick = perf_counter()
    with pdfplumber.open(pdf) as source:
        native_page_count = len(source.pages)
        baseline_raw = extract.collect_pages(source)
    baseline_seconds = perf_counter()-tick
    tick = perf_counter()
    documents = PDFReader(return_full_document=False).load_data(pdf)
    reader_seconds = perf_counter()-tick
    assert len(documents) == native_page_count
    reader_raw = [{'page': i+1, 'text_raw': d.text, 'tables': [], 'has_image': None,
                   'reader_page_label': d.metadata.get('page_label')} for i, d in enumerate(documents)]
    (args.output/'pdfreader_raw_documents.jsonl').write_text(''.join(json.dumps({'physical_page': i+1,
        'metadata': d.metadata, 'text': d.text}, ensure_ascii=False)+'\n' for i, d in enumerate(documents)), encoding='utf8')
    metadata['pdfreader_label_sequence_matches_physical_pages'] = all(str(d.metadata.get('page_label')) == str(i+1) for i, d in enumerate(documents))
    get_model()  # Model cold load excluded from ingestion and retrieval timing.
    audits, retrievals, summaries = [], [], {}
    for name, raw, parse_seconds in [('pdfplumber', baseline_raw, baseline_seconds), ('pdfreader', reader_raw, reader_seconds)]:
        # Reuse exactly the same existing cleaning and downstream chunk policy.
        pages, removed = extract.strip_frame_noise(raw)
        pages = extract.annotate(pages)
        chunks = chunker.build_chunks(pages)
        (args.output/f'{name}_pages.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in pages), encoding='utf8')
        (args.output/f'{name}_chunks.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in chunks), encoding='utf8')
        full_text = {p['page']: extract.page_markdown(p) for p in pages}
        group = []
        for probe in probes['pages']:
            result = audit_page(full_text[probe['page']], probe['anchors'], probe.get('table_tuples', []))
            result.update(parser=name, page=probe['page'])
            group.append(result)
            audits.append(result)
        tick = perf_counter()
        col = chromadb.EphemeralClient().create_collection('pdf-parser-'+name, metadata={'hnsw:space': 'cosine'})
        add_chunks(col, chunks, embed_texts(c['text'] for c in chunks))
        build_seconds = perf_counter()-tick
        data = col.get(include=['documents', 'metadatas'])
        with patch.object(retriever, 'get_collection', return_value=col), \
             patch.object(retriever, '_corpus', (data['ids'], data['documents'], data['metadatas'])), \
             patch.object(retriever, '_bm25_cache', None):
            retriever.retrieve('病假', k=8, strategy='guarded')
            for case in probes['retrieval_cases']:
                tick = perf_counter()
                candidates = retriever.retrieve(case['question'], k=32, strategy='guarded')
                seconds = perf_counter()-tick
                hits = candidates[:8]
                context, mapping = prepare_context(hits, 'evidence')
                context_hits = [h for h in hits if h['id'] in mapping.values()]
                budget_context, budget_mapping = prepare_context(candidates, 'evidence')
                budget_hits = [h for h in candidates if h['id'] in budget_mapping.values()]
                retrievals.append({'parser': name, 'qid': case['qid'], 'question': case['question'],
                    'seconds': seconds, 'hits': hits, 'context_ids': list(mapping.values()),
                    'budget_ids': list(budget_mapping.values()), 'context_chars': len(context),
                    'budget_chars': len(budget_context),
                    'parse_available': evidence_coverage(case['references'], [{'text': t, 'pages': [p]} for p, t in full_text.items()]),
                    'at8': evidence_coverage(case['references'], hits),
                    'context8': evidence_coverage(case['references'], context_hits),
                    'budget6000': evidence_coverage(case['references'], budget_hits)})
        run = [r for r in retrievals if r['parser'] == name]
        summaries[name] = {'pages': len(pages), 'parse_seconds': parse_seconds, 'n_chunks': len(chunks),
            'structured_table_objects': sum(len(p['tables']) for p in pages), 'noise_lines_removed': removed,
            'probe_page_n': len(group), 'anchor_n': sum(r['anchor_n'] for r in group),
            'anchor_present_n': sum(r['present_n'] for r in group),
            'ordered_page_n': sum(r['order_complete'] for r in group),
            'order_pair_n': sum(r['order_pair_n'] for r in group),
            'correct_order_pair_n': sum(r['correct_order_pair_n'] for r in group),
            'table_tuple_n': sum(r['table_tuple_n'] for r in group),
            'table_tuple_total': sum(r['table_tuple_total'] for r in group),
            'build_seconds_warm': build_seconds, 'retrieval_n': len(run),
            'retrieval_seconds_p50': statistics.median(r['seconds'] for r in run),
            **{metric: statistics.mean(r[metric] for r in run) for metric in ['parse_available', 'at8', 'context8', 'budget6000']}}
        print(json.dumps({'parser': name, **summaries[name]}, ensure_ascii=False), flush=True)
    assert before == {str(p.relative_to(ROOT)): digest(p) for p in production_files}
    assert original_ids == sorted(get_collection(create=False).get()['ids'])
    metadata['production_files_and_index_ids_unchanged'] = True
    for filename, value in [('metadata.json', metadata), ('summary.json', summaries)]:
        (args.output/filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    for filename, value in [('page_audits.jsonl', audits), ('retrievals.jsonl', retrievals)]:
        (args.output/filename).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in value), encoding='utf8')


if __name__ == '__main__':
    main()
