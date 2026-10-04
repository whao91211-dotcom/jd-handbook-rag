"""Question-only shadow retrieval after table filtering; labels score returned hits only."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.run_pdf_parser_experiment import normalize, digest


def main():
    import chromadb
    from ragcore import retriever
    from ragcore.embedder import embed_texts
    from ragcore.store import add_chunks

    folder = ROOT / 'evaluation/pdf_table_filter_fix_2026-10-04'
    golden_path = ROOT / 'data/eval/golden.jsonl'
    gold = [json.loads(s) for s in golden_path.read_text(encoding='utf8').splitlines()]
    selection_path = ROOT / 'evaluation/ingestion_budget_2026-10-04/cases.jsonl'
    selected = {c['qid'] for c in map(json.loads, selection_path.read_text(encoding='utf8').splitlines())
                if c['dataset_group'] == 'retrieval46' and c['quote_span'] is not None}
    outputs, summary = [], {}
    for variant in ('pdfplumber', 'pdfplumber_fixed'):
        chunks = list(map(json.loads, (folder/f'{variant}_chunks.jsonl').read_text(encoding='utf8').splitlines()))
        col = chromadb.EphemeralClient().create_collection('table-regression-'+variant, metadata={'hnsw:space':'cosine'})
        add_chunks(col, chunks, embed_texts(c['text'] for c in chunks))
        data = col.get(include=['documents','metadatas'])
        rows = []
        with patch.object(retriever, 'get_collection', return_value=col), \
             patch.object(retriever, '_corpus', (data['ids'],data['documents'],data['metadatas'])), \
             patch.object(retriever, '_bm25_cache', None):
            for case in gold:
                if not case['gold_chunk_ids']:
                    continue
                hits = retriever.retrieve(case['question'], k=8, strategy='guarded')
                ids = {h['id'] for h in hits}
                page_text = normalize('\n'.join(h['text'] for h in hits if set(h['pages']) & set(case['gold_pages'])))
                row = {'variant':variant, 'qid':case['qid'], 'hit_ids':list(ids),
                       'unchanged_id_covered':set(case['gold_chunk_ids']) <= ids if case['qid'] != 'P32' else None,
                       'quote_covered':normalize(case['evidence_quote']) in page_text
                                       if case['qid'] in selected or case['qid'] == 'P32' else None}
                rows.append(row)
                outputs.append(row)
        id_rows = [r for r in rows if r['unchanged_id_covered'] is not None]
        quote_rows = [r for r in rows if r['qid'] in selected]
        summary[variant] = {'unchanged_gold_cases':len(id_rows),
                            'unchanged_gold_complete_at8':sum(r['unchanged_id_covered'] for r in id_rows),
                            'original_unique_quote_cases':len(quote_rows),
                            'original_unique_quotes_at8':sum(r['quote_covered'] for r in quote_rows),
                            'P32_quote_at8':next(r['quote_covered'] for r in rows if r['qid']=='P32')}
    result = {'golden_sha256':digest(golden_path), 'quote_selection_sha256':digest(selection_path),
              'script_sha256':digest(Path(__file__)), 'summary':summary,
              'note':'34 unchanged reference-ID cases; changed page-48 P32 scored with its original source quote separately. 26 quote cases use the pre-existing unique quote selection. Development regression, no answer generation.'}
    (folder/'gold_regression.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    (folder/'gold_regression_hits.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in outputs),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
