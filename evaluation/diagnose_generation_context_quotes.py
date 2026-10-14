"""Exact frozen reference-quote presence; a diagnostic, not semantic Context Recall."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import run_generation_fusion_experiment as runner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    source=ROOT/'evaluation/guard_fusion_experiment_2026-10-13'
    refs_dir=ROOT/'evaluation/generation_fusion_references_v1'
    cases,chunks=runner.load_frozen(source)
    refs,hashes=runner.load_references(refs_dir,chunks)
    rows=[]
    for case in cases:
        row=runner.prepare_case(case,chunks)
        claims=[]
        for claim in refs[(case['dataset'],case['qid'])]['reference_claims']:
            matched=[any(quote in context['text'] and context['id'] in claim['evidence_ids']
                         for context in row['admitted_contexts']) for quote in claim['quotes']]
            claims.append(dict(text=claim['text'],evidence_ids=claim['evidence_ids'],
                quotes=claim['quotes'],quote_present=matched,all_quotes_present=all(matched)))
        rows.append(dict(dataset=case['dataset'],qid=case['qid'],variant=case['variant'],
            positive=refs[(case['dataset'],case['qid'])]['positive'],context_sha256=row['context_sha256'],
            admitted_ids=list(row['source_map'].values()),claims=claims))
    runner.save_json(args.output,dict(reference_sha256=hashes,source_sha256=runner.file_hash(__file__),
        scope='Exact quoted text in admitted reference chunks only; missing quote may have equivalent alternate evidence. Not semantic Context Recall or generated-answer correctness.',rows=rows))
    print(json.dumps(dict(contexts=len(rows),positive_contexts=sum(r['positive'] for r in rows),
        positive_contexts_with_missing_quotes=sum(r['positive'] and any(not c['all_quotes_present'] for c in r['claims']) for r in rows))))


if __name__=='__main__': main()
