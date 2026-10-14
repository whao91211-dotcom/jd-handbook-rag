"""Independent Flash correctness review of frozen successful generation answers.

This creates new journals; prior Pro context, faithfulness and correctness data stay
unchanged. It never generates answers or reviews context/faithfulness.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from threading import Event
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from evaluation import run_generation_fusion_experiment as runner

MODEL = 'deepseek-flash'
SCHEMA_VERSION = 'correctness-flash-review-v1'
NEW_CODE_PATHS = ('evaluation/run_correctness_flash_review.py','tests/test_correctness_flash_review.py')
PARENT_FILES = ('manifest.json','answers.jsonl','context_reviews.jsonl','faithfulness_reviews.jsonl','correctness_reviews.jsonl')


def portable_directory(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_parent(args):
    parent = Path(args.parent)
    manifest_bytes = (parent/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('schema_version')!=runner.SCHEMA_VERSION:
        raise ValueError('Unexpected parent experiment schema')
    for filename,expected in manifest['code_sha256'].items():
        if runner.file_hash(ROOT/filename)!=expected:
            raise ValueError(f'Parent measured code hash mismatch: {filename}')
    runner.assert_frozen_inputs(manifest,args)
    files = {filename:runner.file_hash(parent/filename) for filename in PARENT_FILES if (parent/filename).is_file()}
    if files.get('manifest.json')!=hashlib.sha256(manifest_bytes.replace(b'\r\n',b'\n')).hexdigest():
        raise ValueError('Parent manifest changed while loading')
    cases,chunks = runner.load_frozen(args.source_dir)
    selected = {tuple(key) for key in manifest['selected_cases']}
    cases = [case for case in cases if (case['dataset'],case['qid']) in selected]
    if set(runner._groups(cases))!=selected:
        raise ValueError('Parent selected cases differ from frozen source')
    references,hashes = runner.load_references(args.references_dir,chunks)
    if hashes!=manifest['reference_sha256']:
        raise ValueError('Parent frozen reference hashes differ')
    for case in cases:
        ref = references[(case['dataset'],case['qid'])]
        if ref['question']!=case['question'] or ref['kind']!=case['kind']:
            raise ValueError('Parent reference question/kind mismatch')
    answers = runner.read_keyed(parent/'answers.jsonl')
    runner.validate_existing_answers(answers,cases,chunks,manifest['repeats'],manifest['generator']['model'])
    if runner.file_hash(parent/'answers.jsonl')!=files.get('answers.jsonl'):
        raise ValueError('Parent answer journal changed while loading')
    return dict(parent_manifest=manifest,parent_manifest_bytes=manifest_bytes,parent_files_sha256=files,
                answers=answers,rows=[row for row in answers.values() if row['status']=='complete'],
                references=references,cases=cases,chunks=chunks)


def build_manifest(args,bundle):
    parent = bundle['parent_manifest']
    revision = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    inherited = {key:parent[key] for key in ('generator','source_sha256','reference_sha256','repeats',
                'selected_cases','case_counts','dataset_sha256','hash_normalization') if key in parent}
    code = dict(parent['code_sha256'])
    code.update({path:runner.file_hash(ROOT/path) for path in NEW_CODE_PATHS})
    return dict(**inherited,schema_version=SCHEMA_VERSION,code_revision=revision,code_sha256=code,
        parent_directory=portable_directory(args.parent),source_directory=portable_directory(args.source_dir),
        references_directory=portable_directory(args.references_dir),
        parent_manifest=parent,parent_manifest_sha256=bundle['parent_files_sha256']['manifest.json'],
        parent_answers_sha256=bundle['parent_files_sha256']['answers.jsonl'],
        parent_files_sha256=bundle['parent_files_sha256'],
        service_destination_sha256=parent['service_destination_sha256'],workers=args.workers,
        planned_reviews=len(bundle['rows']),parent_answer_records=len(bundle['answers']),
        parent_generation_failed=sum(row['status']=='failed' for row in bundle['answers'].values()),
        judge=dict(model=MODEL,version=runner.JUDGE_VERSION,attempts=args.max_attempts,temperature=0,
            max_tokens=6000,thinking='disabled',
            prompt_sha256=runner.value_hash(runner.JUDGE_INSTRUCTIONS['correctness'])),
        metric_scope='Correctness only; every successful answer, including negative/ambiguous cases. Old Pro correctness is reference/cost data only.',
        quality_note='Generation requested deepseek-v4-flash; correctness requests catalog model deepseek-flash. Endpoint aliases do not establish identical weights or versions. Flash-family generation and judging may have more correlated errors. Source-grounded references are not human gold adjudication.',
        audit_note='Raw application-level attempts/usage retained; SDK internal transport retries are not individually visible. No credentials recorded.')


def assert_identity(args,manifest):
    parent = Path(args.parent)
    for filename,expected in manifest['parent_files_sha256'].items():
        if runner.file_hash(parent/filename)!=expected:
            raise ValueError(f'Parent {filename} hash changed')
    for dataset,expected in manifest['reference_sha256'].items():
        if runner.file_hash(Path(args.references_dir)/f'{dataset}.json')!=expected:
            raise ValueError('Parent reference hash changed')
    for filename,expected in manifest['code_sha256'].items():
        if runner.file_hash(ROOT/filename)!=expected:
            raise ValueError(f'Review code hash changed: {filename}')
    for filename,expected in manifest['source_sha256'].items():
        if runner.file_hash(Path(args.source_dir)/filename)!=expected:
            raise ValueError(f'Frozen source hash changed: {filename}')


def preserve_parent_manifest(output,bundle,manifest):
    snapshot = Path(output)/'parent_manifest.json'
    if snapshot.exists():
        if runner.file_hash(snapshot)!=manifest['parent_manifest_sha256']:
            raise ValueError('Preserved parent manifest differs from frozen parent')
    else:
        snapshot.write_bytes(bundle['parent_manifest_bytes'])


def review_summary(existing,manifest):
    records = list(existing.values())
    complete = sum(row['status']=='complete' for row in records)
    planned = manifest['planned_reviews']
    return dict(schema_version=SCHEMA_VERSION,model=MODEL,planned=planned,recorded=len(records),
        complete=complete,failed=sum(row['status']=='failed' for row in records),missing=planned-len(records),
        completion_rate=complete/planned if planned else None,
        tokens=runner.metrics.token_summary([attempt for row in records for attempt in row['attempts']]),
        timing=runner.metrics.timing_summary([row['judge_seconds'] for row in records]),
        note=manifest['metric_scope'],quality_note=manifest['quality_note'])


def run_reviews(args,manifest,bundle):
    output = Path(args.output)
    if output.resolve()==Path(args.parent).resolve() or Path(args.parent).resolve() in output.resolve().parents:
        raise ValueError('Flash output must be independent of the parent experiment directory')
    assert_identity(args,manifest)
    preserve_parent_manifest(output,bundle,manifest)
    journal = runner.Journal(output/'correctness_reviews.jsonl')
    existing = runner.read_keyed(journal.path)
    validation_args = SimpleNamespace(judge_model=MODEL)
    runner.validate_existing_reviews(existing,'correctness',bundle['rows'],bundle['references'],validation_args)
    scheduled = [row for row in bundle['rows'] if runner.metrics.row_key(row) not in existing]
    stop = Event()
    def work(row):
        if stop.is_set():
            return 'skipped_after_fatal_error'
        assert_identity(args,manifest)
        review = runner.judge_once('correctness',row,bundle['references'][(row['dataset'],row['qid'])],
                                   model=MODEL,max_attempts=args.max_attempts)
        if any((attempt.get('error') or {}).get('status_code') in runner.FATAL_API_STATUSES for attempt in review['attempts']):
            stop.set()
        assert_identity(args,manifest)
        runner.validate_existing_reviews({runner.metrics.row_key(review):review},'correctness',
                                         bundle['rows'],bundle['references'],validation_args)
        journal.append(review)
        return review['status']
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work,row) for row in scheduled]
        for index,future in enumerate(as_completed(futures),1):
            print(f'Flash correctness {index}/{len(scheduled)}: {future.result()}',flush=True)
    if stop.is_set():
        print('Stopped queued correctness requests after a fatal API response; missing reviews remain unrecorded.',flush=True)
    assert_identity(args,manifest)
    records = runner.read_keyed(journal.path)
    runner.validate_existing_reviews(records,'correctness',bundle['rows'],bundle['references'],validation_args)
    summary = review_summary(records,manifest)
    summary['stopped_after_fatal_error'] = stop.is_set()
    runner.save_json(output/'review_status.json',summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=ROOT/'evaluation/generation_fusion_experiment_2026-10-13')
    parser.add_argument('--output',type=Path,default=ROOT/'evaluation/generation_fusion_correctness_flash_2026-10-14')
    parser.add_argument('--references-dir',type=Path,default=ROOT/'evaluation/generation_fusion_references_v1')
    parser.add_argument('--source-dir',type=Path,default=ROOT/'evaluation/guard_fusion_experiment_2026-10-13')
    parser.add_argument('--env-file',type=Path)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--max-attempts',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args(argv)
    if not 1<=args.workers<=3 or not 1<=args.max_attempts<=2:
        parser.error('workers must be1..3 and max-attempts must be1..2')
    if args.output.resolve()==args.parent.resolve() or args.parent.resolve() in args.output.resolve().parents:
        parser.error('output must be independent of the parent experiment directory')
    if args.env_file:
        if not args.env_file.is_file(): parser.error('env-file does not exist')
        from dotenv import load_dotenv
        load_dotenv(args.env_file,override=True)
        runner.llm.get_client.cache_clear()
    bundle = load_parent(args)
    manifest = build_manifest(args,bundle)
    if not runner.llm.api_ready():
        parser.error('Configured API unavailable; restore service before running this review')
    configured = hashlib.sha256(runner.llm.env_config()['base_url'].encode('utf8')).hexdigest()
    if configured!=manifest['service_destination_sha256']:
        parser.error('Configured service destination differs from the frozen parent')
    manifest = runner.initialize_output(args.output,manifest,resume=args.resume)
    run_reviews(args,manifest,bundle)


if __name__=='__main__': main()
