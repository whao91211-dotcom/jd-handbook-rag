"""Compose frozen Pro context/faithfulness with separately reviewed Flash correctness.

Offline only. Original journals and frozen scoring rules are never modified.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import run_generation_fusion_experiment as runner
from evaluation import generation_metrics as metrics

VERSION = 'mixed-generation-summary-v1'
PRO = 'deepseek-v4-pro'
FLASH = 'deepseek-flash'
KINDS = ('context', 'faithfulness', 'correctness')
INHERITED = ('generator', 'source_sha256', 'reference_sha256', 'dataset_sha256',
             'repeats', 'selected_cases', 'case_counts', 'hash_normalization')


def display_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def validate_manifests(parent, flash, parent_dir):
    if parent['judge']['model'] != PRO or flash['judge']['model'] != FLASH:
        raise ValueError('Expected Pro parent and Flash correctness judge models')
    if flash.get('parent_manifest_sha256') != runner.file_hash(parent_dir/'manifest.json'):
        raise ValueError('Flash parent manifest identity mismatch')
    if flash.get('parent_answers_sha256') != runner.file_hash(parent_dir/'answers.jsonl'):
        raise ValueError('Flash parent answer journal identity mismatch')
    if flash.get('parent_manifest') != parent:
        raise ValueError('Flash parent manifest snapshot differs')
    for field in INHERITED:
        if flash.get(field) != parent[field]:
            raise ValueError(f'Flash inherited input differs: {field}')
    for name, expected in parent['code_sha256'].items():
        if flash['code_sha256'].get(name) != expected or runner.file_hash(ROOT/name) != expected:
            raise ValueError(f'Frozen evaluation code differs: {name}')
    for name, expected in flash['code_sha256'].items():
        if runner.file_hash(ROOT/name) != expected:
            raise ValueError(f'Flash runner code differs: {name}')
    for field in ('version','temperature','max_tokens','thinking'):
        if flash['judge'].get(field) != parent['judge'].get(field):
            raise ValueError(f'Judge contract differs apart from model: {field}')
    if flash['judge'].get('prompt_sha256') != parent['judge']['prompt_sha256']['correctness']:
        raise ValueError('Flash correctness instruction hash differs from frozen parent prompt')
    parent_files = flash.get('parent_files_sha256')
    if not isinstance(parent_files, dict):
        raise ValueError('Missing Flash parent file fingerprint map')
    for name in ('manifest.json','answers.jsonl','context_reviews.jsonl','faithfulness_reviews.jsonl'):
        if parent_files.get(name) != runner.file_hash(parent_dir/name):
            raise ValueError(f'Flash parent source identity mismatch: {name}')
    for name, expected in parent_files.items():
        if runner.file_hash(parent_dir/name) != expected:
            raise ValueError(f'Flash parent source changed: {name}')


def validate_reviews(records, kind, rows, references, model):
    runner.validate_existing_reviews(records, kind, rows, references, SimpleNamespace(judge_model=model))
    for record in records.values():
        if record.get('kind') != kind:
            raise ValueError('Judge kind differs from its source journal')
        for attempt in record['attempts']:
            actual_model = attempt.get('response_model')
            if actual_model is not None and actual_model != model:
                raise ValueError('Judge response model differs from selected metric model')


def judge_summary(records, *, model, planned, eligible, source):
    completed = sum(r['status']=='complete' for r in records.values())
    return dict(model=model, source=source, planned=planned, eligible=eligible,
        recorded=len(records), complete=completed,
        failed=sum(r['status']!='complete' for r in records.values()),
        unreviewed_eligible=eligible-len(records), ineligible_generation_runs=planned-eligible,
        completion_rate_all_planned=completed/planned if planned else None,
        completion_rate_eligible=completed/eligible if eligible else None,
        tokens=metrics.token_summary([a for r in records.values() for a in r['attempts']]),
        timing=metrics.timing_summary([r['judge_seconds'] for r in records.values()]))


def compose(parent_dir, flash_dir, output, *, source_dir=None, references_dir=None):
    composition_hash = runner.file_hash(Path(__file__))
    parent_dir, flash_dir, output = (Path(p).resolve() for p in (parent_dir, flash_dir, output))
    source_dir = Path(source_dir or ROOT/'evaluation/guard_fusion_experiment_2026-10-13').resolve()
    references_dir = Path(references_dir or ROOT/'evaluation/generation_fusion_references_v1').resolve()
    if output in (parent_dir, flash_dir) or (output.exists() and any(output.iterdir())):
        raise ValueError('Combined output must be a separate empty directory')
    source_files = [parent_dir/name for name in
        ('manifest.json','answers.jsonl','context_reviews.jsonl','faithfulness_reviews.jsonl',
         'supplemental_metrics.jsonl','semantic_manifest.json')]
    source_files += [flash_dir/name for name in ('manifest.json','parent_manifest.json')]
    for path in (parent_dir/'correctness_reviews.jsonl', flash_dir/'correctness_reviews.jsonl',
                 flash_dir/'review_status.json'):
        if path.exists():
            source_files.append(path)
    # Freeze the input set before any validation or aggregation, then check again before writing.
    snapshots = {display_path(p): runner.file_hash(p) for p in source_files}
    parent = json.loads((parent_dir/'manifest.json').read_text(encoding='utf8'))
    flash = json.loads((flash_dir/'manifest.json').read_text(encoding='utf8'))
    validate_manifests(parent, flash, parent_dir)
    if json.loads((flash_dir/'parent_manifest.json').read_text(encoding='utf8')) != parent:
        raise ValueError('Flash copied parent manifest differs from generation source')
    runner.assert_frozen_inputs(parent, SimpleNamespace(source_dir=source_dir, references_dir=references_dir))
    all_cases, chunks = runner.load_frozen(source_dir)
    references, ref_hashes = runner.load_references(references_dir, chunks)
    if ref_hashes != parent['reference_sha256']:
        raise ValueError('Loaded references differ from frozen generation references')
    selected = [tuple(x) for x in parent['selected_cases']]
    if len(selected) != len(set(selected)):
        raise ValueError('Duplicate selected case')
    cases = [c for c in all_cases if (c['dataset'],c['qid']) in set(selected)]
    available = {(c['dataset'],c['qid']) for c in cases}
    if available != set(selected):
        raise ValueError('Selected cases not found in frozen retrieval source')
    if {ds:sum(key[0]==ds for key in available) for ds in runner.DATASETS} != parent['case_counts']:
        raise ValueError('Selected case counts differ from frozen manifest')
    repeats = parent['repeats']
    if type(repeats) is not int or repeats < 1:
        raise ValueError('Invalid frozen repeat count')
    answers = runner.read_keyed(parent_dir/'answers.jsonl')
    runner.validate_existing_answers(answers, cases, chunks, repeats, parent['generator']['model'])
    complete = [r for r in answers.values() if r['status']=='complete']
    context_rows = [runner.prepare_case(c,chunks) for c in cases]
    reviews = {kind:runner.read_keyed((flash_dir if kind=='correctness' else parent_dir)/f'{kind}_reviews.jsonl',
                                     include_repeat=kind!='context') for kind in KINDS}
    models = dict(context=PRO, faithfulness=PRO, correctness=FLASH)
    for kind, records in reviews.items():
        validate_reviews(records, kind, context_rows if kind=='context' else complete, references, models[kind])
    legacy = runner.read_keyed(parent_dir/'correctness_reviews.jsonl')
    validate_reviews(legacy,'correctness',complete,references,PRO)
    supplements, semantic_metadata = runner.load_supplemental(parent_dir/'supplemental_metrics.jsonl',
        answers,references,parent,parent_dir/'answers.jsonl')
    rows = []
    for key, original in answers.items():
        row = dict(original)
        ref = references[key[:2]]
        row['positive'] = ref['positive']
        for kind in KINDS:
            review = reviews[kind].get(key[:3] if kind=='context' else key)
            if review and review['status']=='complete':
                row.update(review['scores'])
        supplement = supplements.get(key)
        if supplement:
            row['semantic_similarity'] = supplement['semantic_similarity']
            row['bertscore_f1'] = supplement['bertscore']['f1']
        row['answer_correctness'] = metrics.answer_correctness(row.get('factual_f1'),
            row.get('semantic_similarity'),positive=ref['positive'])
        rows.append(row)
    judge = {kind:judge_summary(records,model=models[kind],
        planned=len(cases)*(1 if kind=='context' else repeats),
        eligible=len(cases) if kind=='context' else len(complete),
        source=display_path((flash_dir if kind=='correctness' else parent_dir)/f'{kind}_reviews.jsonl'))
        for kind,records in reviews.items()}
    legacy_summary = judge_summary(legacy,model=PRO,planned=len(cases)*repeats,eligible=len(complete),
                                  source=display_path(parent_dir/'correctness_reviews.jsonl'))
    legacy_summary.update(included_in_main_scores=False,
                          scope='Discarded Pro correctness branch: usage/reference only; no grades or quality scores merged.')
    context_scored = []
    for review in reviews['context'].values():
        row = {name:review[name] for name in ('dataset','qid','variant','status')}
        row['positive'] = references[(row['dataset'],row['qid'])]['positive']
        if row['status']=='complete':
            row.update(review['scores'])
        context_scored.append(row)
    datasets = {}
    for ds in runner.DATASETS:
        datasets[ds] = metrics.summarize_dataset([r for r in rows if r['dataset']==ds],
            expected_cases=parent['case_counts'][ds],repeats=repeats)
        datasets[ds]['context_evaluation'] = metrics.summarize_contexts(
            [r for r in context_scored if r['dataset']==ds],expected_cases=parent['case_counts'][ds],
            expected_positive_cases=sum(key[0]==ds and references[key]['positive'] for key in available))
    manifest = copy.deepcopy(parent)
    manifest.update(composition_version=VERSION, parent_manifest_sha256=runner.file_hash(parent_dir/'manifest.json'),
        source_directories=dict(generation=display_path(parent_dir),correctness=display_path(flash_dir)),
        source_file_sha256=snapshots,
        composition_code_sha256={display_path(Path(__file__)):composition_hash})
    manifest['judge'] = {k:v for k,v in parent['judge'].items() if k!='model'}
    manifest['judge'].update(models_by_metric=models,
        settings_by_metric={kind:copy.deepcopy((flash if kind=='correctness' else parent)['judge']) for kind in KINDS},
        bias_note='Flash-family generation and Flash correctness may have higher correlated-error risk. Requested generator deepseek-v4-flash and judge deepseek-flash interface names do not prove identical weights or versions. Pro context/faithfulness share the DeepSeek family. These are model annotations, not human accuracy.')
    for name, expected in snapshots.items():
        if runner.file_hash(Path(name) if Path(name).is_absolute() else ROOT/name) != expected:
            raise ValueError('Source input changed during mixed composition')
    for kind, directory in (('generation',parent_dir),('correctness',flash_dir)):
        expected_presence = display_path(directory/'correctness_reviews.jsonl') in snapshots
        if (directory/'correctness_reviews.jsonl').exists() != expected_presence:
            raise ValueError('Review journal appeared or disappeared during composition')
    runner.assert_frozen_inputs(parent,SimpleNamespace(source_dir=source_dir,references_dir=references_dir))
    validate_manifests(parent,flash,parent_dir)
    if runner.file_hash(Path(__file__)) != composition_hash:
        raise ValueError('Composition code changed during aggregation')
    output.mkdir(parents=True,exist_ok=True)
    runner.save_json(output/'manifest.json',manifest)
    result = dict(schema_version=runner.SCHEMA_VERSION,custom_metrics=True,
        manifest_sha256=runner.file_hash(output/'manifest.json'),datasets=datasets,judge=judge,
        legacy_pro_correctness=legacy_summary,supplemental_records=len(supplements),semantic_manifest=semantic_metadata,
        notes=['Offline mixed-judge composition; original journals unchanged.',
            'Pro context and faithfulness; Flash correctness for every eligible answer. No old Pro correctness fallback.',
            'Custom AnswerCorrectness=.75*factualF1+.25*raw local BGE cosine; BERTScore is separate; no official Ragas score.',
            'Question-weighted paired complete repetitions; missing or failed Flash reviews remain unscored.',
            'Context metrics are independent of generation and correctness review completion.',
            'Old Pro correctness all-attempt usage is reported separately, never added to the Flash metric branch.',
            'Generation timings exclude frozen retrieval; offline composition timing is not model latency.',
            manifest['judge']['bias_note'],parent['quality_note']])
    runner.save_json(output/'scored_answers.json',rows)
    runner.save_json(output/'summary.json',result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-dir',type=Path,default=ROOT/'evaluation/generation_fusion_experiment_2026-10-13')
    parser.add_argument('--flash-dir',type=Path,default=ROOT/'evaluation/generation_fusion_correctness_flash_2026-10-14')
    parser.add_argument('--output',type=Path,default=ROOT/'evaluation/generation_fusion_combined_2026-10-14')
    parser.add_argument('--source-dir',type=Path,default=ROOT/'evaluation/guard_fusion_experiment_2026-10-13')
    parser.add_argument('--references-dir',type=Path,default=ROOT/'evaluation/generation_fusion_references_v1')
    args = parser.parse_args(argv)
    result = compose(args.parent_dir,args.flash_dir,args.output,source_dir=args.source_dir,references_dir=args.references_dir)
    print(json.dumps({'output':display_path(args.output),'correctness_model':result['judge']['correctness']['model'],
                      'correctness_complete':result['judge']['correctness']['complete']},ensure_ascii=False))


if __name__=='__main__': main()
