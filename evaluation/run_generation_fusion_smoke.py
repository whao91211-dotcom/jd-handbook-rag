"""Diverse real-call contract smoke; these outputs are not benchmark results."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import run_generation_fusion_experiment as runner

CASES = {('original46', 'P17'), ('development36', 'C-P06'),
         ('development36', 'C-N01'), ('development36', 'C-A01')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('generate', 'judge', 'summarize'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=True)
    runner.llm.get_client.cache_clear()
    args.source_dir = ROOT/'evaluation/guard_fusion_experiment_2026-10-13'
    args.references_dir = ROOT/'evaluation/generation_fusion_references_v1'
    args.repeats, args.workers, args.limit = 1, 2, 4
    args.judge_model, args.judge_attempts = 'deepseek-v4-pro', 2
    args.supplemental = None
    cases, chunks = runner.load_frozen(args.source_dir)
    references, hashes = runner.load_references(args.references_dir, chunks)
    cases = [row for row in cases if (row['dataset'], row['qid']) in CASES]
    if set(runner._groups(cases)) != CASES:
        raise ValueError('Smoke cases missing from frozen evidence')
    manifest = runner.build_manifest(args, references, hashes, cases)
    manifest.update(smoke_only=True, smoke_script_sha256=runner.file_hash(__file__))
    manifest = runner.initialize_output(args.output, manifest, resume=args.stage != 'generate')
    runner.assert_frozen_inputs(manifest, args)
    {'generate': runner.run_generate, 'judge': runner.run_judge,
     'summarize': runner.run_summary}[args.stage](args, manifest, references, cases, chunks)


if __name__ == '__main__':
    main()
