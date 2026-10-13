"""Frozen retrieval ablation, optionally including full-order fusion D."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import unicodedata
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VARIANTS = {'A': 'none', 'B': 'global', 'C': 'guarded'}
FUSION_VARIANTS = {**VARIANTS, 'D': 'fusion'}
METRICS = ('ndcg_at3', 'mrr', 'recall_at8', 'full_at8', 'context_full')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text) if not c.isspace() and c != '|')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def map_references(cases, chunks):
    """Reconcile changed IDs from pre-existing quotes, never from retrieval hits."""
    mapped = copy.deepcopy(cases)
    ids = {c['id'] for c in chunks}
    audit = []
    for case in mapped:
        old = case.get('gold_chunk_ids', [])
        if set(old) <= ids:
            continue
        quote = normalize(case.get('evidence_quote', ''))
        pages = set(case.get('gold_pages', []))
        matches = [c['id'] for c in chunks if pages & set(c['pages']) and quote and quote in normalize(c['text'])]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely map reference {case['qid']}: {matches}")
        # Preserve any still-valid evidence IDs in multi-reference cases.
        new = list(dict.fromkeys([iid for iid in old if iid in ids] + matches))
        audit.append({'qid': case['qid'], 'old_ids': old, 'new_ids': new,
                      'gold_pages': sorted(pages), 'evidence_quote': case['evidence_quote'],
                      'method': 'unique normalized source quote within original source pages'})
        case['gold_chunk_ids'] = new
    return mapped, audit


def verify_index(chunks, data):
    indexed = dict(zip(data['ids'], zip(data['documents'], data['metadatas'])))
    if len(indexed) != len(chunks) or set(indexed) != {c['id'] for c in chunks}:
        raise ValueError('Index IDs/count do not match the frozen chunk artifact')
    for c in chunks:
        text, meta = indexed[c['id']]
        expected = {'pages': ','.join(str(p) for p in c['pages']), 'page_start': c['pages'][0],
                    'path': c['path'] or '(无章节)', 'clause': c['clause_range'], 'char_len': c['char_len']}
        if text != c['text'] or any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Index text/metadata drift: {c['id']}")


def retrieval_metrics(gold, ids, context_ids):
    if not gold:
        return {}
    relevant = [i for i, iid in enumerate(ids, 1) if iid in gold]
    dcg = sum(1. / math.log2(i+1) for i in relevant if i <= 3)
    ideal = sum(1. / math.log2(i+1) for i in range(1, min(3, len(gold))+1))
    return {'ndcg_at3': dcg/ideal, 'mrr': 1./relevant[0] if relevant else 0.,
            'recall_at8': len(gold & set(ids[:8]))/len(gold),
            'full_at8': float(gold <= set(ids[:8])), 'context_full': float(gold <= set(context_ids))}


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered)-1)*p
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower)


def summarize(rows, repeats, variants=None):
    variants = VARIANTS if variants is None else variants
    summary = {}
    for dataset in sorted({r['dataset'] for r in rows}):
        dataset_rows = [r for r in rows if r['dataset'] == dataset]
        paired = []
        unstable = []
        for qid in sorted({r['qid'] for r in dataset_rows}):
            question_rows = [r for r in dataset_rows if r['qid'] == qid]
            if len(question_rows) == len(variants)*repeats and all(r['status'] == 'complete' for r in question_rows):
                paired.append(qid)
                for variant in variants:
                    orders = {tuple(r['selected_ids']) for r in question_rows if r['variant'] == variant}
                    if len(orders) > 1:
                        unstable.append({'qid': qid, 'variant': variant})
        group = {'paired_complete_cases': len(paired), 'unstable_rankings': unstable, 'variants': {}}
        for variant in variants:
            all_rows = [r for r in dataset_rows if r['variant'] == variant]
            successful = [r for r in all_rows if r['status'] == 'complete']
            eligible = [r for r in successful if r['qid'] in paired and r['metrics']]
            # Repeated scores are averaged per question; n is never multiplied by timing repetitions.
            by_question = {qid: [r for r in eligible if r['qid'] == qid] for qid in paired}
            by_question = {qid: rs for qid, rs in by_question.items() if rs}
            metrics = {m: statistics.mean(statistics.mean(r['metrics'][m] for r in rs)
                                         for rs in by_question.values()) if by_question else None for m in METRICS}
            timings = {stage: {f'p{p}': percentile([r.get(stage, 0.) for r in successful], p/100) for p in (50, 95)}
                       for stage in ('recall_seconds', 'rerank_seconds', 'total_seconds', 'guard_seconds', 'fusion_seconds')}
            group['variants'][variant] = {'positive_cases': len(by_question), 'successful_runs': len(successful),
                'failed_runs': len(all_rows)-len(successful), 'metrics': metrics, 'timings': timings,
                'truncated_pairs': sum(sum(n > 512 for n in r['pair_tokens'].values()) for r in successful),
                'scored_pairs': sum(len(r['pair_tokens']) for r in successful)}
        baseline = group['variants']['A']['metrics']
        for variant in (v for v in variants if v != 'A'):
            current = group['variants'][variant]['metrics']
            group['variants'][variant]['delta_vs_A'] = {m: current[m]-baseline[m] if current[m] is not None and baseline[m] is not None else None for m in METRICS}
        summary[dataset] = group
    return summary


def report(summary, rows, manifest):
    comparisons = [v for v in manifest.get('variants', VARIANTS) if v != 'A']
    lines = ['# Rerank 检索层实验', '',
             '固定当前语料与候选池，比较 A 原 guarded、B 直接语义重排、C 保留分面与词法保护的语义重排。', '',
             '> 46/36 题均为模拟开发或回归数据，存在词面偏强、开发规则过拟合和题集重叠。没有回答生成、模型答案评分或用户裁决，不能据此声称回答正确率或真实用户准确率提升。', '',
             f"语料 {manifest['chunk_count']} 块；CPU FP32；线程 {manifest['cpu_threads']}；batch 8；pair 上限 512 tokens；候选最多 40；返回最多 8；上下文预算 6000 字符。",
             f"重排模型首次加载 {manifest['reranker_cold_load_seconds']:.3f}s；随后全程预热测量，每题每组 {manifest['repeats']} 次，按轮次与题目轮换执行顺序。", '',
             '评分读取原问题与片段全文；超过 512 tokens 的模型输入按 tokenizer 默认 longest-first 截断，返回片段和生成上下文保留全文。长度检查也计入重排耗时。MRR 截止最终返回的 8 块。', '',
             'P50/P95 基于各题全部成功运行，含正题和诊断题；质量指标按配对完成的正题逐题计算，再对重复结果求均值。重排耗时为长度检查与模型预测；总耗时还包括候选构造、排序、选择和诊断，不含上下文构造。', '']
    if 'D' in comparisons:
        lines += ['D 对共同候选池独立生成完整 Guard 顺序与完整 BGE 顺序，再按 0.5/(60+Guard名次) + 0.5/(60+BGE名次) 融合，最后取 Top-8。同分沿用 RRF 顺序。Guard 使用原有分面/词法规则遍历全部候选，没有先截断到8再补齐；权重是软偏好，融合结果不强制继承 Guard 名额。', '']
    for dataset, group in summary.items():
        lines += [f'## {dataset}', '', '| 组别 | 正题数 | nDCG@3 | MRR@8 | Recall@8 | 完整覆盖@8 | 预算后完整覆盖 | 总耗时 P50/P95(s) |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name, value in group['variants'].items():
            metrics = value['metrics']
            timing = value['timings']['total_seconds']
            formatted = ['—' if metrics[m] is None else f'{metrics[m]:.4f}' for m in METRICS]
            time_text = '—' if timing['p50'] is None else f"{timing['p50']:.3f}/{timing['p95']:.3f}"
            lines.append(f"| {name} | {value['positive_cases']} | " + ' | '.join(formatted) + f' | {time_text} |')
        lines += ['', '| 对照 | ΔnDCG@3 | ΔMRR@8 | ΔRecall@8(百分点) | Δ完整覆盖(百分点) | Δ预算后覆盖(百分点) |',
                  '|---|---:|---:|---:|---:|---:|']
        for name in comparisons:
            delta = group['variants'][name]['delta_vs_A']
            formatted = ['—' if delta[m] is None else f'{delta[m] * (100 if m in METRICS[2:] else 1):+.4f}' for m in METRICS]
            lines.append(f'| {name} − A | ' + ' | '.join(formatted) + ' |')
        lines += ['', '| 组别 | 召回 P50/P95(s) | 重排 P50/P95(s) | 成功/失败运行 | 截断 pair/评分 pair |', '|---|---:|---:|---:|---:|']
        for name, value in group['variants'].items():
            recall, rerank = (value['timings'][s] for s in ('recall_seconds', 'rerank_seconds'))
            time_text = lambda t: '—' if t['p50'] is None else f"{t['p50']:.3f}/{t['p95']:.3f}"
            lines.append(f"| {name} | {time_text(recall)} | {time_text(rerank)} | {value['successful_runs']}/{value['failed_runs']} | {value['truncated_pairs']}/{value['scored_pairs']} |")
        if 'D' in comparisons:
            extra = group['variants']['D']['timings']
            lines += ['', f"D 额外 Guard 完整排序 P50/P95：{time_text(extra['guard_seconds'])}s；融合 P50/P95：{time_text(extra['fusion_seconds'])}s（均包含在总检索耗时内）。"]
        lines += ['', f"配对完成 {group['paired_complete_cases']} 题；重复排名变化 {len(group['unstable_rankings'])} 项。无 gold 题只保存检索候选与来源诊断，不计算拒答率。", '']
        first = {r['qid']: r for r in rows if r['dataset'] == dataset and r['variant'] == 'A' and r['repeat'] == 0 and r['status'] == 'complete'}
        for name in comparisons:
            changes = []
            for r in rows:
                base = first.get(r['qid'])
                if r['dataset'] != dataset or r['variant'] != name or r['repeat'] != 0 or r['status'] != 'complete' or not r['metrics'] or not base:
                    continue
                delta = {m: r['metrics'][m]-base['metrics'][m] for m in METRICS}
                if any(abs(d) > 1e-9 for d in delta.values()):
                    changes.append((r, delta))
            lines += [f'### {name} 逐题变化', '']
            if not changes:
                lines += ['没有质量指标变化。', '']
            for r, delta in changes:
                lines.append(f"- {r['qid']}：{r['question']}；ΔnDCG@3 {delta['ndcg_at3']:+.4f}，ΔMRR {delta['mrr']:+.4f}，Δ完整覆盖 {delta['full_at8']:+.0f}，Δ预算后覆盖 {delta['context_full']:+.0f}。")
            lines.append('')
    recommendations = []
    for name in comparisons:
        groups = [g['variants'][name] for g in summary.values()]
        qualifies = all(v['failed_runs'] == 0 and v['delta_vs_A']['full_at8'] is not None and v['delta_vs_A']['full_at8'] >= 0 and v['delta_vs_A']['context_full'] >= 0 for v in groups)
        qualifies = qualifies and all(v['delta_vs_A']['ndcg_at3'] >= 0 and v['delta_vs_A']['mrr'] >= 0 for v in groups) and any(v['delta_vs_A']['ndcg_at3'] > 0 or v['delta_vs_A']['mrr'] > 0 for v in groups)
        recommendations.append(f"{name}：" + ('可进入下一轮回答对照实验，仍需检验语义支持、答案质量和交互耗时。' if qualifies else '暂不推荐进入回答实验；未同时满足排序改善和完整证据覆盖不下降。'))
    lines += ['## 结论', ''] + recommendations + ['', '本轮没有调参或更换模型。不能从一次开发集比较推断统计显著性或泛化收益；片段相关性评分也不保证政策适用性，条件与例外仍需逐题检查。', '', '运行命令、版本、哈希见 manifest.json；参考 ID 映射见 reference_mapping.json；全部原文见 chunks.jsonl；候选、分数、位置、选择原因与失败记录见 runs.jsonl。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data')
    parser.add_argument('--output', type=Path, default=ROOT/'evaluation/rerank_experiment_2026-10-12')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--include-fusion', action='store_true', help='Add D: equal-weight complete Guard/BGE rank fusion')
    parser.add_argument('--limit', type=int, help='Smoke test only: take the first N cases per dataset')
    args = parser.parse_args()
    variants = FUSION_VARIANTS if args.include_fusion else VARIANTS
    if args.repeats < 1 or (args.limit is not None and args.limit < 1):
        parser.error('repeats and limit must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be a new or empty directory; frozen runs are not overwritten')
    args.output.mkdir(parents=True, exist_ok=True)
    print('Loading retrieval runtime and checking actual index...', flush=True)
    from ragcore import config
    config.CHROMA_DIR = args.data_root/'chroma'
    from ragcore import retriever, reranker
    from ragcore.store import get_collection
    from ragcore.llm import prepare_context
    import torch
    import numpy as np
    threads = min(4, os.cpu_count() or 1)
    torch.set_num_threads(threads)
    chunk_path = args.data_root/'chunks.jsonl'
    chunks = read_jsonl(chunk_path)
    if len(chunks) != 123:
        raise ValueError('This frozen experiment expects the current 123-chunk corpus')
    index = get_collection(create=False).get(include=['documents', 'metadatas', 'embeddings'])
    verify_index(chunks, index)
    sort = sorted(range(len(index['ids'])), key=lambda i: index['ids'][i])
    embedding_hash = hashlib.sha256(np.asarray([index['embeddings'][i] for i in sort], dtype='<f4').tobytes()).hexdigest()
    paths = {'original46': args.data_root/'eval/golden.jsonl', 'development36': ROOT/'evaluation/baseline_v1_2026-09-26/cases.jsonl'}
    datasets, mapping = {}, {}
    for name, path in paths.items():
        cases, audit = map_references(read_jsonl(path), chunks)
        datasets[name] = cases[:args.limit] if args.limit else cases
        mapping[name] = audit
        save(args.output/f'{name}.json', datasets[name])
    save(args.output/'reference_mapping.json', mapping)
    (args.output/'chunks.jsonl').write_bytes(chunk_path.read_bytes())
    manifest = {'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {p: digest(ROOT/p) for p in ('ragcore/retriever.py', 'ragcore/evidence.py', 'ragcore/reranker.py', 'evaluation/run_rerank_experiment.py')},
        'chunks_sha256': digest(chunk_path), 'chunk_count': len(chunks), 'index_embeddings_sha256': embedding_hash,
        'dataset_sha256': {name: digest(path) for name, path in paths.items()}, 'model_id': reranker.MODEL_ID,
        'model_revision': reranker.MODEL_REVISION, 'device': 'cpu', 'dtype': 'float32', 'batch_size': 8,
        'max_pair_tokens': 512, 'truncation_policy': 'longest_first', 'cpu_threads': threads,
        'variants': variants,
        'fusion': {'guard_weight': .5, 'rank_constant': config.RRF_K,
                   'guard_order': 'select_evidence on complete common pool, k=pool size'} if args.include_fusion else None,
        'repeats': args.repeats, 'smoke_limit': args.limit, 'python': platform.python_version(),
        'versions': {p: version(p) for p in ('torch', 'sentence-transformers', 'transformers', 'chromadb', 'rank-bm25', 'jieba')},
        'command': subprocess.list2cmdline([sys.executable, *sys.argv]),
        'quality_note': 'Development/regression sets; no answer generation, judge or user adjudication. Labels never passed to retrieval.'}
    print('Index verified; loading pinned local reranker...', flush=True)
    tick = perf_counter()
    model = reranker.get_model()
    manifest['reranker_cold_load_seconds'] = perf_counter()-tick
    manifest['model_dtype'] = str(next(model.parameters()).dtype)
    if manifest['model_dtype'] != 'torch.float32':
        raise ValueError('Expected FP32 model')
    save(args.output/'manifest.json', manifest)
    first_question = next(iter(datasets.values()))[0]['question']
    for method in variants.values():
        retriever.retrieve(first_question, strategy='guarded', rerank=method)
    rows, expected_pools = [], {}
    run_path = args.output/'runs.jsonl'
    with run_path.open('w', encoding='utf8') as output:
        for dataset, cases in datasets.items():
            for repeat in range(args.repeats):
                for number, case in enumerate(cases):
                    names = list(variants)
                    offset = (repeat+number) % len(names)
                    names = names[offset:] + names[:offset]
                    for name in names:
                        trace = {}
                        row = {'dataset': dataset, 'qid': case['qid'], 'question': case['question'],
                               'kind': case.get('kind', case.get('type')), 'variant': name, 'repeat': repeat,
                               'execution_order': names, 'gold_ids': case.get('gold_chunk_ids', [])}
                        tick = perf_counter()
                        try:
                            # Runtime receives the question and switches only. No case object or gold labels.
                            hits = retriever.retrieve(case['question'], k=8, strategy='guarded', rerank=variants[name], guard_weight=.5, diagnostics=trace)
                            pool_key = (dataset, case['qid'])
                            expected = expected_pools.setdefault(pool_key, trace['candidate_ids'])
                            if expected != trace['candidate_ids']:
                                raise ValueError('Candidate pool changed across paired variants/repetitions')
                            context, source_map = prepare_context(hits, 'evidence')
                            context_ids = list(source_map.values())
                            gold = set(case.get('gold_chunk_ids', []))
                            row.update(trace, status='complete', context_ids=context_ids, context_chars=len(context),
                                metrics=retrieval_metrics(gold, trace['selected_ids'], context_ids),
                                candidate_gold_coverage=len(gold & set(trace['candidate_ids']))/len(gold) if gold else None,
                                ranked_candidates=sorted(trace['candidate_ids'], key=lambda iid: -trace['scores'][iid]) if trace['scores'] else trace['candidate_ids'])
                        except Exception as exc:
                            row.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                                       elapsed_to_failure=perf_counter()-tick)
                        rows.append(row)
                        output.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                        output.flush()
                    print(f'{dataset} repeat {repeat+1}/{args.repeats}: {number+1}/{len(cases)} {case["qid"]}', flush=True)
    summary = summarize(rows, args.repeats, variants=variants)
    save(args.output/'summary.json', summary)
    (args.output/'REPORT.md').write_text(report(summary, rows, manifest), encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if any(r['status'] != 'complete' for r in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
