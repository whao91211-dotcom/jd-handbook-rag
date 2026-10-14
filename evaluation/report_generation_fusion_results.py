"""Render audited measurements without changing scores or declaring automatic gains."""
import argparse
import json
from pathlib import Path


def number(value, digits=4):
    return 'NA' if value is None else f'{value:.{digits}f}'


def render(folder):
    read=lambda name:json.loads((folder/name).read_text(encoding='utf8'))
    summary, manifest, audit = read('summary.json'),read('manifest.json'),read('verification.json')
    if not audit['audit_passed']:
        raise ValueError('Only audited results can be rendered')
    scored=read('scored_answers.json')
    model_map=manifest['judge'].get('models_by_metric')
    judge_description=('；'.join(f'{kind}={model}' for kind,model in model_map.items()) if model_map
                       else manifest['judge']['model'])
    bias='同家族模型可能共享偏差。'
    if model_map:
        bias+='生成和正确性评审均使用Flash接口，相关错误风险仍需复核；请求别名不能证明服务商内部权重相同。'
    lines=['# Guard / Guard+BGE 生成对照实测','',
        '本轮是原始46题和模拟开发36题的开发/回归实验。参考由Codex依据冻结来源标注并交叉复核，自动评审不等于完整用户人工裁定，也不能据此宣称真实用户准确率。',
        '',f"生成测量代码版本：`{manifest['code_revision']}`。生成：`{manifest['generator']['model']}`；评审：{judge_description}。{bias}API标识和返回标识被记录，无法独立固定服务商内部权重快照。",
        '', 'A为原Guard；D为50/50 Guard/BGE秩融合。复用已冻结的检索证据，实际6000字符上下文与生成设置保持一致，不重新计召回/重排耗时。',
        '', '## 同期质量比较','',
        '表中分数均为0–1附近的评分，不是用户准确率；Δ为D−A。答案指标先平均同题成功配对重复，再等权平均题目；Context指标独立按正题每组一次评审。失败、无效评审与无事实回答不记满分。', '']
    for dataset,title in (('original46','原始46题（35道正题）'),('development36','模拟开发36题（9道正题）')):
        result=summary['datasets'][dataset]
        lines += [f'### {title}','', '| 指标 | A | D | 绝对变化 | 配对题数 | 有效重复对 | 评分NA对 |','|---|---:|---:|---:|---:|---:|---:|']
        for field,name in (('faithfulness','Faithfulness'),('citation_support_rate','引用支持率'),
            ('context_recall','Context Recall'),('context_precision','Context Precision（AP）'),
            ('factual_f1','Factual F1'),('answer_correctness','Answer Correctness（自定义）'),
            ('bertscore_f1','BERTScore F1'),('semantic_similarity','BGE余弦相似度')):
            context=field.startswith('context_')
            metric=(result['context_evaluation']['paired']['metrics'] if context else result['paired']['positive_metrics'])[field]
            digits=6 if field=='answer_correctness' else 4
            runs='—' if context else metric['A']['paired_run_count']
            na='—' if context else metric['A']['na_pair_count']
            lines.append(f"| {name} | {number(metric['A']['mean'], digits)} | {number(metric['D']['mean'], digits)} | {number(metric['delta_D_minus_A'], digits)} | {metric['A']['question_count']} | {runs} | {na} |")
        lines += ['', '| 组别 | 计划生成 | 完成 / 失败 / 未执行 | 自动评审等级：正确 / 部分 / 错误 | 全计划行为通过数 |',
            '|---|---:|---|---|---:|']
        for variant in ('A','D'):
            item=result['variants'][variant]
            grades=item['grades']
            lines.append(f"| {variant} | {item['planned_runs']} | {item['complete']} / {item['failed']} / {item['missing']} | {grades['correct']} / {grades['partial']} / {grades['incorrect']}（已评{item['grade_denominator']}） | {item['rubric']['passed']} / {item['planned_runs']} |")
        lines += ['', '正、负、含糊题的行为等级分开保存于summary的by_kind；上表全体行为通过比例不能直接命名为答案准确率。重复对/评分NA以双方成功生成为起点：原始35正题105个计划重复对中3对因生成失败排除，剩102对；开发9正题27对均成功。Context每题每组一次，不使用三重复分母。BERT/BGE与Correctness可能使用不同有效配对，不能直接从其总体均值反推组合分数。','']
    lines += ['## 测量完整性及成本','',
        f"计划{audit['planned_runs']}次，记录{audit['recorded_runs']}次，未执行{audit['missing_runs']}次；审计校验了源输入、标签隔离、原始记录、评分公式、NA/分母、用量及时间汇总。审计不替代语义人工裁定。", '',
        '| 阶段 | 全计划单元 | 完成 | 失败 | 未评审的合格单元 | 生成失败而不合格 | 记录的应用层请求数 | 输入token | 输出token |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for kind,item in summary['judge'].items():
        t=item['tokens']
        lines.append(f"| {kind} | {item['planned']} | {item['complete']} | {item['failed']} | {item['unreviewed_eligible']} | {item.get('ineligible_generation_runs', 0)} | {t['attempt_count']} | {t['prompt_tokens']} | {t['completion_tokens']} |")
    legacy=summary.get('legacy_pro_correctness')
    if legacy:
        t=legacy['tokens']
        lines += ['', f"已停止的旧Pro正确性分支另有{legacy['recorded']}份记录（{legacy['complete']}成功、{legacy['failed']}失败），记录请求{t['attempt_count']}次、输入{t['prompt_tokens']} token、输出{t['completion_tokens']} token。这些分数不参与Flash主成绩；以上用量是额外实验成本。"]
    lines += ['', '| 评审阶段（全部记录，含失败） | 模型 | 单元耗时P50 / P95秒 |',
              '|---|---|---|']
    for kind,item in summary['judge'].items():
        clock=item['timing']
        lines.append(f"| {kind} | {item.get('model', judge_description)} | {number(clock['p50'])} / {number(clock['p95'])} |")
    lines += ['', '| 数据 / 组 | 生成输入token | 生成输出token | 生成请求数 | 完成生成P50 / P95秒 |',
        '|---|---:|---:|---:|---|']
    for dataset,result in summary['datasets'].items():
        for variant,item in result['variants'].items():
            t,clock=item['tokens'],item['generation_timing_complete']
            lines.append(f"| {dataset} / {variant} | {t['prompt_tokens']} | {t['completion_tokens']} | {t['attempt_count']} | {number(clock['p50'])} / {number(clock['p95'])} |")
    lines += ['', '输出token包含服务返回的推理用量时，不再把reasoning_tokens重复相加。未知用量、失败尝试及全体耗时见summary；SDK内部HTTP重试未逐次可见。服务商实际计费未核验，因此只报告用量，不推算金额。冒烟和服务探针独立保留，不混入正式质量。', '',
        '本轮耗时只衡量生成/评审，不代表在线端到端延迟。CPU模型加载及语义评分时间见semantic_manifest。', '',
        '## 分数含义与限制','',
        '- Faithfulness仅检查实际送入的证据支持程度；无事实声明为NA。引用支持率分母是全部事实，未引用也计未支持。',
        '- Context Recall是冻结事实条目覆盖，一些条目包含多项条件；Context Precision按相关上下文顺序算AP。负题/含糊题不算这两项正题指标。',
        '- Answer Correctness为0.75×事实F1+0.25×BGE原始余弦，属于自定义组合，未运行官方Ragas。',
        '- BERTScore为官方bert-score 0.3.13、中文BERT第8层、idf=false、rescale=false，只有辅助意义；相似措辞可以包含错误数字或否定。',
        '- BERT和BGE均有原生512 token上限，逐条保留未截断token长度及上限，全文答案另存。', '']
    generation=(Path(__file__).resolve().parents[1]/manifest['source_directories']['generation']) if model_map else folder
    supplemental=generation/'supplemental_metrics.jsonl'
    if supplemental.exists():
        rows=[json.loads(line) for line in supplemental.read_text(encoding='utf8').splitlines() if line.strip()]
        lines += [f"成功正题语义评分共{len(rows)}条。"]
        for encoder in ('bert','bge'):
            key=encoder+'_tokens'
            a=sum(r[key]['answer']>r[key]['max_length'] for r in rows)
            b=sum(r[key]['reference']>r[key]['max_length'] for r in rows)
            lines.append(f'- {encoder}：回答超过上限{a}条，参考超过上限{b}条。')
        lines.append('')
    failures=[r for r in scored if r['status']!='complete']
    lines += ['## 失败记录','']
    for row in failures:
        lines.append(f"- {row['dataset']} / {row['qid']} / {row['variant']} / repeat={row['repeat']}：{row['error']}；耗时{row['generation_seconds']:.3f}秒。原始尝试及已返回的响应/推理均保留，未替换失败记录；未返回用量仍为未知。")
    if not failures: lines.append('无记录的生成失败。')
    lines += ['', '402等服务失败不能归因于排序质量；质量均值仅使用成功配对，失败同时计入全计划完成率。', '',
        '## 案例复核与推荐结论','']
    conclusion=folder/'CONCLUSION.md'
    lines += [conclusion.read_text(encoding='utf8') if conclusion.exists() else '案例复核与推荐结论尚未完成；此报告为中间结果。', '',
        '原始逐题数据由manifest的source_directories/source_file_sha256定位；各指标内部两组使用同一评审模型，Pro旧正确性结果独立保留，不混入Flash主分数。版本、评分与审计：manifest.json、summary.json、scored_answers.json、verification.json。']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    render(parser.parse_args().folder)
