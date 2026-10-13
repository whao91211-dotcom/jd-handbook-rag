"""Pure, auditable custom generation metrics; these are not official Ragas scores."""
from __future__ import annotations

from collections import defaultdict
import math
import statistics


POSITIVE_METRICS = ('faithfulness', 'citation_support_rate', 'factual_f1',
                    'semantic_similarity', 'answer_correctness', 'bertscore_f1')


def _strings(value, field):
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError(f'{field} must be a list of nonempty strings')
    if len(value) != len(set(value)):
        raise ValueError(f'{field} contains duplicates')
    return value


def _bools(value, length, field):
    if not isinstance(value, list) or len(value) != length or any(type(x) is not bool for x in value):
        raise ValueError(f'{field} must contain exactly {length} booleans')
    return value


def _reason(review):
    if not isinstance(review, dict) or not isinstance(review.get('reason'), str) or not review['reason'].strip():
        raise ValueError('Review must include a nonempty reason')


def faithfulness_scores(review, labels):
    if not isinstance(review, dict) or not isinstance(review.get('claims'), list):
        raise ValueError('claims must be a list')
    allowed = set(labels)
    supported = cited = 0
    seen = set()
    for claim in review['claims']:
        if not isinstance(claim, dict) or not isinstance(claim.get('claim'), str) or not claim['claim'].strip():
            raise ValueError('Each claim must contain factual claim text')
        if claim['claim'] in seen:
            raise ValueError('Duplicate atomic claims')
        seen.add(claim['claim'])
        _reason(claim)
        support = claim.get('support')
        if support not in ('supported', 'unsupported', 'contradicted'):
            raise ValueError('Invalid claim support')
        evidence = _strings(claim.get('evidence_labels'), 'evidence_labels')
        citations = _strings(claim.get('cited_labels'), 'cited_labels')
        if not set(evidence + citations) <= allowed:
            raise ValueError('Unknown evidence or citation source label')
        if type(claim.get('citation_supported')) is not bool:
            raise ValueError('citation_supported must be boolean')
        if support in ('supported', 'contradicted') and not evidence:
            raise ValueError('Supported/contradicted claims require evidence')
        if claim['citation_supported'] and (support != 'supported' or not citations or not set(citations) <= set(evidence)):
            raise ValueError('Supported citation must cite supporting evidence')
        supported += support == 'supported'
        cited += support == 'supported' and bool(citations) and claim['citation_supported']
    n = len(review['claims'])
    return dict(faithfulness=supported/n if n else None,
                citation_support_rate=cited/n if n else None, claim_count=n)


def correctness_scores(review, reference):
    _reason(review)
    if review.get('grade') not in ('correct', 'partial', 'incorrect'):
        raise ValueError('Invalid correctness grade')
    groups = [_strings(review.get(k), k) for k in ('TP', 'FP', 'FN')]
    if sum(map(len, groups)) != len(set().union(*map(set, groups))):
        raise ValueError('TP/FP/FN factual lists overlap')
    checks = {key: _bools(review.get(key), len(reference[ref_key]), key)
              for key, ref_key in [('required_checks', 'required_points'),
                                   ('clarification_checks', 'required_clarifications'),
                                   ('forbidden_checks', 'forbidden_inferences')]}
    if review['grade']=='correct' and (groups[1] or groups[2] or not all(checks['required_checks']) or
                                     not all(checks['clarification_checks']) or any(checks['forbidden_checks'])):
        raise ValueError('Correct grade contradicts erroneous/missing facts or required behavior checks')
    tp, fp, fn = map(len, groups)
    denominator = tp + .5 * (fp + fn)
    return dict(grade=review['grade'], factual_f1=tp/denominator if denominator else None,
                TP_count=tp, FP_count=fp, FN_count=fn,
                required_pass=all(checks['required_checks']),
                clarification_pass=all(checks['clarification_checks']),
                forbidden_pass=not any(checks['forbidden_checks']),
                **checks)


def context_scores(review, reference, context_count):
    if not isinstance(review, dict):
        raise ValueError('Context review must be an object')
    supported = _bools(review.get('claim_supported'), len(reference['reference_claims']), 'claim_supported')
    relevant = _bools(review.get('context_relevant'), context_count, 'context_relevant')
    reasons = review.get('reasons')
    if not isinstance(reasons, list) or any(not isinstance(x, str) for x in reasons):
        raise ValueError('reasons must be a string list')
    if not reference['positive']:
        return dict(context_recall=None, context_precision=None)
    found = 0
    precision_sum = 0.
    for rank, hit in enumerate(relevant, 1):
        if hit:
            found += 1
            precision_sum += found / rank
    return dict(context_recall=sum(supported)/len(supported) if supported else None,
                context_precision=precision_sum/found if found else 0.)


def answer_correctness(factual_f1, semantic_similarity, *, positive):
    if not positive or factual_f1 is None or semantic_similarity is None:
        return None
    for value in (factual_f1, semantic_similarity):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError('Non-finite correctness inputs')
    # Raw cosine is retained; do not silently rescale or clip the supplemental score.
    return .75 * factual_f1 + .25 * semantic_similarity


def row_key(row, include_repeat=True):
    fields = ('dataset', 'qid', 'variant', 'repeat') if include_repeat else ('dataset', 'qid', 'variant')
    return tuple(row[field] for field in fields)


def _average(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def _percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered)-1)*p
    low = int(index)
    return ordered[low] + (ordered[min(low+1, len(ordered)-1)]-ordered[low])*(index-low)


def timing_summary(values):
    return dict(count=len(values), mean=_average(values), p50=_percentile(values, .5),
                p95=_percentile(values, .95), total=sum(values))


def token_summary(attempts):
    result = dict(attempt_count=len(attempts))
    for field in ('prompt_tokens', 'completion_tokens', 'reasoning_tokens'):
        known = [x[field] for x in attempts if isinstance(x.get(field), int)]
        result[field] = sum(known)
        result[field+'_unknown_attempts'] = len(attempts)-len(known)
    return result


def _rubric_pass(row):
    return row.get('grade') == 'correct' and all(row.get(k) is True for k in
                                               ('required_pass', 'clarification_pass', 'forbidden_pass'))


def summarize_dataset(rows, *, expected_cases, repeats):
    """Question-weighted paired positive quality; all planned runs remain in coverage denominators."""
    keyed = {}
    for row in rows:
        key = row_key(row)
        if key in keyed:
            raise ValueError('Duplicate answer key in summary')
        keyed[key] = row
    variants = {}
    planned = expected_cases * repeats
    for variant in ('A', 'D'):
        group = [r for r in rows if r['variant'] == variant]
        completed = [r for r in group if r['status'] == 'complete']
        reviewed = [r for r in completed if r.get('grade') in ('correct', 'partial', 'incorrect')]
        passed = sum(_rubric_pass(r) for r in reviewed)
        by_kind = {}
        for kind in sorted({r.get('kind', 'unknown') for r in group}):
            subset = [r for r in group if r.get('kind', 'unknown') == kind]
            subset_reviewed = [r for r in subset if r['status'] == 'complete' and r.get('grade')]
            by_kind[kind] = dict(recorded_runs=len(subset), completed=sum(r['status']=='complete' for r in subset),
                                 reviewed=len(subset_reviewed),
                                 grades={grade: sum(r.get('grade')==grade for r in subset_reviewed)
                                         for grade in ('correct', 'partial', 'incorrect')})
        variants[variant] = dict(planned_runs=planned, recorded=len(group), complete=len(completed),
            failed=sum(r['status']!='complete' for r in group), missing=max(0, planned-len(group)),
            completion_rate=len(completed)/planned if planned else None,
            tokens=token_summary([a for r in group for a in r.get('usage_attempts', [])]),
            generation_timing_all=timing_summary([r['generation_seconds'] for r in group if r.get('generation_seconds') is not None]),
            generation_timing_complete=timing_summary([r['generation_seconds'] for r in completed if r.get('generation_seconds') is not None]),
            grades={g: sum(r.get('grade')==g for r in reviewed) for g in ('correct','partial','incorrect')},
            grade_denominator=len(reviewed), by_kind=by_kind,
            rubric=dict(reviewed=len(reviewed), unreviewed=planned-len(reviewed), passed=passed,
                        reviewed_pass_rate=passed/len(reviewed) if reviewed else None,
                        all_case_pass_rate=passed/planned if planned else None))
    pairs = []
    for key, a in keyed.items():
        if a['variant'] != 'A':
            continue
        d = keyed.get((key[0],key[1],'D',key[3]))
        if d and a['status'] == d['status'] == 'complete':
            pairs.append((a,d))
    positive_pairs = [(a,d) for a,d in pairs if a.get('positive') and d.get('positive')]
    metric_results = {}
    for metric in POSITIVE_METRICS:
        per_question = defaultdict(lambda: {'A': [], 'D': []})
        usable_pairs = 0
        for a,d in positive_pairs:
            if a.get(metric) is None or d.get(metric) is None:
                continue
            usable_pairs += 1
            for row in (a,d):
                per_question[(row['dataset'],row['qid'])][row['variant']].append(row[metric])
        result = {}
        means = {}
        for variant in ('A','D'):
            means[variant] = [_average(v[variant]) for v in per_question.values()]
            result[variant] = dict(mean=_average(means[variant]), question_count=len(per_question),
                                   paired_run_count=usable_pairs,
                                   na_pair_count=len(positive_pairs)-usable_pairs)
        result['delta_D_minus_A'] = (_average(means['D'])-_average(means['A'])) if per_question else None
        metric_results[metric] = result
    return dict(expected_cases=expected_cases, repeats=repeats, variants=variants,
                paired=dict(eligible_pairs=len(pairs), eligible_questions=len({a['qid'] for a,d in pairs}),
                            positive_pairs=len(positive_pairs), positive_metrics=metric_results),
                note='Custom auditable metrics; development/regression data; incomplete counterparts excluded from paired quality.')


def summarize_contexts(rows, *, expected_cases, expected_positive_cases):
    """Independent evidence assessment: each case/variant reviewed once, before any answer."""
    keyed = {}
    for row in rows:
        key = row_key(row,include_repeat=False)
        if key in keyed:
            raise ValueError('Duplicate context case/variant in summary')
        keyed[key] = row
    names = ('context_recall','context_precision')
    variants = {}
    for variant in ('A','D'):
        group = [row for row in rows if row['variant']==variant]
        complete = [row for row in group if row['status']=='complete']
        positive_complete = [row for row in complete if row['positive']]
        variants[variant] = dict(planned_questions=expected_cases,recorded=len(group),complete=len(complete),
            failed=sum(row['status']!='complete' for row in group),missing=expected_cases-len(group),
            completion_rate=len(complete)/expected_cases if expected_cases else None,
            planned_positive_questions=expected_positive_cases,positive_complete=len(positive_complete),
            negative_or_ambiguous_questions=expected_cases-expected_positive_cases,
            positive_na_or_unreviewed={name:expected_positive_cases-sum(row.get(name) is not None for row in positive_complete)
                                      for name in names})
    pairs = []
    for key,a in keyed.items():
        if a['variant']!='A' or not a['positive'] or a['status']!='complete':
            continue
        d = keyed.get((key[0],key[1],'D'))
        if d and d['positive'] and d['status']=='complete':
            pairs.append((a,d))
    scores = {}
    for name in names:
        available = [(a,d) for a,d in pairs if a.get(name) is not None and d.get(name) is not None]
        a_mean = _average([a[name] for a,d in available])
        d_mean = _average([d[name] for a,d in available])
        scores[name] = {variant:dict(mean=mean,question_count=len(available),
                                    na_or_unreviewed_positive_questions=expected_positive_cases-len(available))
                        for variant,mean in [('A',a_mean),('D',d_mean)]}
        scores[name]['delta_D_minus_A'] = d_mean-a_mean if available else None
    return dict(expected_cases=expected_cases,expected_positive_cases=expected_positive_cases,variants=variants,
                paired=dict(eligible_questions=len(pairs),
                            unavailable_positive_questions=expected_positive_cases-len(pairs),metrics=scores),
                note='Independent context reviews; positive A/D question pairs counted once; generation completion and repetitions do not affect evidence metrics.')
