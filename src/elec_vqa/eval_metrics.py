"""Conservative lexical/anchor metrics, not a semantic electrical correctness judge."""
import json
import math
import unicodedata
from collections import Counter, defaultdict


def normalize(value):
    return ''.join(unicodedata.normalize('NFKC', str(value)).casefold().split())


def parse_output(text):
    stripped = text.strip()
    strict = True
    if stripped.startswith('```') and stripped.endswith('```'):
        stripped = '\n'.join(stripped.splitlines()[1:-1]).strip()
        strict = False
    try:
        obj = json.loads(stripped)
    except (ValueError, TypeError):
        return None, False
    if not isinstance(obj, dict) or not isinstance(obj.get('answer'), str):
        return None, False
    return obj, strict


def char_f1(prediction, reference):
    p, r = Counter(normalize(prediction)), Counter(normalize(reference))
    if not p or not r:
        return float(p == r)
    common = sum((p & r).values())
    return 2 * common / (sum(p.values()) + sum(r.values()))


def anchors(items):
    out = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        xy = item.get('anchor_norm')
        if not isinstance(item.get('label'), str) or not isinstance(xy, list) or len(xy) != 2:
            continue
        if all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) and 0 <= v <= 1 for v in xy):
            out.append((normalize(item['label']), xy))
    return out


def anchor_match(predicted, reference, tolerance):
    # Maximum bipartite matching: repeated labels cannot reuse a reference anchor.
    links = [[j for j, (label, xy) in enumerate(reference)
              if label == plabel and math.dist(pxy, xy) <= tolerance]
             for plabel, pxy in predicted]
    matched = {}
    def augment(i, seen):
        for j in links[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched or augment(matched[j], seen):
                matched[j] = i
                return True
        return False
    return sum(augment(i, set()) for i in range(len(predicted)))


def result_signature(value):
    if not isinstance(value, list) or not value:
        return None
    if any(not isinstance(v, dict) or 'result' not in v for v in value):
        return None
    return json.dumps([v['result'] for v in value], sort_keys=True, ensure_ascii=False)


def score_one(prediction, reference, tolerance=0.05):
    ref, _ = parse_output(reference)
    if ref is None:
        raise ValueError('Reference must be a JSON object with string answer')
    pred, strict = parse_output(prediction)
    answer = pred['answer'] if pred else prediction.strip()
    pa = anchors(pred.get('evidence', [])) if pred else []
    ra = anchors(ref.get('evidence', []))
    # Invalid predicted anchors count as false positives too.
    emitted = pred.get('evidence', []) if pred else []
    pred_count = len(emitted) if isinstance(emitted, list) else int(bool(emitted))
    matches = anchor_match(pa, ra, tolerance)
    required = ('answer', 'evidence', 'basis', 'conditions', 'scope', 'knowledge_source')
    schema = bool(pred and all(k in pred for k in required)
                  and isinstance(pred['evidence'], list)
                  and isinstance(pred['conditions'], list)
                  and all(isinstance(pred[k], str) for k in ('basis', 'scope', 'knowledge_source')))
    rs = result_signature(ref.get('inference'))
    ps = result_signature(pred.get('inference')) if pred else None
    return {
        'json_parsed': bool(pred), 'strict_json': strict if pred else False,
        'required_schema': schema,
        'answer_exact_match': normalize(answer) == normalize(ref['answer']),
        'answer_char_f1': char_f1(answer, ref['answer']),
        'reference_anchor_count': len(ra), 'predicted_anchor_count': pred_count,
        'matched_anchor_count': matches,
        'inference_result_exact_match': (ps == rs) if rs is not None else None,
    }


def summarize(records):
    def group_summary(rows):
        n = len(rows)
        if not n:
            return {'count': 0}
        result = {'count': n}
        for key in ('json_parsed', 'strict_json', 'required_schema',
                    'answer_exact_match', 'answer_char_f1'):
            result[key] = sum(r['scores'][key] for r in rows) / n
        eligible = [r for r in rows if r['scores']['reference_anchor_count'] > 0]
        result['anchor_evaluated_samples'] = len(eligible)
        if eligible:
            tp = sum(r['scores']['matched_anchor_count'] for r in eligible)
            pc = sum(r['scores']['predicted_anchor_count'] for r in eligible)
            rc = sum(r['scores']['reference_anchor_count'] for r in eligible)
            result.update(anchor_precision=tp / pc if pc else 0.0,
                          anchor_recall=tp / rc,
                          anchor_f1=2 * tp / (pc + rc) if pc + rc else 0.0)
        conditional = [r['scores']['inference_result_exact_match'] for r in rows
                       if r['scores']['inference_result_exact_match'] is not None]
        result['inference_result_evaluated_samples'] = len(conditional)
        result['inference_result_exact_match'] = sum(conditional) / len(conditional) if conditional else None
        result['generation_limit_hit_count'] = sum(r.get('generation_limit_hit', False) for r in rows)
        result['total_generation_seconds'] = sum(r.get('generation_seconds', 0) for r in rows)
        return result
    groups = defaultdict(list)
    for record in records:
        groups[record['type_id']].append(record)
    return {'overall': group_summary(records),
            'by_type': {k: group_summary(v) for k, v in sorted(groups.items())}}
