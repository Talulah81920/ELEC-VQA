"""Conservative, auditable answer scoring for ELEC-VQA R3. Standard library only.

Extraction never sees gold answers/results. Ambiguous prose is not silently wrong.
This checks final answers against rule labels, not visual evidence faithfulness.
"""
import collections
import itertools
import re
import unicodedata

VERSION = 'elec-vqa-semantic-1.0'
BINARY = {'T01': 'connected', 'T06': 'after_connected', 'T09': 'compliant',
          'T10': 'restores_static_connections', 'T13': 'unique_diagnosis',
          'T14': 'sufficient_information', 'T15': 'interrupted'}


def norm(s):
    s = unicodedata.normalize('NFKC', s).casefold()
    s = re.sub('[\u2010-\u2015\u2212]', '-', s)
    return re.sub(r'\s+', ' ', s).strip()


def clean(s):
    return re.sub(r'[*`]', '', s).strip()


def final_span(text):
    """Return span + extraction provenance; do not consult a reference answer."""
    text = clean(text)
    marks = list(re.finditer(r'(?im)^\s*(?:#{1,6}\s*)?(?:final\s+answer|answer)\s*:\s*', text))
    if marks:
        return text[marks[-1].end():].strip(), 'explicit_answer'
    cues = list(re.finditer(r'(?im)(?:^|\n|(?<=[.!?])\s+)(?:#{1,6}\s*)?(?:in\s+(?:summary|conclusion)|therefore|thus|hence|so[, ]|the\s+(?:correct\s+)?answer\s+is|conclusion\s*:)', text))
    if cues:
        return text[cues[-1].start():].strip(), 'conclusion'
    lists = list(re.finditer(r'(?im)(?:^|\n)(?:the\s+)?(?:smallest sufficient combination[^\n:]*|(?:correct )?choice|pins? to be tested|contact pairs crossed in order|candidates? that fit[^\n:]*)\s+(?:is|are)\s*:', text))
    if lists:
        return text[lists[-1].end():].strip(), 'explicit_answer_list'
    return text, 'whole_response'


def display(node):
    label = node.get('label')
    if label and ':' not in label:
        x, y = node['anchor_norm']
        return f"pin {label} near the {'top' if y < .25 else 'bottom' if y > .75 else 'middle'} {'left' if x < .25 else 'right' if x > .75 else 'centre'}"
    return label


class Vocabulary:
    def __init__(self, graph):
        self.g = graph
        self.nodes = {n['id']: n for n in graph['nodes']}
        self.pins = {n['id']: display(n) for n in graph['nodes'] if n.get('label')}
        self.contacts = {}
        for e in graph['edges']:
            if e['kind'] != 'contact':
                continue
            a, b = self.pins[e['a']], self.pins[e['b']]
            if ':' in a and ':' in b and a.rsplit(':', 1)[0] == b.rsplit(':', 1)[0]:
                dev, p = a.rsplit(':', 1); q = b.rsplit(':', 1)[1]
                aliases = [f'{dev} contact {p}-{q}', f'{dev} contact {q}-{p}',
                           f'{dev}:{p}-{q}', f'{dev}:{q}-{p}',
                           f'{a}-{b}', f'{b}-{a}',
                           f'{dev} pins {p}-{q}', f'{dev} pins {q}-{p}']
            else:
                aliases = [f'the contact between {a} and {b}', f'the contact between {b} and {a}']
            self.contacts[e['id']] = aliases

    def scan(self, text, kind='pin'):
        text = norm(clean(text)); matches = []
        if kind == 'contact':
            # Shared pin suffix: "both SPC1 and SPC3 pins 11-14".
            text = re.sub(r'\b([a-z0-9_.+-]+)\s+and\s+([a-z0-9_.+-]+)\s+(?:pins?|contacts?)\s+(\w+)\s*-\s*(\w+)',
                          lambda m: f'{m[1]} contact {m[3]}-{m[4]} and {m[2]} contact {m[3]}-{m[4]}', text)
        vocabulary = {k: [v] for k, v in self.pins.items()} if kind == 'pin' else self.contacts
        for key, aliases in vocabulary.items():
            for alias in aliases:
                pattern = re.escape(norm(alias)).replace(r'\ ', r'\s+').replace(':', r'\s*:\s*')
                pattern = pattern.replace(r'\-', r'\s*-\s*')
                for m in re.finditer(r'(?<![\w.+-])' + pattern + r'(?![\w:-])', text):
                    matches.append((m.start(), m.end(), key))
        matches.sort(key=lambda x: (x[0], -(x[1]-x[0])))
        # Do not count nested aliases twice.
        chosen = []
        for item in matches:
            if not any(item[0] < e and item[1] > s for s, e, _ in chosen):
                chosen.append(item)
        return chosen

    def ids(self, text, kind='pin'):
        return [key for _, _, key in self.scan(text, kind)]

    def unknown_pins(self, text):
        known = {norm(x) for x in self.pins.values()}
        found = set(re.findall(r'(?<![\w.+-])([a-z0-9_.+-]+\s*:\s*[a-z]?\d+(?::\d+)*)(?![\w:])', norm(clean(text))))
        return sorted(p for p in found if re.sub(r'\s*:\s*', ':', p) not in known)


def result(value=None, status='resolved', reason=None, **extras):
    return {'status': status, 'value': value, 'reason': reason, **extras}


def boolean(span, task, question, method):
    s = norm(clean(span))
    # Prefix or explicitly introduced answer is stronger than incidental internal claims.
    m = re.match(r'^(?:(?:therefore|thus|hence|so)[,:]?\s*)?(?:(?:the\s+)?(?:final\s+)?answer\s+is\s*)?(yes|no)(?=$|\s*[.,;:!?]|\s+(?:it|they|this|that|because)\b)', s)
    if not m:
        m = re.search(r'\b(?:the\s+)?answer\s+is\s*[:,]?\s*(yes|no)\b', s)
    if m:
        tail = s[m.end():]
        if re.search(r'\b(?:but|however|actually)\s*,?\s*(?:the answer is\s+)?(?:yes|no)\b', tail) or re.search(r'(?:^|[.!?]\s+)(?:yes|no)[.!?]', tail):
            return result(status='ambiguous', reason='conflicting_explicit_answers')
        if task in ('T01', 'T06'):
            neg = re.search(r'\bthey (?:are not connected|remain disconnected|will not connect)\b', tail)
            pos = re.search(r'\bthey (?:are connected|remain connected|will connect)\b', tail)
            if (m.group(1) == 'yes' and neg) or (m.group(1) == 'no' and pos):
                return result(status='ambiguous', reason='polarity_and_same_subject_claim_conflict')
        return result(m.group(1) == 'yes', reason='explicit_polarity')
    if re.search(r'\b(?:cannot determine|cannot be determined|not enough information|insufficient information|unclear|depends on)\b', s):
        return result(status='unresolved', reason='uncertain_binary_conclusion')
    patterns = {
      'T01': ([r'\b(?:are|is|remain|remains)\s+(?:electrically\s+)?connected\b', r'\b(?:will|can)\s+(?:still\s+)?connect\b', r'\b(?:there is|a)\s+(?:a\s+)?(?:conducting|continuous|conductive)\s+(?:path|route)\b'],
              [r'\b(?:not connected|disconnected|isolated)\b', r'\b(?:will not|cannot|can\s+not|won\x27t)\s+(?:still\s+)?connect\b', r'\bno\s+(?:conducting|continuous|conductive)\s+(?:path|route|connection)\b']),
      'T06': ([r'\b(?:will|can)\s+(?:now\s+)?connect\b', r'\b(?:are|is|become|becomes)\s+(?:now\s+)?connected\b'],
              [r'\b(?:will not|cannot|won\x27t)\s+connect\b', r'\b(?:not connected|remain disconnected|remains disconnected)\b']),
      'T09': ([r'\b(?:meets?|satisf(?:y|ies))\s+(?:both|all|the|these|specified)\s+(?:specified\s+)?requirements\b', r'\b(?:requirements? (?:are|is) met|compliant)\b'],
              [r'\b(?:does not|doesn\x27t|fails? to)\s+(?:meet|satisfy)\b', r'\b(?:not all|neither)\s+requirements\b', r'\b(?:non-compliant|not compliant|requirements? (?:are|is) not met)\b']),
      'T10': ([r'\b(?:restores?|would restore|will restore)\s+(?:all\s+|the\s+|original\s+)*(?:original\s+)?connections\b'],
              [r'\b(?:does not|doesn\x27t|would not|will not|cannot)\s+(?:fully\s+)?restore\b', r'\bnot\s+(?:fully\s+)?restor(?:e|ed)\b']),
      'T13': ([r'\b(?:uniquely identifies|proves? the|unique diagnosis)\b'], [r'\b(?:does not prove|cannot uniquely|not sufficient|not unique)\b']),
      'T14': ([r'\b(?:sufficient information|proves? that)\b'], [r'\b(?:insufficient|does not prove|cannot establish|not sufficient)\b']),
      'T15': ([r'\b(?:will interrupt|interrupts? the route)\b'], [r'\b(?:will not interrupt|does not interrupt|unaffected)\b']),
    }
    # Only a bounded conclusion, not an entire multi-step analysis, is eligible.
    if len(s.split()) > 140 or (method == 'whole_response' and ('\n\n' in span or len(s.split()) > 65)):
        return result(status='unresolved', reason='no_bounded_binary_conclusion')
    if task in ('T01', 'T06'):
        if re.search(r'\b(?:no|not a) direct connection\b', s):
            return result(status='ambiguous', reason='no_direct_link_does_not_rule_out_indirect_path')
        # Require the queried endpoints or an explicit plural reference, not an internal contact.
        question_pins = re.findall(r'[\w.+-]+:[\w:]+', question.split('Assume')[0])[:2]
        named = all(re.search(r'(?<![\w.+-])'+re.escape(norm(x))+r'(?![\w:])', s) for x in question_pins) if len(question_pins) == 2 else False
        if not named and not re.search(r'\b(?:they|these pins|the (?:two|requested) (?:pins|terminals))\b', s):
            return result(status='unresolved', reason='polarity_not_bound_to_queried_endpoints')
    yes, no = patterns[task]
    negative = any(re.search(p, s) for p in no)
    # Remove negative clauses before checking positive phrases to avoid "not connected" overlap.
    positive_text = s
    for p in no:
        positive_text = re.sub(p, ' NEGATIVE ', positive_text)
    positive = any(re.search(p, positive_text) for p in yes)
    if positive and negative:
        return result(status='ambiguous', reason='mixed_binary_claims')
    if positive or negative:
        return result(positive, reason='task_specific_conclusion')
    return result(status='unresolved', reason='unrecognized_binary_conclusion')


def extract(row, text, vocabulary):
    """row here contains question/task/task_details only; no reference answer or result."""
    span, method = final_span(text)
    task = row['task'][:3]; s = norm(span)
    meta = {'answer_span': span, 'span_method': method}
    if not s:
        return result(status='unresolved', reason='empty_output', **meta)
    if task in BINARY:
        return {**boolean(span, task, row['question'], method), **meta}
    if task == 'T12':
        found = set(re.findall(r'\b(series|parallel)\b', s))
        return result(next(iter(found)) if len(found) == 1 else None,
                      'resolved' if len(found) == 1 else 'ambiguous', 'series_parallel', **meta)
    if task == 'T05':
        dependence = bool(re.search(r'\b(?:depends|cannot (?:be )?determin(?:e|ed)|unknown|undetermined|insufficient information|not enough information)\b', s))
        positive = bool(re.search(r'\b(?:remain|remains|always|still)\s+connected\b|\bconnected (?:either way|regardless)\b', s))
        negative = bool(re.search(r'\b(?:remain|remains|always|still)\s+disconnected\b|\bdisconnected (?:either way|regardless)\b', s))
        cases = {}
        for state in ['open', 'closed']:
            for pattern in [rf'\b(connected|disconnected)\s+when\s+(?:the (?:contact|pair) is\s+)?{state}\b',
                            rf'\b(?:when|with)\s+(?:that |the )?(?:contact |pair )?(?:is |it )?{state}[^.;:,]*?\b(disconnected|connected)\b']:
                for m in re.finditer(pattern, s):
                    value = m.group(1) == 'connected'
                    if state in cases and cases[state] != value:
                        return result(status='ambiguous', reason='conflicting_state_cases', **meta)
                    cases[state] = value
        if positive and negative:
            return result(status='ambiguous', reason='conflicting_invariant_states', **meta)
        if len(cases) == 2:
            value = cases['open'] if cases['open'] == cases['closed'] else None
            return result({'connected': value, 'cases': cases}, reason='explicit_state_cases', **meta)
        if positive or negative:
            if dependence:
                return result(status='ambiguous', reason='dependence_and_invariance', **meta)
            return result({'connected': positive, 'cases': cases}, reason='state_invariant', **meta)
        if dependence:
            return result({'connected': None, 'cases': cases}, reason='undetermined_connectivity', **meta)
        return result(status='unresolved', reason='unrecognized_unknown_state_answer', **meta)
    if task == 'T03':
        chunks = re.findall(r'\{([^{}]*)\}', span)
        syntax = 'braces'
        if not chunks:
            chunks = [re.sub(r'^\s*(?:[-*]|\d+[.)])\s*', '', line) for line in span.splitlines()
                      if re.match(r'^\s*(?:[-*]|\d+[.)])\s+', line)]
            syntax = 'bullet_groups'
        if not chunks and ';' in span and len(span.split()) < 120:
            chunks = span.split(';'); syntax = 'semicolon_groups'
        if not chunks:
            return result(status='unresolved', reason='partition_boundaries_not_explicit', **meta)
        groups = []; unknown = []
        for chunk in chunks:
            ids = vocabulary.ids(chunk)
            if not ids:
                return result(status='unresolved', reason='group_without_identifiable_pin', **meta)
            groups.append(ids); unknown.extend(vocabulary.unknown_pins(chunk))
        # Explicit unmentioned-pin singleton statements are accepted; no silent completion.
        universe = set(row.get('task_details', {}).get('pin_universe', []))
        if re.search(r'\b(?:all (?:other|remaining) pins|the (?:other|remaining) pins)\b.*\b(?:isolated|singletons|separate)\b', s):
            groups += [[p] for p in sorted(universe - set(itertools.chain.from_iterable(groups)))]
        return result(groups, reason=syntax, unknown_labels=unknown, **meta)
    if task in ('T02', 'T07', 'T11'):
        mentions = re.findall(r'[\w.+-]+\s+(?:contacts?|pins?)\s+\w+\s*-\s*\w+', s)
        # Do not silently discard an invented contact while retaining a valid one.
        if any(not vocabulary.ids(m, 'contact') for m in mentions):
            return result(status='unresolved', reason='unknown_contact_mention', **meta)
        if task == 'T02':
            if re.search(r'\bnot impossible\b|\bno (?:contact )?combination\b.*\b(?:except|unless|but)\b', s):
                return result(status='ambiguous', reason='qualified_impossibility_statement', **meta)
            impossible = bool(re.search(r'\b(?:no contact combination|no combination|cannot be connected|cannot connect|impossible|no conducting (?:path|route))\b', s))
            empty = bool(re.search(r'\b(?:none|no contacts? (?:need|needs|required)|no closure (?:is )?required|wire[- ]only)\b', s))
            if impossible:
                return result([], reason='impossible_connection', **meta)
            if empty and not vocabulary.ids(span, 'contact'):
                return result([[]], reason='no_closure_required', **meta)
            chunks = re.split(r'\s+OR\s+|\s+or\s+|;', span)
            sets = [vocabulary.ids(c, 'contact') for c in chunks]
            if not all(sets):
                return result(status='unresolved', reason='unrecognized_contact_alternative', **meta)
            # A long unbounded analysis can mention irrelevant contact states.
            if method == 'whole_response' and len(span.split()) > 100:
                return result(status='unresolved', reason='no_bounded_contact_answer', **meta)
            conflict = bool(re.search(r'\b(?:must|should|need to)\s+(?:be\s+)?(?:open|remain open)|\b(?:must|should) not close\b', s))
            return result(sets, reason='contact_alternatives', polarity_conflict=conflict, **meta)
        ids = vocabulary.ids(span, 'contact')
        if not ids:
            return result(status='unresolved', reason='no_identifiable_contact_pairs', **meta)
        if method == 'whole_response' and len(span.split()) > 100:
            return result(status='unresolved', reason='no_bounded_contact_answer', **meta)
        conflict = task == 'T07' and bool(re.search(r'\b(?:is|are|must be|stuck)\s+closed\b', s))
        return result(ids, reason='ordered_contacts' if task == 'T11' else 'candidate_set',
                      polarity_conflict=conflict, unknown_labels=vocabulary.unknown_pins(span), **meta)
    if task in ('T04', 'T08'):
        if task == 'T04' and len(row['task_details'].get('pin_universe', [])) == 1:
            b = re.match(r'^(?:yes|no)\b', s)
            if b:
                return result(row['task_details']['pin_universe'] if b[0] == 'yes' else [], reason='single_target_interruption', **meta)
        if re.search(r'\b(?:none|no pins?|no pin test|0 tests?|zero tests?)\b', s):
            return result([], reason='explicit_empty_pin_set', **meta)
        # For prose, use an explicit final lost/tested list, rather than every mentioned endpoint.
        target = span
        if task == 'T04':
            m = list(re.finditer(r'(?i)(?:pins? (?:that )?(?:lose|losing|lost)[^:\n]*:|(?:become disconnected|lose (?:their|the) connection) (?:are|:)\s*)', span))
            if m:
                target = span[m[-1].end():]
        ids = vocabulary.ids(target)
        if not ids:
            return result(status='unresolved', reason='no_identifiable_pin_set', **meta)
        if method == 'whole_response' and len(span.split()) > 75:
            return result(status='unresolved', reason='no_bounded_pin_answer', **meta)
        if task == 'T04' and re.search(r'\b(?:remain|retained|still connected)\b', norm(target)):
            return result(status='ambiguous', reason='lost_and_retained_mixed_in_answer', **meta)
        if task == 'T04':
            universe = set(row['task_details'].get('pin_universe', []))
            outside = set(ids) - universe
            source = row['task_details'].get('source')
            relational = bool(re.search(r'\b(?:connection|connections|between|to|from|lose|loses|disconnected)\b', norm(target)))
            if relational and outside:
                if outside == {source}:
                    ids = [i for i in ids if i != source]
                else:
                    return result(status='unresolved', reason='endpoint_roles_in_lost_pin_prose_need_review', **meta)
        declared = re.search(r'\b(\d+)\s+tests?\b', s) if task == 'T08' else None
        alternative_plans = None
        if task == 'T08' and re.search(r'\s+or\s+', target, re.I):
            chunks = re.split(r'\s+or\s+', target, flags=re.I)
            alternative_plans = [vocabulary.ids(c) for c in chunks]
            if not all(alternative_plans):
                return result(status='unresolved', reason='unclear_measurement_alternatives', **meta)
        return result(ids, reason='pin_set', unknown_labels=vocabulary.unknown_pins(target),
                      declared_test_count=int(declared[1]) if declared else None,
                      alternative_plans=alternative_plans, **meta)
    return result(status='unsupported', reason='unsupported_task', **meta)


def partition(groups):
    return frozenset(frozenset(g) for g in groups)


def ordered_route(graph, source, target, contacts, states):
    """Accept any simple conducting path with exactly this contact sequence."""
    adjacency = collections.defaultdict(list)
    for e in graph['edges']:
        if e['kind'] == 'contact' and states.get(e['id']) is not True:
            continue
        adjacency[e['a']].append((e['b'], e)); adjacency[e['b']].append((e['a'], e))
    stack = [(source, 0, frozenset([source]))]; budget = 100000
    while stack:
        budget -= 1
        if budget < 0:
            return None
        u, i, visited = stack.pop()
        if u == target and i == len(contacts):
            return True
        for v, e in adjacency[u]:
            if v in visited:
                continue
            j = i
            if e['kind'] == 'contact':
                if j >= len(contacts) or contacts[j] != e['id']:
                    continue
                j += 1
            stack.append((v, j, visited | {v}))
    return False


def grade(row, parsed, vocabulary):
    if parsed['status'] != 'resolved':
        return {'correct': None, 'reason': parsed['reason']}
    task = row['task'][:3]; value = parsed['value']; gold = row['result']
    if parsed.get('unknown_labels'):
        return {'correct': False, 'reason': 'unrecognized_or_out_of_graph_pin_in_explicit_answer'}
    if parsed.get('polarity_conflict'):
        return {'correct': False, 'reason': 'contact_state_contradicts_requested_operation'}
    if task in BINARY:
        return {'correct': value == gold[BINARY[task]], 'reason': 'binary_decision'}
    if task == 'T12':
        return {'correct': value == gold['connection'], 'reason': 'series_parallel'}
    if task == 'T05':
        cases = value['cases']
        correct = value['connected'] == gold['connected'] and all(v == gold[f'{k}_case'] for k, v in cases.items())
        return {'correct': correct, 'reason': 'three_way_connectivity_and_stated_cases', 'both_state_cases_explicit': len(cases) == 2}
    if task == 'T02':
        duplicate = any(len(v) != len(set(v)) for v in value) or len(partition(value)) != len(value)
        expected = partition(gold['minimal_closed_contact_sets'])
        valid = all(frozenset(v) in expected for v in value) if value else not expected
        # A valid minimal solution is accepted; completeness of alternatives is separate.
        return {'correct': not duplicate and valid, 'reason': 'proposed_minimal_contact_solutions',
                'all_minimal_alternatives_complete': not duplicate and partition(value) == expected,
                'all_proposed_sets_valid': valid,
                'minimum_cardinality_solution_present': bool(expected and any(frozenset(v) in expected and len(v) == min(map(len,expected)) for v in value))}
    if task == 'T03':
        ids = list(itertools.chain.from_iterable(value)); gold_groups = gold['wire_groups']
        pred_groups = [[vocabulary.pins[p] for p in group] for group in value]
        # Display names are used in generated references, including spatial pin descriptions.
        valid = len(ids) == len(set(ids)) and len(value) == len(partition(value))
        return {'correct': valid and partition(pred_groups) == partition(gold_groups), 'reason': 'full_partition_with_boundaries_and_singletons'}
    if task == 'T04':
        return {'correct': len(value) == len(set(value)) and {vocabulary.pins[p] for p in value} == set(gold['lost_pins']), 'reason': 'exact_lost_pin_set'}
    if task == 'T07':
        return {'correct': len(value) == len(set(value)) and set(value) == set(gold['compatible_contacts']), 'reason': 'all_compatible_candidates'}
    if task == 'T08':
        details = row['task_details']; order = details['test_pin_order']; vectors = details['all_vectors']
        all_patterns = len({tuple(v) for v in vectors.values()})
        checks = []
        for plan in parsed.get('alternative_plans') or [value]:
            if len(plan) != len(set(plan)) or any(p not in order for p in plan):
                return {'correct': False, 'reason': 'duplicate_or_unavailable_test_pin'}
            indices = [order.index(p) for p in plan]
            enough = len({tuple(v[i] for i in indices) for v in vectors.values()}) == all_patterns
            checks.append({'separates_observable_patterns': enough, 'minimum_size': len(plan) == gold['minimum_test_count'],
                           'declared_count_consistent': parsed.get('declared_test_count') in (None, len(plan))})
        return {'correct': all(all(c.values()) for c in checks), 'reason': 'minimum_size_and_observable_pattern_separation', 'plans': checks}
    if task == 'T11':
        checks = row['verification']['graph_checks']
        q = checks[0]
        if len(value) != len(set(value)):
            return {'correct': False, 'reason': 'repeated_contact_not_simple_route'}
        valid = ordered_route(vocabulary.g, q['source'], q['target'], value, q['states'])
        return {'correct': valid, 'reason': 'valid_alternative_ordered_path' if valid is not None else 'path_search_budget_exceeded'}
    return {'correct': None, 'reason': 'unsupported_task'}


def score_one(row, prediction, graph):
    vocabulary = Vocabulary(graph)
    # Extraction is deliberately denied reference answer, result, and verification.
    public = {k: row[k] for k in ('task', 'question', 'task_details')}
    parsed = extract(public, prediction['output_text'], vocabulary)
    graded = grade(row, parsed, vocabulary)
    return {'id': row['id'], 'task': row['task'], 'parent': row['base_circuit_id'],
            'source_project': row['source_project'], 'parsed': parsed, **graded,
            'needs_review': graded['correct'] is None,
            'generation_limit_hit': prediction.get('generation_limit_hit'),
            'evidence_faithfulness_scored': False}


def summarize(items):
    groups = collections.defaultdict(list)
    for item in items:
        groups['overall'].append(item); groups[item['task']].append(item)
    output = {}
    for name, values in groups.items():
        n = len(values); c = sum(v['correct'] is True for v in values); w = sum(v['correct'] is False for v in values)
        u = n-c-w
        output[name] = {'count': n, 'correct': c, 'incorrect': w, 'unresolved': u,
                        'resolution_rate': (c+w)/n,
                        'accuracy_on_resolved_only': c/(c+w) if c+w else None,
                        'correct_fraction_all_samples': c/n,
                        'unresolved_score_interval': [c/n, (c+u)/n]}
    task_values = [v for k, v in output.items() if k != 'overall']
    return {'groups': output,
            'task_macro_unresolved_interval': [sum(v['unresolved_score_interval'][i] for v in task_values)/len(task_values) for i in (0, 1)] if task_values else None,
            'note': 'Intervals reflect unresolved answers only, NOT statistical confidence or proven extraction correctness. Resolved-only accuracy is selection-biased; do not rank models by it alone.'}
