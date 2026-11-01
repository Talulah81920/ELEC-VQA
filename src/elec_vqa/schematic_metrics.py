"""Gold-graph certificate checks; no gold graph is supplied to model inference."""
import copy
import collections
import json
import math
import unicodedata
from .schematic_graph import reachable
from .eval_metrics import parse_output, score_one, anchors, anchor_match
from . import schematic_engineering as engineering

SET_FIELDS={'minimal_closed_contact_sets','required_contacts','lost_terminals','retained_terminals',
            'compatible_single_open_edges','tested','minimal_sets','minimum_test_targets','distinguishable_fault_groups',
            'remaining_lost_pairs','unintended_connected_pairs','external_wire_groups','reference_connected_pairs',
            'proposed_connected_pairs','violations','reference_terminal_groups','proposed_terminal_groups','reference_groups','proposed_groups'}

def canonical(obj,key=None):
    if isinstance(obj,dict):return {k:canonical(v,k) for k,v in sorted(obj.items())}
    if isinstance(obj,list):
        values=[canonical(v) for v in obj]
        if key in SET_FIELDS:
            values=[sorted(x) if isinstance(x,list) else x for x in values]
            return sorted(values,key=lambda x:json.dumps(x,ensure_ascii=False,sort_keys=True))
        return values
    if isinstance(obj,str):return ''.join(unicodedata.normalize('NFKC',obj).casefold().split())
    return obj

def equal(a,b):return json.dumps(canonical(a),ensure_ascii=False,sort_keys=True)==json.dumps(canonical(b),ensure_ascii=False,sort_keys=True)

def result_correct(g,result,record):
    if record['operation']=='ordered_path':
        if not isinstance(result,dict) or set(result)!={'ordered_terminal_labels','contact_sequence'}:return False
        labels=result['ordered_terminal_labels'];contacts=result['contact_sequence'];inp=record['inputs']
        nn={n['id']:n for n in g['nodes']}
        if not isinstance(labels,list) or not isinstance(contacts,list) or not labels:return False
        if len(set(labels))!=len(labels) or len(set(contacts))!=len(contacts):return False
        if labels[0]!=nn[inp['source']]['label'] or labels[-1]!=nn[inp['target']]['label']:return False
        adj=collections.defaultdict(list)
        for e in g['edges']:adj[e['a']].append((e['b'],e));adj[e['b']].append((e['a'],e))
        queue=collections.deque([(inp['source'],1,0)]);seen=set(queue)
        while queue:
            u,i,j=queue.popleft()
            if u==inp['target'] and i==len(labels) and j==len(contacts):return True
            for v,e in adj[u]:
                ii,jj=i,j
                if e['kind']=='contact':
                    if jj>=len(contacts) or contacts[jj]!=e['id']:continue
                    jj+=1
                if nn[v]['label'] is not None:
                    if ii>=len(labels) or labels[ii]!=nn[v]['label']:continue
                    ii+=1
                state=(v,ii,jj)
                if state not in seen:seen.add(state);queue.append(state)
        return False
    if record['operation']!='measurement_plan':return equal(result,record['result'])
    if not isinstance(result,dict):return False
    reference=record['result'];targets=result.get('minimum_test_targets')
    if not isinstance(targets,list) or len(targets)!=reference['minimum_test_count'] or len(set(targets))!=len(targets):return False
    vectors=engineering.test_vectors(g,record['inputs'])
    if any(t not in next(iter(vectors.values())) for t in targets):return False
    if not equal(engineering.grouped_vectors(vectors,targets),reference['distinguishable_fault_groups']):return False
    return equal({k:v for k,v in result.items() if k!='minimum_test_targets'},
                 {k:v for k,v in reference.items() if k!='minimum_test_targets'})

def _path(g,a,b,ids,states,removed=(),possible=False):
    em={e['id']:e for e in g['edges']};u=a
    for eid in ids:
        if eid not in em or eid in removed:return False
        e=em[eid];state=True if e['kind']=='wire' else states.get(eid)
        if state is not True and not (possible and state is None):return False
        if u not in (e['a'],e['b']):return False
        u=e['b'] if u==e['a'] else e['a']
    return u==b

def _witness(g,d,a,b,states,removed=()):
    kind=d.get('kind')
    if kind=='conducting_path':return _path(g,a,b,d.get('edge_ids',[]),states,removed)
    if kind=='possible_path_with_unknown_contacts':
        ids=d.get('edge_ids',[]);em={e['id']:e for e in g['edges']}
        return (_path(g,a,b,ids,states,removed,True) and not reachable(g,a,b,states,removed)[0]
                and any(em[e]['kind']=='contact' and states.get(e) is None for e in ids))
    if kind=='blocked_frontier_in_declared_graph':
        cut=d.get('blocked_edges',[]);em={e['id']:e for e in g['edges']}
        if any(e not in em for e in cut):return False
        if any(not (e in removed or (em[e]['kind']=='contact' and states.get(e) is False)) for e in cut):return False
        allclosed={e['id']:True for e in g['edges'] if e['kind']=='contact'}
        return not reachable(g,a,b,allclosed,cut,True)[0]
    return False

def _relations(g,pred,record):
    nn={n['id']:n for n in g['nodes']};gold=list(g['edges']);inp=record['inputs']
    if record['operation']=='jumper':
        c=next(e for e in gold if e['id']==inp['contact'])
        gold.append({'id':'J_added','kind':'hypothetical_jumper','a':c['a'],'b':c['b'],
                     'polyline_norm':[nn[c['a']]['anchor_norm'],nn[c['b']]['anchor_norm']]})
    def label(n):return nn[n]['label'] or '@'+','.join(map(str,nn[n]['anchor_norm']))
    def endpoint(a,b):
        if isinstance(a,str) and isinstance(b,str) and a.startswith('@') and b.startswith('@'):
            try:return math.dist(list(map(float,a[1:].split(','))),list(map(float,b[1:].split(','))))<=.025
            except ValueError:return False
        return equal(a,b)
    fixed=set(inp.get('candidates',[]))|{inp.get('edge'),inp.get('contact')}
    mapping={};used=set();valid=0;items=pred.get('relations',[])
    if not isinstance(items,list):return {},0,1
    for e in items:
        if not isinstance(e,dict) or not isinstance(e.get('id'),str) or e['id'] in mapping:continue
        poly=e.get('polyline_norm')
        if not isinstance(poly,list) or len(poly)<2:continue
        try:
            if any(not isinstance(p,list) or len(p)!=2 or any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or not 0<=x<=1 for x in p) for p in poly):continue
        except TypeError:continue
        for x in gold:
            if x['id'] in used or x['kind']!=e.get('kind'):continue
            if (x['kind']=='contact' or e['id'] in fixed) and e['id']!=x['id']:continue
            ends=[label(x['a']),label(x['b'])]
            if not ((endpoint(e.get('from'),ends[0]) and endpoint(e.get('to'),ends[1])) or
                    (endpoint(e.get('to'),ends[0]) and endpoint(e.get('from'),ends[1]))):continue
            gp=x['polyline_norm']
            forward=math.dist(poly[0],gp[0])<=.025 and math.dist(poly[-1],gp[-1])<=.025
            backward=math.dist(poly[0],gp[-1])<=.025 and math.dist(poly[-1],gp[0])<=.025
            if not (forward or backward):continue
            # Wires are merged straight sections; reject a polyline taking an unsupported detour.
            if x['kind']=='wire':
                ax,ay=gp[0];bx,by=gp[-1];den=(bx-ax)**2+(by-ay)**2
                if den:
                    def dist(p):
                        t=max(0,min(1,((p[0]-ax)*(bx-ax)+(p[1]-ay)*(by-ay))/den))
                        return math.dist(p,[ax+t*(bx-ax),ay+t*(by-ay)])
                    if any(dist(p)>.025 for p in poly):continue
            mapping[e['id']]=x['id'];used.add(x['id']);valid+=1;break
    return mapping,valid,len(items)

def _remap(obj,mapping):
    if isinstance(obj,str):return mapping.get(obj,obj)
    if isinstance(obj,list):return [_remap(x,mapping) for x in obj]
    if isinstance(obj,dict):return {k:_remap(v,mapping) for k,v in obj.items()}
    return obj

def certificate_valid(g,pred,r,mapping,relation_valid,relation_total,lexical):
    if relation_valid!=relation_total:return False
    if not equal(pred.get('conditions'),r['supervision']['conditions']) or not equal(pred.get('scope'),r['supervision']['scope']):return False
    steps=pred.get('proof_steps');inp=r['inputs'];op=r['operation'];nn={n['id']:n for n in g['nodes']}
    if not isinstance(steps,list) or not steps:return False
    if not isinstance(steps[-1],dict) or not equal(pred.get('result'),steps[-1].get('result')):return False
    facts=pred.get('evidence',[]);rels=pred.get('relations',[])
    if any(not isinstance(f,dict) or not isinstance(f.get('id'),str) for f in facts):return False
    pa=anchors(facts);allfacts=[];seen=set()
    for f in g['visual_evidence']:
        sig=(f['label'],tuple(f['anchor_norm']))
        if sig not in seen:allfacts.append(f);seen.add(sig)
    if len(pa)!=len(facts) or anchor_match(pa,anchors(allfacts),.05)!=len(pa):return False
    needed={inp[k] for k in ('source','target') if k in inp}
    for e in g['edges']:
        if e['id'] in mapping.values():needed.update([e['a'],e['b']])
    if op in ('fault_impact','diagnosis'):needed.update(n['id'] for n in g['nodes'] if n.get('is_terminal'))
    position=0;required=[];seen=set()
    for n in g['nodes']:
        for _ in n['labels']:
            f=g['visual_evidence'][position];position+=1;sig=(f['label'],tuple(f['anchor_norm']))
            if n['id'] in needed and sig not in seen:required.append(f);seen.add(sig)
    if anchor_match(pa,anchors(required),.05)!=len(required):return False
    defined={f['id'] for f in facts}|{e['id'] for e in rels}
    if len(defined)!=len(facts)+len(rels):return False
    for step in steps:
        if not isinstance(step,dict) or not isinstance(step.get('id'),str) or step['id'] in defined:return False
        if not isinstance(step.get('uses'),list) or any(x not in defined for x in step['uses']):return False
        defined.add(step['id'])
    gold=r['supervision']['proof_steps']
    if op=='identify_ports':return len(steps)==1 and steps[0].get('rule')==gold[0]['rule'] and equal(steps[0].get('result'),r['result'])
    if len(steps)!=3 or [s.get('rule') for s in steps]!=[s['rule'] for s in gold]:return False
    if not equal(steps[-1].get('inputs'),gold[-1]['inputs']) or not result_correct(g,steps[-1].get('result'),r):return False
    d=_remap(steps[1].get('result',{}),mapping)
    if not isinstance(d,dict):return False
    a=inp.get('source');b=inp.get('target');kk=[e['id'] for e in g['edges'] if e['kind']=='contact'];closed={k:True for k in kk}
    if op in engineering.OPERATIONS:
        _,expected=engineering.proof_witness(g,op,inp)
        return set(mapping.values())=={e['id'] for e in g['edges']} and equal(d,expected)
    if op in ('connectivity','wire_claim','ordered_path'):
        st=inp.get('states',closed)
        if op=='wire_claim':st={k:False for k in kk}
        if op=='ordered_path':
            if not _witness(g,d,a,b,st,inp.get('removed',[])):return False
            u=a;labels=[nn[a]['label']];contacts=[];em={e['id']:e for e in g['edges']}
            for eid in d['edge_ids']:
                e=em[eid];u=e['b'] if u==e['a'] else e['a']
                if nn[u]['label']:labels.append(nn[u]['label'])
                if e['kind']=='contact':contacts.append(eid)
            return equal(pred.get('result'),{'ordered_terminal_labels':labels,'contact_sequence':contacts})
        return _witness(g,d,a,b,st,inp.get('removed',[]))
    if op=='control_dependency':
        sets=r['result']['minimal_closed_contact_sets']
        if not equal(d.get('minimal_sets'),sets) or len(d.get('paths',[]))!=len(sets):return False
        remaining=list(sets)
        for path in d['paths']:
            matched=next((i for i,ss in enumerate(remaining) if _witness(g,path,a,b,{k:k in ss for k in kk})),None)
            if matched is None:return False
            remaining.pop(matched)
        return not remaining
    if op=='fault_impact':
        targets=d.get('targets',{});named={n['label']:n['id'] for n in g['nodes'] if n.get('is_terminal') and n['id']!=a}
        if set(targets)!=set(named) or d.get('opened_edge')!=inp['edge']:return False
        return all(_witness(g,targets[label],a,n,closed,[inp['edge']]) for label,n in named.items())
    if op=='diagnosis':
        return equal(d.get('tested'),inp['candidates']) and equal(d.get('observations'),gold[1]['result']['observations'])
    if op=='jumper':
        if d.get('contact')!=inp['contact'] or not _witness(g,d.get('before',{}),a,b,inp['states']):return False
        gg=copy.deepcopy(g);e=next(e for e in g['edges'] if e['id']==inp['contact'])
        gg['edges'].append({'id':'J_added','kind':'wire','a':e['a'],'b':e['b']})
        return _path(gg,a,b,d.get('after_path',[]),inp['states']) if r['result']['after'] is True else d.get('after_path')==[]
    return False

def score(prediction,record,graph,view='structured'):
    reference=json.dumps(record['supervision'],ensure_ascii=False);lex=score_one(prediction,reference)
    pred,_=parse_output(prediction)
    try:correct=bool(pred and 'result' in pred and result_correct(graph,pred['result'],record))
    except (TypeError,ValueError,KeyError):correct=False
    out={'parsed':pred is not None,'result_exact':correct,'answer_exact':lex['answer_exact_match'],
         'answer_char_f1':lex['answer_char_f1'],'proof_evaluated':view=='structured',
         'proof_certificate_valid':None,'result_and_certificate_correct':None,'strict_answer_and_certificate_correct':None}
    if view=='structured':
        mapping,valid,total=_relations(graph,pred or {},record)
        try:cert=bool(pred and certificate_valid(graph,pred,record,mapping,valid,total,lex))
        except (KeyError,TypeError,ValueError,IndexError):cert=False
        out.update(proof_certificate_valid=cert,result_and_certificate_correct=correct and cert,
                   strict_answer_and_certificate_correct=correct and cert and lex['answer_exact_match'],
                   supported_relation_count=valid,predicted_relation_count=total,
                   reference_anchor_count=lex['reference_anchor_count'],predicted_anchor_count=lex['predicted_anchor_count'],
                   matched_anchor_count=lex['matched_anchor_count'])
    return out
