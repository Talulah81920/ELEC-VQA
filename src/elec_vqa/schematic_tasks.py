"""Executable local-circuit tasks and independently callable answer/proof contract."""
import collections
import copy
import itertools
from .schematic_graph import solve, reachable

def node_map(g): return {n['id']:n for n in g['nodes']}
def gates(g): return [e['id'] for e in g['edges'] if e['kind']=='contact']

def minimal_sets(g,a,b):
    kk=gates(g); found=[]
    for size in range(len(kk)+1):
        for ss in itertools.combinations(kk,size):
            if any(set(x).issubset(ss) for x in found): continue
            state={k:k in ss for k in kk}
            if solve(g,a,b,state)['reachable'] is True: found.append(list(ss))
    return found

def states_description(g, states):
    nn=node_map(g); parts=[]
    for e in g['edges']:
        if e['kind']!='contact':continue
        value=states.get(e['id']); text='未知' if value is None else ('闭合' if value else '断开')
        parts.append(f"{e['id']}（{nn[e['a']]['label']}—{nn[e['b']]['label']}）{text}")
    return '；'.join(parts)

def wire_description(e):
    return f"{e['id']}（归一化折线{e['polyline_norm']}）"

def execute(g, op, inp):
    kk=gates(g);a=inp.get('source');b=inp.get('target');nn=node_map(g)
    if op=='identify_ports': return {'terminals':[nn[a]['label'],nn[b]['label']]}
    if op=='control_dependency':
        sets=minimal_sets(g,a,b); required=sorted(set.intersection(*(set(s) for s in sets))) if sets else []
        return {'source':nn[a]['label'],'target':nn[b]['label'],'minimal_closed_contact_sets':sets,
                'required_contacts':required,'wire_only_connected':sets==[[]]}
    if op=='connectivity':
        return {'source':nn[a]['label'],'target':nn[b]['label'],
                'reachable':solve(g,a,b,inp['states'],inp.get('removed',[]))['reachable']}
    if op=='fault_impact':
        st={k:True for k in kk}; lost=[];retained=[]
        for n in g['nodes']:
            if not n.get('is_terminal') or n['id']==a: continue
            before=solve(g,a,n['id'],st)['reachable'];after=solve(g,a,n['id'],st,[inp['edge']])['reachable']
            if before and not after:lost.append(n['label'])
            elif before and after:retained.append(n['label'])
        return {'source':nn[a]['label'],'opened_edge':inp['edge'],
                'lost_terminals':sorted(lost),'retained_terminals':sorted(retained)}
    if op=='diagnosis':
        valid=[];st={k:True for k in kk}
        for eid in inp['candidates']:
            if all(solve(g,a,n,st,[eid])['reachable']==state for n,state in inp['observations'].items()):valid.append(eid)
        return {'compatible_single_open_edges':valid,'unique':len(valid)==1}
    if op=='jumper':
        gg=copy.deepcopy(g);e=next(e for e in g['edges'] if e['id']==inp['contact'])
        gg['edges'].append({'id':'J_added','kind':'wire','a':e['a'],'b':e['b']})
        return {'before':solve(g,a,b,inp['states'])['reachable'],
                'after':solve(gg,a,b,inp['states'])['reachable'], 'jumper_is_hypothetical':True}
    if op=='wire_claim':
        value=solve(g,a,b,{k:False for k in kk})['reachable']
        return {'wire_only_connected':value,'claim_supported':value==inp['claim']}
    if op=='ordered_path':
        ok,path=reachable(g,a,b,{k:True for k in kk});assert ok
        em={e['id']:e for e in g['edges']};u=a;pp=[nn[a]['label']];cg=[]
        for eid in path:
            e=em[eid];u=e['b'] if e['a']==u else e['a']
            if nn[u]['label']:pp.append(nn[u]['label'])
            if e['kind']=='contact':cg.append(eid)
        return {'ordered_terminal_labels':pp,'contact_sequence':cg}
    raise ValueError(op)

def proof_witness(g,op,inp):
    """Witness paths or blocked frontiers, checked against the full gold local graph."""
    kk=gates(g);a=inp.get('source');b=inp.get('target');edges=set();detail={}
    st=inp.get('states',{k:True for k in kk});removed=inp.get('removed',[])
    def witness(source,target,states,removed=()):
        lower,p=reachable(g,source,target,states,removed,False)
        upper,up=reachable(g,source,target,states,removed,True)
        if lower:return p,{'kind':'conducting_path','edge_ids':p}
        if upper:
            unknown=[eid for eid in up if eid in kk and states.get(eid) is None]
            return up,{'kind':'possible_path_with_unknown_contacts','edge_ids':up,'unknown_contacts':unknown}
        seen={n['id'] for n in g['nodes'] if reachable(g,source,n['id'],states,removed,True)[0]}
        cut=[e['id'] for e in g['edges'] if (e['a'] in seen)!=(e['b'] in seen)]
        return cut,{'kind':'blocked_frontier_in_declared_graph','blocked_edges':cut,'source_region':sorted(seen)}
    if op in ('connectivity','wire_claim','ordered_path'):
        if op=='wire_claim':st={k:False for k in kk}
        ids,detail=witness(a,b,st,removed);edges.update(ids)
    elif op=='control_dependency':
        sets=minimal_sets(g,a,b);details=[]
        for ss in sets:
            ids,d=witness(a,b,{k:k in ss for k in kk});edges.update(ids);details.append(d)
        detail={'kind':'minimal_sufficient_sets','paths':details,'minimal_sets':sets}
    elif op=='fault_impact':
        detail={'kind':'separate_branch_reachability','opened_edge':inp['edge'],'targets':{}}
        edges.add(inp['edge'])
        for n in g['nodes']:
            if not n.get('is_terminal') or n['id']==a:continue
            ids,d=witness(a,n['id'],{k:True for k in kk},[inp['edge']]);edges.update(ids);detail['targets'][n['label']]=d
    elif op=='diagnosis':
        edges.update(inp['candidates']);detail={'kind':'enumerate_single_open_candidates','tested':inp['candidates'],
                                               'observations':inp['observations']}
        for eid in inp['candidates']:
            for target in inp['observations']:
                ids,_=witness(a,target,{k:True for k in kk},[eid]);edges.update(ids)
    elif op=='jumper':
        ids,d=witness(a,b,st);edges.update(ids);edges.add(inp['contact'])
        gg=copy.deepcopy(g);ce=next(e for e in g['edges'] if e['id']==inp['contact'])
        gg['edges'].append({'id':'J_added','kind':'wire','a':ce['a'],'b':ce['b']})
        _,after=reachable(gg,a,b,st)
        edges.update(after)
        detail={'kind':'hypothetical_parallel_bridge','contact':inp['contact'],'before':d,
                'after_path':after,'after_rule':'add_conducting_edge_between_contact_endpoints_then_solve'}
    elif op=='identify_ports':
        detail={'kind':'read_source_matched_terminal_attributes','nodes':[a,b]}
    return sorted(edges),detail

def answer_text(op,result):
    if op=='identify_ports':return '两处完整端子标识为'+'、'.join(result['terminals'])+'。'
    if op=='control_dependency':
        if result['wire_only_connected']:return '所问两端由外部导线同网相连，不需要这些接点闭合来建立该局部连接。'
        ss=result['minimal_closed_contact_sets']; text='；或'.join('与'.join(x)+'同时闭合' for x in ss)
        if not ss:return '在当前图示局部结构中，不存在仅靠闭合列明接点就能建立所问通路的组合；不能把接点闭合当作导线连接完整。'
        return f"该局部结构对{result['source']}到{result['target']}起通路许可控制作用。最小许可组合为：{text}；外部导线仍须完好。"
    if op=='connectivity':
        v=result['reachable'];return (f"{result['source']}到{result['target']}在限定局部模型中"+
            ('可达。' if v is True else '不可达。' if v is False else '无法确定；必要接点条件仍有未知。'))
    if op=='fault_impact':
        a='、'.join(result['lost_terminals']) or '无';b='、'.join(result['retained_terminals']) or '无'
        return f"该假设开路后，从{result['source']}失去局部通路的端子：{a}；仍保留通路的端子：{b}。这不是现场故障判断。"
    if op=='diagnosis':
        valid='、'.join(result['compatible_single_open_edges']) or '无'
        return ('能在给定候选内唯一确定。' if result['unique'] else '不能唯一确定。')+f'相容的单处开路候选为：{valid}。'
    if op=='jumper':
        f=lambda x:'可达' if x is True else '不可达' if x is False else '未知'
        return f"新增题设跨线前{f(result['before'])}，跨线后{f(result['after'])}；不表示原图已经存在跨线。"
    if op=='wire_claim':return ('候选说法成立。' if result['claim_supported'] else '候选说法不成立。')+('所问两端外部导线同网。' if result['wire_only_connected'] else '不能只靠所示外部导线连通两端。')
    if op=='ordered_path':return '有序经过的带号端子为'+'→'.join(result['ordered_terminal_labels'])+'；接点次序为'+('→'.join(result['contact_sequence']) or '无，仅外部导线')+'。'
    raise ValueError(op)
