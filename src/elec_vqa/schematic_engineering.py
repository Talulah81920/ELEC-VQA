"""Engineering checks on declared terminal networks; no assumed live device states."""
import collections,copy,itertools
from .schematic_graph import reachable
from .schematic_tasks import gates,node_map

OPERATIONS={'wire_partition','repair_check','measurement_plan','design_check'}
def terminals(g):return sorted((n for n in g['nodes'] if n.get('is_terminal')),key=lambda n:n['label'])
def matrix(g,removed=(),extra=None,wire_only=False):
    gg=copy.deepcopy(g)
    if extra:gg['edges'].append({'id':'J_repair','kind':'wire','a':extra[0],'b':extra[1]})
    st={k:not wire_only for k in gates(gg)}
    return {tuple(sorted((a['label'],b['label']))):reachable(gg,a['id'],b['id'],st,removed)[0]
            for a,b in itertools.combinations(terminals(gg),2)}
def test_vectors(g,inp):
    tt=[n for n in terminals(g) if n['id']!=inp['source']];closed={k:True for k in gates(g)}
    return {eid:{n['label']:reachable(g,inp['source'],n['id'],closed,[eid])[0] for n in tt}
            for eid in inp['candidates']}
def partition(g,removed=(),extra=None):
    values=matrix(g,removed,extra);remaining={n['label'] for n in terminals(g)};groups=[]
    while remaining:
        a=min(remaining);members=sorted(b for b in remaining if a==b or values[tuple(sorted((a,b)))])
        groups.append(members);remaining-=set(members)
    return sorted(groups)
def grouped_vectors(vectors,tests):
    groups=collections.defaultdict(list)
    for e,values in vectors.items():groups[tuple(values[t] for t in tests)].append(e)
    return sorted((sorted(v) for v in groups.values()))
def execute(g,op,inp):
    if op=='design_check':
        nn=node_map(g);violations=[]
        for p in inp['requirements']:
            actual=reachable(g,p['a'],p['b'],inp['states'])[0]
            if actual!=p['connected']:
                violations.append({'from':nn[p['a']]['label'],'to':nn[p['b']]['label'],
                                   'required_connected':p['connected'],'shown_connected':actual})
        return {'complies_with_given_requirements':not violations,'violations':violations,
                'unlisted_design_requirements_not_evaluated':True}
    if op=='wire_partition':
        groups=[];remaining={n['id']:n for n in terminals(g)};st={k:False for k in gates(g)}
        while remaining:
            a=next(iter(remaining));members=[n for n in remaining if reachable(g,a,n,st)[0]]
            groups.append(sorted(remaining[n]['label'] for n in members))
            for n in members:remaining.pop(n)
        return {'external_wire_groups':sorted(groups),'device_internal_paths_excluded':True}
    if op=='repair_check':
        before=partition(g);after=partition(g,[inp['opened_wire']],inp['proposed_link'])
        return {'restores_original_terminal_partition':before==after,
                'reference_terminal_groups':before,'proposed_terminal_groups':after,
                'only_static_given_contact_states_checked':True}
    if op=='measurement_plan':
        vectors=test_vectors(g,inp);labels=sorted(next(iter(vectors.values())))
        allgroups=grouped_vectors(vectors,labels);chosen=None
        for n in range(len(labels)+1):
            for ss in itertools.combinations(labels,n):
                if grouped_vectors(vectors,ss)==allgroups:chosen=list(ss);break
            if chosen is not None:break
        return {'minimum_test_targets':chosen,'minimum_test_count':len(chosen),
                'distinguishable_fault_groups':allgroups,
                'all_candidates_uniquely_identifiable':all(len(x)==1 for x in allgroups),
                'test_model':'isolated_deenergized_continuity_from_given_reference_terminal'}
    raise ValueError(op)
def proof_witness(g,op,inp):
    edges=[e['id'] for e in g['edges']]
    if op=='design_check':
        nn=node_map(g);checks=[]
        for p in inp['requirements']:
            value,path=reachable(g,p['a'],p['b'],inp['states'])
            checks.append({'from':nn[p['a']]['label'],'to':nn[p['b']]['label'],'required_connected':p['connected'],
                           'shown_connected':value,'support_path':path})
        return edges,{'kind':'compare_explicit_design_requirements_with_drawn_paths','checks':checks}
    if op=='measurement_plan':
        return edges,{'kind':'candidate_fault_continuity_table','predicted_measurements':test_vectors(g,inp),
                      'selection_rule':'smallest_target_subset_preserving_all_observable_candidate_partitions'}
    if op=='repair_check':
        return edges,{'kind':'reference_and_proposed_terminal_partitions',
                      'reference_groups':partition(g),
                      'proposed_groups':partition(g,[inp['opened_wire']],inp['proposed_link']),
                      'proposed_link_is_question_hypothesis_not_original_wire':True}
    return edges,{'kind':'external_wire_connected_components','groups':execute(g,op,inp)['external_wire_groups']}
def answer_text(op,r):
    if op=='design_check':
        if r['complies_with_given_requirements']:return '当前截图满足题目列明的连接和隔离要求；未列出的设计要求不在本次核对范围。'
        return '当前截图不满足全部给定要求：'+'；'.join(x['from']+'—'+x['to']+'要求'+('连通' if x['required_connected'] else '隔离')+'，图示却'+('连通' if x['shown_connected'] else '不连通') for x in r['violations'])+'。'
    if op=='wire_partition':return '仅沿外部导线形成的端口组为：'+'；'.join('、'.join(x) for x in r['external_wire_groups'])+'。不同组之间不能借装置内部或截图外路径补接。'
    if op=='repair_check':
        fmt=lambda groups:'；'.join('、'.join(group) for group in groups)
        return ('该方案恢复了题设静态端口连接分组。' if r['restores_original_terminal_partition'] else '该方案未恢复题设静态端口连接分组，存在分组断开或误并。')+'原连接分组：'+fmt(r['reference_terminal_groups'])+'；拟方案后的分组：'+fmt(r['proposed_terminal_groups'])+'。此结果不证明动态控制功能已恢复。'
    if op=='measurement_plan':
        text='最少需要'+str(r['minimum_test_count'])+'个通断测点，可选端口为：'+('、'.join(r['minimum_test_targets']) or '无需增加测点')+'。'
        return text+('给定候选可全部区分。' if r['all_candidates_uniquely_identifiable'] else '仍存在端口通断观测等价的候选组：'+'；'.join('、'.join(x) for x in r['distinguishable_fault_groups'] if len(x)>1)+'；不能据此唯一定位。')
    raise ValueError(op)
