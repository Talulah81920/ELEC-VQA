"""Conservative terminal graphs; external LINE endpoints and explicit contact symbols."""
import collections
import math
import re

CONTACTS = {'CCSYM00301002', 'CCSYM00200102', 'CCSYM00200202', 'CCSYM00300602', 'CCSYM00300702'}

def key(p):
    return tuple(round(float(x), 3) for x in p[:2])

def code(block):
    for a in block.get('attributes', []):
        if a['tag'].upper() in ('CODE', 'NAME', '设备编号', '代号') and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+-]*', a['text'].strip()):
            return a['text'].strip(), a
    return None, None

def extract(data):
    blocks = {b['id']: b for b in data['blocks']}
    points = collections.defaultdict(list)
    for p in data['points']:
        parts = p['id'].split('/')
        for n in range(2, len(parts)):
            prefix = '/'.join(parts[:n])
            if prefix in blocks:
                points[prefix].append(p)
    terminals = []
    contact_terms = {}
    for b in data['blocks']:
        name, label = code(b)
        if not name or not points[b['id']]:
            continue
        chosen = []
        used = set()
        for a in b.get('attributes', []):
            if a['tag'].upper() in ('CODE', 'NAME') or not re.fullmatch(r'[A-Za-z]?\d+(?::\d+){0,3}', a['text'].strip()):
                continue
            ds = sorted((math.dist(p['position'], [a['x'], a['y']]), p['id'], p) for p in points[b['id']])
            if not ds or ds[0][0] > 9 or (len(ds) > 1 and ds[1][0]-ds[0][0] < .5):
                continue
            p = ds[0][2]
            if key(p['position']) in used:
                continue
            used.add(key(p['position']))
            row = {'id': p['id'], 'label': name+':'+a['text'].strip(), 'position': p['position'],
                   'block_id': b['id'], 'symbol': b['name'], 'code': name,
                   'labels': [label, a]}
            terminals.append(row); chosen.append(row)
        if b['name'] in CONTACTS and len(chosen) == 2 and key(chosen[0]['position']) != key(chosen[1]['position']):
            contact_terms[b['id']] = chosen
    lines = [e for e in data['lines'] if len(e['id'].split('/')) == 2 or
             (len(e['id'].split('/')) == 3 and e['id'].split('/')[-1].isdigit())]
    adj = collections.defaultdict(list)
    for e in lines:
        a, b = key(e['a']), key(e['b'])
        adj[a].append((b, e)); adj[b].append((a, e))
    nets = {}; net_points = collections.defaultdict(list)
    for p in adj:
        if p in nets:
            continue
        ni = len(net_points); nets[p] = ni; stack = [p]
        while stack:
            u = stack.pop(); net_points[ni].append(u)
            for v, _ in adj[u]:
                if v not in nets:
                    nets[v] = ni; stack.append(v)
    by_net = collections.defaultdict(list)
    by_pin = {}
    for t in terminals:
        if key(t['position']) in nets:
            t['net'] = nets[key(t['position'])]; by_net[t['net']].append(t); by_pin[t['id']] = t
    gates = []
    for bid, pair in contact_terms.items():
        if all(t['id'] in by_pin for t in pair):
            gates.append({'id': bid, 'symbol': blocks[bid]['name'], 'pins': [t['id'] for t in pair],
                          'nets': [t['net'] for t in pair]})
    net_edges=collections.defaultdict(list)
    for e in lines:
        net_edges[nets[key(e['a'])]].append(e)
    return {'adj': adj, 'nets': nets, 'net_points': net_points, 'net_edges':net_edges, 'by_net': by_net,
            'by_pin': by_pin, 'gates': gates, 'blocks': blocks}

def local_graph(g, net_ids, gates):
    """Retain all actual external segments, including cycles; merge only invisible collinear splits."""
    terminals={key(t['position']):t for ni in net_ids for t in g['by_net'][ni]}
    count=sum(len(g['by_net'][ni]) for ni in net_ids)
    if len(terminals)!=count: return None
    raw={}
    for ni in net_ids:
        for e in g['net_edges'][ni]:
            a,b=sorted([key(e['a']),key(e['b'])])
            if a==b: continue
            edge=raw.setdefault((a,b),{'a':a,'b':b,'source_edges':[]})
            edge['source_edges'].append(e['id'])
    if len(raw)>80: return None
    edges=[dict(e,kind='wire',points_cad=[list(e['a']),list(e['b'])]) for e in raw.values()]
    for e in gates:
        a,b=[key(g['by_pin'][p]['position']) for p in e['pins']]
        edges.append({'a':a,'b':b,'kind':'contact','source_block_id':e['id'],'symbol':e['symbol'],
                      'points_cad':[list(a),list(b)]})
    while True:
        adjacent=collections.defaultdict(list)
        for i,e in enumerate(edges):
            adjacent[e['a']].append(i);adjacent[e['b']].append(i)
        merged=False
        for p,idx in adjacent.items():
            if p in terminals or len(idx)!=2: continue
            e,f=[edges[i] for i in idx]
            if e['kind']!='wire' or f['kind']!='wire': continue
            a=e['b'] if e['a']==p else e['a'];b=f['b'] if f['a']==p else f['a']
            if a==b: continue
            u=[a[i]-p[i] for i in (0,1)];v=[b[i]-p[i] for i in (0,1)]
            den=math.hypot(*u)*math.hypot(*v)
            if not den or abs(u[0]*v[1]-u[1]*v[0])/den>.0001 or sum(u[i]*v[i] for i in (0,1))>=0: continue
            edges=[x for i,x in enumerate(edges) if i not in idx]+[
                {'a':a,'b':b,'kind':'wire','source_edges':e['source_edges']+f['source_edges'],
                 'points_cad':[list(a),list(b)]}]
            merged=True;break
        if not merged: break
    coords=sorted({e[k] for e in edges for k in ('a','b')})
    ids={p:f'N{i}' for i,p in enumerate(coords)};nodes=[]
    for p in coords:
        t=terminals.get(p)
        nodes.append({'id':ids[p],'label':t['label'] if t else None,'cad_xy':list(p),
                      'source_pin_id':t['id'] if t else None,'block_id':t['block_id'] if t else None,
                      'symbol':t['symbol'] if t else 'unlabelled_wire_junction',
                      'labels':t['labels'] if t else [],'is_terminal':t is not None})
    counts=collections.Counter()
    for e in edges:
        e['a'],e['b']=ids[e['a']],ids[e['b']]
        prefix='W' if e['kind']=='wire' else 'K';counts[prefix]+=1;e['id']=prefix+str(counts[prefix])
    return nodes,edges

def route(adj, start, end):
    start, end = key(start), key(end)
    queue = collections.deque([start]); prev = {start: None}
    while queue and end not in prev:
        u = queue.popleft()
        for v, edge in adj.get(u, []):
            if v not in prev:
                prev[v] = (u, edge); queue.append(v)
    if end not in prev:
        return None
    pts = [end]; edges = []; u = end
    while prev[u] is not None:
        u, edge = prev[u]; pts.append(u); edges.append(edge)
    return list(reversed(pts)), list(reversed(edges))

def reachable(graph, source, target, states=None, removed=(), possible=False):
    """Solve only the explicitly represented local graph; unknown gates are 3-valued."""
    states = states or {}; removed = set(removed)
    adj = collections.defaultdict(list)
    for edge in graph['edges']:
        if edge['id'] in removed:
            continue
        value = True if edge['kind'] == 'wire' else states.get(edge['id'])
        if value is True or (value is None and possible):
            adj[edge['a']].append((edge['b'], edge['id'])); adj[edge['b']].append((edge['a'], edge['id']))
    prev = {source: None}; queue = collections.deque([source])
    while queue and target not in prev:
        u = queue.popleft()
        for v, eid in adj[u]:
            if v not in prev:
                prev[v] = (u, eid); queue.append(v)
    if target not in prev:
        return False, []
    path = []; u = target
    while prev[u] is not None:
        u, eid = prev[u]; path.append(eid)
    return True, list(reversed(path))

def solve(graph, source, target, states=None, removed=()):
    lower, path = reachable(graph, source, target, states, removed, False)
    upper, _ = reachable(graph, source, target, states, removed, True)
    return {'reachable': True if lower else (None if upper else False), 'support_edges': path,
            'scope': 'declared_visible_local_graph_no_hidden_internal_connections'}
