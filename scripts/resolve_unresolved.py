"""Provisional, auditable resolution of originally null test judgments.

Original Qwen records are immutable. This produces a separate mixed-evaluator
analysis and a human-review queue; it does not claim expert adjudication.
"""
from pathlib import Path
import argparse,collections,csv,copy,hashlib,json,re

R=Path(__file__).resolve().parents[1]
O=R/'results/assistant_review'
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def writecsv(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def norm(s):return re.sub(r'[^\w]+','',s.casefold())
def raw_grades(r):
 out=[]
 for a in r.get('attempts',[]):
  try:x=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',a.get('raw','').strip()))
  except (ValueError,TypeError):continue
  if isinstance(x,dict) and type(x.get('rating')) is int and x['rating'] in [1,2,3]:out.append(x)
 return out

def resolve(r,tr,dim,m,p):
 sid=r['id'];model=r['model_key'];text=p['output_text'];raw=raw_grades(r)
 if dim=='answer':
  if norm(text)==norm(m['answer']):return 3,'与参考答案语义及范围一致；不能仅因未重复单个目标端子而扣分。','reference_match'
  if tr=='direct':
   if model=='llava' and sid=='E4QA_ef37c9ba05ecd4a313cb':return 3,'问题仅列一个目标端子，Yes 明确表示该端子失去连接，与参考结论一致。','assistant_semantic'
   if model=='E' and sid=='E4QA_195aa5d66f9a5fb0eeeb':return 3,'最终答案明确给出参考中的唯一故障。附带证据自相矛盾，另列证据问题，答案暂按最终结论计分；请重点复核。','assistant_semantic'
   return 1,'未给出正确的完整目标分组、接点顺序或状态结论；错误合并/拆分及反复无效序列不属于有效部分答案。','assistant_semantic'
  if model=='E' and sid=='E4QA_14365806c509a9447bac':return 3,'最终故障答案与参考一致；候选故障证据存在错误，答案和证据分别计分，请重点复核。','assistant_semantic'
  if model=='base':
   if sid in ['E4QA_022cb36d23de0f2ef8aa','E4QA_c34240fbcb44e3457810','E4QA_ca94dd71d3364dbda5c2']:
    return 3,'最终答案与参考的故障/二元结论一致；不因解释冗余或引文抽取校验失败扣答案分。','assistant_semantic'
   partial={
    'E4QA_156982e1881bcbd379b4':'提出包含有效最小测点的冗余测点方案，答案未达到最少测试数，暂给部分分。',
    'E4QA_815f2a71d4e1769fb241':'最终答案列出正确受影响端子的非空子集但遗漏 1ZJ:24，未在最终集合加入错误端子。',
    'E4QA_61e99e16b810f7f2209d':'明确识别 1DK:4 与 CZ:4 的正确线网关系，但未完整给出全部端子分组。',
    'E4QA_8398adb1e23f261bbfce':'包含有效测试点 2ZJ:24，但增加冗余测点，未达到最少测试数。',
   }
   if sid in partial:return 2,partial[sid],'assistant_semantic'
  return 1,'最终答案包含错误接点/分组/测试参考点，或未完成作答；不是仅缺格式。错误追加故障候选、错误分组和不可区分的测试点不计部分正确。','assistant_semantic'
 # Evidence: valid support is required even when the answer itself is correct.
 if norm(m['explanation']) and norm(m['explanation']) in norm(text):
  return 3,'显式证据完整包含发布参考说明，状态、路径或故障推断一致；不要求额外列出未标注中间结点。此为参考相对判定，仍待人工核验。','reference_evidence_match'
 if model=='E':
  incomplete=['E4QA_533af9ccdaa359fa4e84','E4QA_38298919ad1effb07f65','E4QA_7003d0c146383b8d190d','E4QA_f2934a25e0f2051e77a0','E4QA_8b38db496742f885029f','E4QA_6d059fa4e344acc50872','E4QA_98bf01c2957e7e17cbe7']
  if sid in incomplete:return 2,'给出了有效路径片段和相关状态/干预，但省略另一个具名触点的必要支持。暂视为不充分而非充分证据；是否构成错误直接连接请人工复核。','assistant_evidence'
  return 1,'存在与参考相反的初始状态、不可达结论或失效路径等核心证据错误；正确最终答案不能补足错误证据。','assistant_evidence'
 if model=='A':
  full=['E4QA_be575b2a925dc887015e','E4QA_dd775013cf246fffc882']
  if sid in full:return 3,'题中无触点状态条件，参考为两端直接线连；回答明确指出直接连接，已经提供该简单问题所需支持。','assistant_evidence'
  partial=['E4QA_735df5bc2c402217db86','E4QA_3cbcbf7345bab3f9d255','E4QA_6d059fa4e344acc50872','E4QA_7ada914f7df090d18d09','E4QA_2df5b5f41795c8b934db','E4QA_2d366d56e0843bc90cb6','E4QA_149e5f255c9b02e5f6df','E4QA_1a1d45ddf972eec3e19b']
  if sid in partial:return 2,'提到了相关闭合触点或旁路作用，提供少量有效支持，但未解释完整的连接条件/路径。','assistant_evidence'
  return 1,'只给结论、泛称“按题设条件”或重述待判断关系，未给可核验的具体依据；若包含错误直接连线则同样不通过。此项不影响单独答案评分。','assistant_evidence'
 if model=='base':
  if sid=='E4QA_022cb36d23de0f2ef8aa':return 3,'由 KA2:23–24 实测连通排除该接点故障，并结合单故障假设推出 QE 故障；核心推断充分。','assistant_evidence'
  if sid in ['E4QA_c34240fbcb44e3457810','E4QA_c703060205799d3d07fd','E4QA_92182af5db8224d4eab8','E4QA_815f2a71d4e1769fb241','E4QA_10f42b1b46d84c663d13','E4QA_a9948cd07b37fe83273f','E4QA_a7448f1705dc98bc6d64','E4QA_6da5856d475e00ea44b0','E4QA_69d567e7d5c419d2678b']:
   return 1,'证据含虚构观测、错误线网/状态或改变测试参考点的核心错误，按原证据量表不能给予部分分。','assistant_evidence'
  if raw and len({g['rating'] for g in raw})==1:
   v=raw[-1]['rating']
   if v in [1,2]:return v,'暂采纳原裁判尝试中一致的数字建议，并保留全部原始理由供审理；未进行独立视觉专家核验。','raw_candidate_recovery'
  return 1,'未完成有效证据，或错误/循环路径不能支持目标结论。','assistant_evidence'
 raise ValueError((model,sid))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--human-labels',type=Path);a=ap.parse_args()
 master={r['id']:r for r in read(R/'dataset/annotations/qa_master.jsonl')};pred={}
 for tr in ['direct','explain']:
  for f in (R/f'results/predictions/{tr}').glob('*/predictions.jsonl'):pred[tr,f.parent.name]={r['id']:r for r in read(f)}
 human={}
 if a.human_labels:
  data=json.loads(a.human_labels.read_text(encoding='utf-8'));entries=data['reviews'] if isinstance(data,dict) else data
  for x in entries:
   if x.get('human_rating') in [1,2,3]:human[x['key']]=x
 queue=[];old={};new={};inputs={}
 for tr,dim in [('direct','answer'),('explain','answer'),('explain','evidence')]:
  src=R/f'results/scores/{tr}/{dim}/scores.jsonl';inputs[str(src.relative_to(R))]=hashlib.sha256(src.read_bytes()).hexdigest();original=read(src);adjusted=[]
  for record in original:
   r=copy.deepcopy(record);key=f'{tr}/{dim}/{r["model_key"]}/{r["id"]}';m=master[r['id']]
   if r['grade']['rating'] is None:
    p=pred[tr,r['model_key']][r['id']];v,reason,method=resolve(r,tr,dim,m,p)
    item={'key':key,'id':r['id'],'track':tr,'dimension':dim,'model':r['model_key'],'task':r['task'],'image':m['image'],'question':m['question'],'reference_answer':m['answer'],'reference_evidence':m['explanation'],'response':p['output_text'],'original_grade':r['grade'],'raw_candidates':raw_grades(r),'provisional_rating':v,'provisional_reason':reason,'method':method,'human_rating':None,'human_note':'','prediction_sha256':r['prediction_sha256']}
    if key in human:
     x=human[key]
     if x.get('prediction_sha256') and x['prediction_sha256']!=r['prediction_sha256']:raise ValueError('Human response hash mismatch: '+key)
     item['human_rating']=x['human_rating'];item['human_note']=x.get('human_note','')
    queue.append(item);r['original_grade']=r['grade'];r['grade']={'rating':item['human_rating'] or v,'needs_review':False,'reason':item['human_note'] if item['human_rating'] else reason}
    r['resolution']={'status':'human_confirmed' if item['human_rating'] else 'provisional_pending_human','source':method,'key':key};r['judge_kind']='qwen_with_assistant_resolution'
   adjusted.append(r)
  out=O/f'scores/{tr}/{dim}/scores.jsonl';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in adjusted),encoding='utf-8')
  old[tr,dim]=original;new[tr,dim]=adjusted
 summary=[]
 for (tr,dim),rs in new.items():
  for model in sorted({r['model_key'] for r in rs}):
   x=[r for r in rs if r['model_key']==model and not r['task'].startswith('T14')];prior=[r for r in old[tr,dim] if r['model_key']==model and not r['task'].startswith('T14')]
   summary.append({'track':tr,'dimension':dim,'model':model,'n':len(x),'original_full':sum(r['grade']['rating']==3 for r in prior),'original_unresolved':sum(r['grade']['rating'] is None for r in prior),'provisional_full':sum(r['grade']['rating']==3 for r in x),'provisional_partial':sum(r['grade']['rating']==2 for r in x),'full_credit_rate':sum(r['grade']['rating']==3 for r in x)/len(x),'semantic_score':sum((r['grade']['rating']-1)/2 for r in x)/len(x)})
 writecsv(O/'tables/overall.csv',summary)
 for axis in ['task','drawing_id','origin']:
  tab=[]
  for (tr,dim),rs in new.items():
   groups=collections.defaultdict(list)
   for r in rs:groups[r['model_key'],r.get(axis,master[r['id']].get(axis,'unknown'))].append(r)
   for (model,k),x in groups.items():tab.append({'track':tr,'dimension':dim,'model':model,axis:k,'n':len(x),'full':sum(r['grade']['rating']==3 for r in x),'full_credit_rate':sum(r['grade']['rating']==3 for r in x)/len(x)})
  writecsv(O/f'tables/by_{axis}.csv',tab)
 write(O/'review_queue.json',{'status':'provisional_pending_human','scale':{'1':'incorrect / absent evidence','2':'valid but incomplete where allowed','3':'fully correct / sufficient evidence'},'reviews':queue})
 write(O/'provenance.json',{'original_inputs_sha256':inputs,'unresolved_records':len(queue),'unique_qa':len({r['id'] for r in queue}),'unique_responses':len({(r['track'],r['model'],r['id']) for r in queue}),'classification_methods':dict(collections.Counter(r['method'] for r in queue)),'ratings':dict(collections.Counter(r['provisional_rating'] for r in queue)),'human_confirmed':sum(bool(r['human_rating']) for r in queue)})
 writecsv(O/'review_queue.csv',[{k:r[k] for k in ['key','id','track','dimension','model','task','provisional_rating','provisional_reason','human_rating','human_note','prediction_sha256']} for r in queue])
 print(json.dumps(json.loads((O/'provenance.json').read_text(encoding='utf-8')),ensure_ascii=True,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
