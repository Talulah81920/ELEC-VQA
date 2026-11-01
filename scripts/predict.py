"""ELEC-VQA image/question inference, resumable prediction export and human review.

Uses the existing project's Qwen loader; never adds reference evidence to input.
Optional semantic scores reuse the conservative scorer, with unresolved cases.
"""
import argparse
import collections
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from elec_vqa.common import load_config,resolve


def readl(path):
    rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError(f'Duplicate IDs: {path}')
    return rows


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def select(rows,limit):
    if not limit:return rows
    if limit<1:raise ValueError('--limit must be positive')
    # Deterministic task round robin, not the first N rows of a task-sorted file.
    groups=collections.defaultdict(list)
    for r in sorted(rows,key=lambda r:sha_text(r['id'])):groups[r['task']].append(r)
    chosen=[]
    while len(chosen)<min(limit,len(rows)):
        for task in sorted(groups):
            if groups[task] and len(chosen)<limit:chosen.append(groups[task].pop())
    return chosen


def sha_text(text):return hashlib.sha256(text.encode()).hexdigest()


def export_review(out,rows,preds,data):
    lookup={r['id']:r for r in rows};esc=html.escape;cards=[];md=[]
    for p in preds:
        r=lookup[p['id']]
        try:image=os.path.relpath(data/r['image'],out).replace(os.sep,'/')
        except ValueError:image=(data/r['image']).as_uri()
        cards.append(f'<article><h2>{esc(r["id"])} · {esc(r["task"])}</h2>'
          f'<a href="{esc(image,quote=True)}"><img loading="lazy" src="{esc(image,quote=True)}"></a>'
          f'<h3>Question</h3><p>{esc(r["question"])}</p><h3>Model answer</h3>'
          f'<pre>{esc(p["output_text"])}</pre><details><summary>Reference answer and evidence</summary>'
          f'<p>{esc(r["answer"])}</p><p>{esc(r["explanation"])}</p></details></article>')
        md.append(f'## {r["id"]} · {r["task"]}\n\nQ: {r["question"]}\n\nModel:\n\n{p["output_text"]}\n\nReference: {r["answer"]}')
    (out/'human_review.html').write_text('<!doctype html><meta charset="utf-8"><title>ELEC-VQA ELEC-VQA model answers</title>'
      '<style>body{max-width:1100px;margin:24px auto;font:16px/1.6 system-ui;background:#eef2f6}article{background:white;padding:24px;margin:24px 0}img{max-width:100%;max-height:650px}pre{white-space:pre-wrap}h2{font-size:18px}</style>'
      '<h1>ELEC-VQA ELEC-VQA model answers</h1><p>Reference answers are automatically checked labels, not independent expert gold.</p>'+''.join(cards),encoding='utf-8')
    (out/'human_review.md').write_text('\n\n'.join(md),encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/answer_only.yaml')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--adapter');p.add_argument('--base-only',action='store_true')
    p.add_argument('--ids-file',help='Fixed ordered QA ID panel shared by every model')
    p.add_argument('--model-path',help='Override the local base-model path')
    p.add_argument('--data-root',help='Override ELEC-VQA dataset root')
    p.add_argument('--output',required=True);p.add_argument('--limit',type=int)
    p.add_argument('--prompt-mode',choices=['native','direct','explain'],default='direct')
    p.add_argument('--max-new-tokens',type=int,default=768)
    p.add_argument('--check-only',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--score-only',action='store_true');p.add_argument('--semantic',action='store_true')
    a=p.parse_args();cfg=load_config(a.config)
    if a.model_path:cfg['model_path']=a.model_path
    if a.data_root:cfg['image_root']=a.data_root
    data=resolve(cfg['image_root'])
    master=data/'annotations/qa_master.jsonl';allrows=readl(master)
    if not allrows:raise ValueError('Empty ELEC-VQA master annotations')
    if cfg.get('dataset_name','ELEC-VQA')!='ELEC-VQA':raise ValueError('Unexpected dataset identity')
    rows=[r for r in allrows if r['split']==a.split]
    if a.ids_file:
        ids=json.loads(resolve(a.ids_file).read_text(encoding='utf-8'))
        lookup={r['id']:r for r in rows}
        if len(ids)!=len(set(ids)) or not set(ids)<=set(lookup):raise ValueError('Invalid fixed ID panel')
        if a.limit:raise ValueError('Use either --ids-file or --limit, not both')
        rows=[lookup[i] for i in ids]
    else:rows=select(rows,a.limit)
    if not rows:raise ValueError('Empty split')
    for r in rows:
        if not (data/r['image']).is_file():raise FileNotFoundError(data/r['image'])
    print(json.dumps({'samples':len(rows),'images':len({r['image'] for r in rows}),
        'split':a.split,'tasks':dict(collections.Counter(r['task'] for r in rows))}),flush=True)
    if a.check_only:return
    if a.max_new_tokens<1:raise ValueError('--max-new-tokens must be positive')
    out=resolve(a.output);out.mkdir(parents=True,exist_ok=True);dest=out/'predictions.jsonl'
    adapter=None if a.base_only else resolve(a.adapter or str(resolve(cfg['output_dir'])/'final_adapter'))
    if not a.score_only and adapter is not None and not (adapter/'adapter_config.json').is_file():raise FileNotFoundError(adapter)
    signature={'evaluator':'elec-vqa-fixed-panels-1.0','config':cfg,'master_sha256':sha(master),
        'ids':[r['id'] for r in rows],'split':a.split,'prompt_mode':a.prompt_mode,
        'max_new_tokens':a.max_new_tokens,'base_only':a.base_only,'adapter':str(adapter) if adapter else None,
        'adapter_hashes':{f.name:sha(f) for f in adapter.iterdir() if f.name=='adapter_config.json' or f.suffix in ('.safetensors','.bin')} if adapter and adapter.exists() else {}}
    preds=readl(dest) if dest.exists() else []
    expected={r['id'] for r in rows}
    if not {r['id'] for r in preds}<=expected:raise ValueError('Predictions outside selected panel')
    if not a.score_only:
        if dest.exists() and (not a.resume or json.loads((out/'run.json').read_text(encoding='utf-8'))!=signature):
            raise ValueError('Existing output: use --resume with identical settings, or choose a new output directory')
        dump(out/'run.json',signature);dump(out/'selected_ids.json',[r['id'] for r in rows])
        done={r['id'] for r in preds}
        if len(done)<len(rows):
            import torch
            from PIL import Image
            from peft import PeftModel
            from elec_vqa.model import load_model,load_processor
            processor=load_processor(cfg);model=load_model(cfg,training=False)
            if adapter:model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
            model.eval();model.config.use_cache=True
            with dest.open('a',encoding='utf-8') as f:
                for r in rows:
                    if r['id'] in done:continue
                    q=r['question']
                    if a.prompt_mode=='direct':q+='\nGive a brief answer in English. Do not include an explanation.'
                    elif a.prompt_mode=='explain':q+='\nExplain the relevant visible components, connections and stated conditions. End with a separate line beginning Answer: followed by a brief answer in English.'
                    msgs=[{'role':'user','content':[{'type':'image'},{'type':'text','text':q}]}]
                    prompt=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
                    with Image.open(data/r['image']) as im:
                        original_size=list(im.size);batch=processor(text=[prompt],images=[im.convert('RGB')],return_tensors='pt',padding=True)
                    batch={k:v.to(model.device) if hasattr(v,'to') else v for k,v in batch.items()}
                    torch.cuda.synchronize();start=time.perf_counter()
                    with torch.inference_mode():generated=model.generate(**batch,max_new_tokens=a.max_new_tokens,do_sample=False)
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    tokens=generated[:,batch['input_ids'].shape[1]:]
                    raw=processor.batch_decode(tokens,skip_special_tokens=True)[0]
                    item={'id':r['id'],'task':r['task'],'question':r['question'],'model_question':q,
                        'image_used':r['image'],'image_sha256':sha(data/r['image']),'original_image_size':original_size,
                        'input_tokens':batch['input_ids'].shape[1],'reference_answer':r['answer'],
                        'output_text':raw,'generated_tokens':tokens.shape[1],'seconds':elapsed,
                        'generation_limit_hit':tokens.shape[1]>=a.max_new_tokens}
                    f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush();preds.append(item)
                    print(f'{len(preds)}/{len(rows)} {r["id"]}: {raw[:180]}',flush=True)
        else:print('All selected predictions already present; exporting reports.',flush=True)
    if not preds:raise ValueError('No predictions to summarize')
    export_review(out,rows,preds,data)
    report={'count':len(preds),'expected_count':len(rows),'complete':len(preds)==len(rows),
        'master_sha256':sha(master),'generation_limit_hit_count':sum(r.get('generation_limit_hit',False) for r in preds),
        'generation_seconds':sum(r.get('seconds',0) for r in preds),
        'note':'Inference completion is not answer accuracy. No natural-language exact match used as semantic accuracy.'}
    if a.semantic:
        from elec_vqa.semantic import VERSION,score_one,summarize
        graphs={g['circuit_id']:g for g in readl_graphs(data/'circuits/circuit_cases.jsonl')}
        lookup={r['id']:r for r in rows}
        scores=[score_one(lookup[v['id']],v,graphs[lookup[v['id']]['circuit_id']]) for v in preds]
        (out/'semantic_scores.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in scores),encoding='utf-8')
        report['semantic_scorer']=VERSION
        report['semantic']=summarize(scores)
        primary=[s for s in scores if not s['task'].startswith('T14')]
        report['semantic_primary_14_tasks']=summarize(primary)
        report['scoring_status']='Conservative automatic scorer reused on ELEC-VQA; review unresolved cases and calibrate before paper claims. Evidence validity is not scored.'
    dump(out/'metrics.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))


def readl_graphs(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


if __name__=='__main__':main()
