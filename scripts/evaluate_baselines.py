"""Single-GPU, offline InternVL2.5/LLaVA-OneVision inference for ELEC-VQA.

Only image and question reach the model. Diagnostic text scores preserve punctuation.
InternVL custom code is loaded from the explicitly pinned local official snapshot.
"""
import argparse,collections,hashlib,importlib.metadata,json,os,platform,re,sys,time,unicodedata
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPOS={'internvl':'OpenGVLab/InternVL2_5-8B','llava':'llava-hf/llava-onevision-qwen2-7b-ov-hf'}
DEFAULT_PATHS={'internvl':'models/baselines/InternVL2_5-8B','llava':'models/baselines/LLaVA-OneVision-Qwen2-7B'}
PROMPTS={
 'native':'',
 'direct':'\nGive a brief answer in English. Do not include an explanation.',
 'explain':'\nExplain the relevant visible components, connections and stated conditions. End with a separate line beginning Answer: followed by a brief answer in English.'}
BINARY={'T01_conditional_connection':'connected','T06_hypothetical_bypass':'after_connected',
 'T09_design_check':'compliant','T10_repair_check':'restores_static_connections'}

def resolve(value):
    p=Path(value).expanduser()
    return (p if p.is_absolute() else ROOT/p).resolve()

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readl(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
def writej(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def normalized(text):return ' '.join(unicodedata.normalize('NFKC',text).lower().split())
def short_answer(text):
    markers=list(re.finditer(r'(?im)^\s*(?:final\s+)?answer\s*:',text))
    return text[markers[-1].end():].strip() if markers else text.strip()
def prefix_bool(text):
    m=re.match(r'^\s*(yes|no)\b',text,re.I)
    return None if not m else m.group(1).lower()=='yes'

def load_selection(root,split,ids_file=None,limit=None):
    master=root/'annotations/qa_master.jsonl'
    rows=[r for r in readl(master) if r['split']==split]
    if ids_file:
        ids=json.loads(Path(ids_file).read_text(encoding='utf-8'))
        if not isinstance(ids,list) or len(ids)!=len(set(ids)):raise ValueError('IDs must be a unique JSON list')
        lookup={r['id']:r for r in rows}
        if not set(ids)<=set(lookup):raise ValueError('IDs outside selected split')
        rows=[lookup[i] for i in ids]
    if limit is not None:rows=rows[:limit]
    if not rows:raise ValueError('Empty selection')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate QA IDs')
    for r in rows:
        image=(root/r['image']).resolve()
        if not image.is_relative_to(root.resolve()) or not image.is_file():raise FileNotFoundError(image)
    return master,rows

def resume_rows(path,expected):
    if not path.exists():return []
    data=path.read_bytes()
    if data and not data.endswith(b'\n'):
        raise ValueError('Prediction file ends in a partial line. Preserve it and inspect/repair the final line before --resume.')
    rows=readl(path)
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)) or ids!=expected[:len(ids)]:raise ValueError('Resume IDs are duplicated, out of order, or outside this run')
    return rows

def tile_layout(width,height,max_tiles=12,size=448):
    # Standard native aspect-ratio grid selection (InternVL model card, MIT).
    ratios=sorted({(x,y) for x in range(1,max_tiles+1) for y in range(1,max_tiles+1)
                   if x*y<=max_tiles},key=lambda v:(v[0]*v[1],v[0],v[1]))
    best=(1,1);distance=float('inf')
    for x,y in ratios:
        d=abs(width/height-x/y)
        if d<distance or (d==distance and width*height>0.5*size*size*x*y):
            best=(x,y);distance=d
    return best

def internvl_crops(image,max_tiles=12,size=448):
    nx,ny=tile_layout(*image.size,max_tiles,size)
    resized=image.resize((nx*size,ny*size))
    crops=[resized.crop((x*size,y*size,(x+1)*size,(y+1)*size)) for y in range(ny) for x in range(nx)]
    if len(crops)>1:crops.append(image.resize((size,size)))
    return crops

def diagnostics(rows,predictions):
    lookup={r['id']:r for r in rows};group=collections.defaultdict(list)
    for p in predictions:
        r=lookup[p['id']];a=short_answer(p['output_text']);field=BINARY.get(r['task'])
        item={'strict_text_equal':normalized(a)==normalized(r['answer']),
              'gold':r['result'][field] if field else None,'parsed':prefix_bool(a)}
        group['overall'].append(item);group[r['task']].append(item)
    result={}
    for key,items in group.items():
        binary=[x for x in items if x['gold'] is not None]
        result[key]={'count':len(items),'punctuation_preserving_text_EM':sum(x['strict_text_equal'] for x in items)/len(items),
          'binary_evaluated_samples':len(binary),
          'strict_prefix_binary_parse_rate':sum(x['parsed'] is not None for x in binary)/len(binary) if binary else None,
          'strict_prefix_binary_accuracy_including_parse_failures':sum(x['parsed']==x['gold'] for x in binary)/len(binary) if binary else None}
    return {'metrics':result,'expected_samples':len(rows),'complete':len(predictions)==len(rows),
      'status':'automatic_rule_labels_not_expert_gold',
      'scope':'DIAGNOSTIC ONLY: prefix parser can reject valid prose. Text EM preserves group punctuation but is not semantic scoring. Not directly comparable to legacy punctuation-stripping EM.',
      'explanations_scored':False,'scorer_version':'baseline_diagnostics_20261027',
      'generation_limit_known_samples':sum(p.get('generation_limit_hit') is not None for p in predictions),
      'generation_limit_hit_count':sum(p.get('generation_limit_hit') is True for p in predictions)}

class InternVL:
    def __init__(self,path,tiles):
        import torch
        import torchvision.transforms as T
        from torchvision.transforms import InterpolationMode
        from transformers import AutoModel,AutoTokenizer
        self.torch=torch;self.tiles=tiles
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),trust_remote_code=True,use_fast=False,local_files_only=True)
        self.model=AutoModel.from_pretrained(str(path),trust_remote_code=True,local_files_only=True,
          torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=False).eval().to('cuda:0')
        self.transform=T.Compose([T.Resize((448,448),interpolation=InterpolationMode.BICUBIC),T.ToTensor(),
                                  T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    def predict(self,image,question,max_tokens):
        torch=self.torch
        crops=internvl_crops(image,self.tiles)
        pixels=torch.stack([self.transform(c) for c in crops]).to(device='cuda:0',dtype=torch.bfloat16)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            text=self.model.chat(self.tokenizer,pixels,'<image>\n'+question,
                                 {'max_new_tokens':max_tokens,'do_sample':False,'num_beams':1})
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        # The official chat API returns decoded text, not original generated token IDs.
        return {'output_text':text,'seconds':elapsed,'generated_tokens':None,'generation_limit_hit':None,
          'retokenized_output_tokens':len(self.tokenizer.encode(text,add_special_tokens=False)),
          'preprocessing':{'native_tile_size':448,'tiles_including_thumbnail':len(crops),
            'processed_tile_pixels':len(crops)*448*448,'thumbnail':len(crops)>1},
          'token_count_note':'retokenized text length only; actual generated tokens and truncation unavailable from native chat API'}

def load_llava_processor(path):
    """Load each modality independently, preserving local checkpoint settings."""
    from transformers import (AutoTokenizer,LlavaOnevisionImageProcessor,
        LlavaOnevisionVideoProcessor,LlavaOnevisionProcessor)
    path=Path(path)
    tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True,use_fast=True)
    if not tokenizer.is_fast:raise RuntimeError('Expected fast Qwen2 tokenizer')
    image_processor=LlavaOnevisionImageProcessor.from_pretrained(str(path),local_files_only=True)
    video_path=path/'video_processor' if (path/'video_processor/preprocessor_config.json').is_file() else path
    video_processor=LlavaOnevisionVideoProcessor.from_pretrained(str(video_path),local_files_only=True)
    params,kwargs=LlavaOnevisionProcessor.get_processor_dict(str(path),local_files_only=True)
    processor=LlavaOnevisionProcessor.from_args_and_dict([image_processor,tokenizer,video_processor],params,**kwargs)
    cfg=json.loads((path/'config.json').read_text(encoding='utf-8'))
    if processor.num_image_tokens is None:
        count=cfg.get('image_seq_length')
        if count is None:
            vision=cfg['vision_config'];side=vision['image_size'];patch=vision['patch_size']
            if not isinstance(side,int) or not isinstance(patch,int) or patch<=0:raise ValueError('Invalid vision grid configuration')
            count=(side//patch)**2
        processor.num_image_tokens=count
    if not isinstance(processor.num_image_tokens,int) or processor.num_image_tokens<=0:raise ValueError('Invalid image token count')
    if processor.vision_feature_select_strategy is None:
        processor.vision_feature_select_strategy=cfg.get('vision_feature_select_strategy')
    if processor.vision_feature_select_strategy not in ('full','default'):raise ValueError('Missing/unsupported vision feature strategy')
    if cfg.get('vision_feature_select_strategy') and processor.vision_feature_select_strategy!=cfg['vision_feature_select_strategy']:
        raise ValueError('Processor/model vision feature strategy mismatch')
    if not processor.chat_template:
        processor.chat_template=tokenizer.chat_template
    if not processor.chat_template and (path/'chat_template.jinja').is_file():
        processor.chat_template=(path/'chat_template.jinja').read_text(encoding='utf-8')
    if not processor.chat_template:raise ValueError('Checkpoint chat template missing; do not substitute an invented template')
    return processor


class Llava:
    def __init__(self,path):
        import torch
        from transformers import LlavaOnevisionForConditionalGeneration
        self.torch=torch
        self.processor=load_llava_processor(path)
        self.processor.tokenizer.padding_side='left'
        self.model=LlavaOnevisionForConditionalGeneration.from_pretrained(str(path),local_files_only=True,
          torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa').eval().to('cuda:0')
    def predict(self,image,question,max_tokens):
        torch=self.torch
        turns=[{'role':'user','content':[{'type':'image'},{'type':'text','text':question}]}]
        prompt=self.processor.apply_chat_template(turns,add_generation_prompt=True,tokenize=False)
        batch=self.processor(images=image,text=prompt,return_tensors='pt')
        batch={k:(v.to('cuda:0',dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda:0'))
               if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        n_input=batch['input_ids'].shape[-1]
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            output=self.model.generate(**batch,max_new_tokens=max_tokens,do_sample=False,num_beams=1)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        tokens=output[0,n_input:];text=self.processor.decode(tokens,skip_special_tokens=True)
        eos=self.model.generation_config.eos_token_id
        eos_ids=set(eos if isinstance(eos,list) else [eos])
        return {'output_text':text,'seconds':elapsed,'generated_tokens':len(tokens),
          'generation_limit_hit':bool(len(tokens)>=max_tokens and (not len(tokens) or int(tokens[-1]) not in eos_ids)),
          'retokenized_output_tokens':len(self.processor.tokenizer.encode(text,add_special_tokens=False)),
          'preprocessing':{'native_processor':type(self.processor.image_processor).__name__,
             'tokenizer_class':type(self.processor.tokenizer).__name__,'tokenizer_is_fast':self.processor.tokenizer.is_fast,
             'pixel_values_shape':list(batch['pixel_values'].shape),'input_tokens':n_input,
             'image_sizes':batch['image_sizes'].tolist() if 'image_sizes' in batch else None}}

def package_versions():
    result={}
    for p in ['torch','torchvision','transformers','accelerate','timm','Pillow','huggingface-hub']:
        try:result[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:result[p]=None
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=REPOS,required=True);p.add_argument('--model-path')
    p.add_argument('--data-root',default='dataset')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--prompt-mode',choices=PROMPTS,default='native');p.add_argument('--ids-file')
    p.add_argument('--limit',type=int);p.add_argument('--max-new-tokens',type=int,default=768)
    p.add_argument('--internvl-max-tiles',type=int,default=12)
    p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--check-only',action='store_true');p.add_argument('--score-only',action='store_true')
    a=p.parse_args()
    if a.limit is not None and a.limit<1:p.error('--limit must be positive')
    if a.max_new_tokens<1 or not 1<=a.internvl_max_tiles<=36:p.error('Invalid token/tile budget')
    data=resolve(a.data_root);model_path=resolve(a.model_path or DEFAULT_PATHS[a.model])
    master,rows=load_selection(data,a.split,resolve(a.ids_file) if a.ids_file else None,a.limit)
    out=resolve(a.output);out.mkdir(parents=True,exist_ok=True)
    expected=[r['id'] for r in rows];dest=out/'predictions.jsonl'
    if a.check_only:
        print(json.dumps({'QA':len(rows),'images':len({r['image'] for r in rows}),'split':a.split,
            'model':a.model,'model_path':str(model_path),'model_downloaded':(model_path/'config.json').exists(),
            'prompt_mode':a.prompt_mode,'mode':'DATA CHECK ONLY; no model load'},ensure_ascii=False,indent=2));return
    if a.score_only:
        predictions=resume_rows(dest,expected)
        if not predictions:raise ValueError('No existing predictions')
    else:
        # Set before importing transformers or any custom model modules.
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        snapshot=model_path/'baseline_snapshot.json'
        if not snapshot.is_file():raise FileNotFoundError(f'{snapshot}: use download_elec_baselines.py to record a pinned snapshot')
        snap=json.loads(snapshot.read_text(encoding='utf-8'))
        if snap['repo_id']!=REPOS[a.model] or snap.get('status')!='complete':raise ValueError('Snapshot identity/completion mismatch')
        code_hashes={str(f.relative_to(model_path)):sha(f) for f in sorted(model_path.rglob('*.py'))}
        small_files={f.name:sha(f) for f in model_path.glob('*.json') if f.name!='baseline_snapshot.json'}
        for filename,digest in snap.get('metadata_hashes',{}).items():
            if sha(model_path/filename)!=digest:raise ValueError(f'Model metadata/code modified after download: {filename}')
        versions=package_versions()
        required={'internvl':'4.37.2','llava':'4.46.3'}[a.model]
        if versions['transformers']!=required:
            raise RuntimeError(f'Use dedicated {a.model} environment with transformers=={required}; current={versions["transformers"]}')
        contract={'schema':'elec_baselines_20261027','model':a.model,'repo_id':REPOS[a.model],
          'revision':snap['revision'],'model_path':str(model_path),'model_code_sha256':code_hashes,
          'model_metadata_sha256':small_files,'dataset_root':str(data),'master_sha256':sha(master),
          'split':a.split,'ids':expected,'prompt_mode':a.prompt_mode,'prompt_suffix':PROMPTS[a.prompt_mode],
          'image_policy':'same raw image; native family preprocessing; no extra reference crops',
          'internvl_max_tiles':a.internvl_max_tiles if a.model=='internvl' else None,
          'max_new_tokens':a.max_new_tokens,'do_sample':False,'num_beams':1,'precision':'bfloat16',
          'attention':'eager, flash-attn disabled' if a.model=='internvl' else 'sdpa',
          'software':versions,'script_sha256':sha(__file__)}
        run_path=out/'run.json'
        if run_path.exists():
            if not a.resume:raise FileExistsError('Output already has run.json; choose new output or use --resume')
            prior=json.loads(run_path.read_text(encoding='utf-8'))
            if prior['contract']!=contract:raise ValueError('Resume provenance differs; use a new output directory')
        elif dest.exists():raise ValueError('Predictions exist without run.json')
        predictions=resume_rows(dest,expected)
        if len(predictions)<len(rows):
            import torch
            from PIL import Image
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
            torch.cuda.set_device(0)
            if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16-capable GPU required')
            free,total=torch.cuda.mem_get_info()
            if free<24*2**30:raise RuntimeError(f'Only {free/2**30:.1f} GiB free; select a free H100. This is a lower bound, not a memory guarantee.')
            runtime={'python':sys.version,'platform':platform.platform(),'gpu':torch.cuda.get_device_name(0),
                'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'free_GiB_before_load':free/2**30,
                'torch_cuda':torch.version.cuda,'time_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            if not run_path.exists():writej(run_path,{'contract':contract,'runtime':runtime})
            runner=InternVL(model_path,a.internvl_max_tiles) if a.model=='internvl' else Llava(model_path)
            torch.cuda.reset_peak_memory_stats()
            with dest.open('a',encoding='utf-8',newline='\n') as f:
                for r in rows[len(predictions):]:
                    image_path=data/r['image'];question=r['question']+PROMPTS[a.prompt_mode]
                    with Image.open(image_path) as im:
                        picture=im.convert('RGB');size=list(picture.size)
                        result=runner.predict(picture,question,a.max_new_tokens)
                    item={'id':r['id'],'task':r['task'],'question':r['question'],
                      'model_question':question,'image_used':r['image'],'image_mode':'original',
                      'image_sha256':sha(image_path),'original_image_size':size,
                      'reference_answer':r['answer'],**result}
                    item['short_answer']=short_answer(item['output_text'])
                    f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush();predictions.append(item)
                    print(f'{len(predictions)}/{len(rows)}',flush=True)
            writej(out/f'resources_{time.time_ns()}.json',{'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
              'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'note':'current process only; reset after model load'})
    report=diagnostics(rows,predictions);writej(out/'metrics.json',report)
    refs={r['id']:r for r in rows}
    (out/'human_review.md').write_text('\n\n'.join(f"## {x['id']}\n\nQ: {refs[x['id']]['question']}\n\nReference: {refs[x['id']]['answer']}\n\nModel:\n\n{x['output_text']}" for x in predictions),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
