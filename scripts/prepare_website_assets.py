from pathlib import Path
import csv,json,shutil,zipfile,collections,hashlib
R=Path(__file__).resolve().parents[1];W=R/'website/dist';assets=W/'assets';assets.mkdir(parents=True,exist_ok=True)
rows=[json.loads(l) for l in (R/'dataset/annotations/qa_master.jsonl').read_text(encoding='utf-8').splitlines()]
metrics=json.loads((R/'results/assistant_review/publication/metrics.json').read_text(encoding='utf-8'))
sid='E4QA_04b45d17904f4265d252';hero=next(r for r in rows if r['id']==sid)
selected=[hero];used={hero['image']}
for prefix in ['T08','T07','T05','T15','T03','T02','T11']:
 pool=[r for r in rows if r['task'].startswith(prefix) and r['origin']=='original_DWG_local_window' and r['image'] not in used]
 pool.sort(key=lambda r:(len(r['question'])+len(r['explanation']),r['id']))
 pick=pool[0];selected.append(pick);used.add(pick['image'])
examples=[]
for i,r in enumerate(selected):
 dest=f'example_{i+1:02d}.png';shutil.copy2(R/'dataset'/r['image'],assets/dest)
 examples.append({k:r[k] for k in ['id','task','question','answer','explanation','origin','split']}|{'image':'assets/'+dest,'circuit_id':r['circuit_id']})
actual={}
for tr in ['direct','explain']:
 for m in ['A','E']:
  pred=[json.loads(l) for l in (R/f'results/predictions/{tr}/{m}/predictions.jsonl').read_text(encoding='utf-8').splitlines()];actual[f'{tr}_{m}']=next(r['output_text'] for r in pred if r['id']==sid)
data={'examples':examples,'metrics':metrics,'case_outputs':actual,'original_overall':list(csv.DictReader((R/'results/analysis/overall.csv').open(encoding='utf-8-sig'))),'original_joint':list(csv.DictReader((R/'results/final_analysis/joint_overall.csv').open(encoding='utf-8-sig'))),'pairs':list(csv.DictReader((R/'results/assistant_review/deep_analysis/paired_changes.csv').open(encoding='utf-8-sig'))),'complexity':list(csv.DictReader((R/'results/assistant_review/deep_analysis/complexity.csv').open(encoding='utf-8-sig'))),'tasks':dict(collections.Counter(r['task'] for r in rows))}
(W/'data.js').write_text('window.ELEC_DATA = '+json.dumps(data,ensure_ascii=False).replace('<','\\u003c')+';\n',encoding='utf-8')
for name in ['answer_comparison','answer_evidence_joint','task_effects','conditional_support']:
 shutil.copy2(R/f'results/assistant_review/publication/{name}.png',assets/f'{name}.png')
shutil.copy2(R/'results/assistant_review/tables/overall.csv',assets/'results_supplemented.csv')
shutil.copy2(R/'results/analysis/overall.csv',assets/'results_original_qwen.csv')
shutil.copy2(R/'dataset/LICENSE.md',assets/'DATASET_LICENSE.md');shutil.copy2(R/'LICENSE',assets/'CODE_LICENSE.txt')
shutil.copy2(R/'paper/ELEC-VQA.pdf',assets/'ELEC-VQA.pdf');shutil.copy2(R/'paper/manuscript_zh.md',assets/'manuscript_zh.md')
with zipfile.ZipFile(assets/'ELEC-VQA_code.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 for folder in ['src','scripts','configs','evaluation']:
  for f in (R/folder).rglob('*'):
   if f.is_file() and '__pycache__' not in f.parts:z.write(f,'ELEC-VQA/'+f.relative_to(R).as_posix())
 for name in ['README.md','requirements.txt','pyproject.toml','LICENSE','LICENSE_NOTICE.md']:z.write(R/name,'ELEC-VQA/'+name)
 z.writestr('ELEC-VQA/ARCHIVE_CONTENTS.md','# Code-only archive\n\nThis download contains the project code and configuration, not the full dataset, recorded results, model weights or manuscript. Obtain the complete release from https://github.com/Talulah81920/ELEC-VQA before running commands that require dataset/ or results/.\n')
with zipfile.ZipFile(assets/'ELEC-VQA_sample.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 z.writestr('ELEC-VQA_sample/annotations.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
 for r in selected:z.write(R/'dataset'/r['image'],'ELEC-VQA_sample/'+r['image'])
 z.write(R/'dataset/LICENSE.md','ELEC-VQA_sample/LICENSE.md')
 z.writestr('ELEC-VQA_sample/README.md','# ELEC-VQA sample\n\nEight reference examples from original DWG windows, selected for the research project page. This is a qualitative sample, not an evaluation split. Original relative image paths and stable QA IDs are retained. The full dataset comprises 10,201 QA records and 1,019 image configurations. References are graph-checked, not independently expert-validated.\n')
print('Website assets:',len(examples),'real examples;',sum(f.stat().st_size for f in W.rglob('*') if f.is_file()),'bytes')

from PIL import Image
im=Image.open(assets/'example_01.png');im.crop((0,0,4300,im.height)).save(assets/'example_01_detail.png')
with zipfile.ZipFile(assets/'ELEC-VQA_chinese_reading.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 z.write(R/'paper/manuscript_zh.md','ELEC-VQA_reading/manuscript_zh.md')
 for f in (R/'paper/assets').glob('chinese_figure_*.png'):z.write(f,'ELEC-VQA_reading/assets/'+f.name)
