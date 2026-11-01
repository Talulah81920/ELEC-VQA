"""Validate public split alignment and image availability without GPU dependencies."""
from pathlib import Path
import collections,json
ROOT=Path(__file__).resolve().parents[1]
def read(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def main():
 data=ROOT/'dataset';rows=read(data/'annotations/qa_master.jsonl')
 assert len({r['id'] for r in rows})==len(rows),'Duplicate QA identifiers'
 drawings=collections.defaultdict(set)
 for r in rows:
  image=(data/r['image']).resolve();assert image.is_relative_to(data.resolve()) and image.is_file(),r['id']
  drawings[r['drawing_id']].add(r['split'])
 assert all(len(v)==1 for v in drawings.values()),'Drawing overlap between splits'
 for split in ['train','validation','test']:
  expected={r['id'] for r in rows if r['split']==split}
  for view in ['answer_only','evidence_answer']:
   actual=read(data/f'training/{split}_{view}.jsonl')
   assert len(actual)==len(expected) and {r['id'] for r in actual}==expected,(split,view)
 print(json.dumps({'qa':len(rows),'images':len({r['image'] for r in rows}),'drawings':len(drawings),'splits':dict(collections.Counter(r['split'] for r in rows)),'aligned_views':True,'source_disjoint':True},indent=2))
if __name__=='__main__':main()
