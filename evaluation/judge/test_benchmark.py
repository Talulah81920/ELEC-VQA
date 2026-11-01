"""CPU-only regression tests. Fake judge is a fixture, never model evidence."""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import benchmark as b


class BenchmarkTests(unittest.TestCase):
    def test_grade_contract(self):
        good={'rating':3,'needs_review':False,'reason':'same decision','answer_span':'No.'}
        self.assertEqual(b.parse_grade('```json\n'+json.dumps(good)+'\n```')['rating'],3)
        for invalid in [dict(good,rating=True),dict(good,needs_review='false'),dict(good,rating=5),dict(good,reason='')]:
            with self.assertRaises(ValueError):b.parse_grade(json.dumps(invalid))
        with self.assertRaises(ValueError):b.parse_grade(json.dumps(dict(good,rating=2)),False)
        self.assertIsNone(b.parse_grade(json.dumps(dict(good,needs_review=True)))['rating'])

    def test_prometheus_not_mapped_to_lave(self):
        self.assertEqual(b.numeric({'judge_kind':'prometheus','grade':{'rating':5}}),(1,None))
        self.assertEqual(b.numeric({'judge_kind':'prometheus','grade':{'rating':4}}),(0,None))

    def test_workflow_with_fixture_only(self):
        data=b.path(b.DATA_DEFAULT)
        if not data.exists():self.skipTest('Local R4 dataset not present')
        with tempfile.TemporaryDirectory(prefix='elec_vqa_cpu_test_') as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);panel=root/'panel'
            b.prepare(argparse.Namespace(data_root=str(data),split='validation',limit=30,output=str(panel)))
            meta,_,rows=b.panel(panel)
            self.assertEqual(len(rows),30)
            self.assertEqual(set(meta['tasks'].values()),{2})
            dev=b.readj(panel/'calibration_dev_ids.json');hold=b.readj(panel/'calibration_holdout_ids.json')
            lookup={r['id']:r for r in rows}
            self.assertFalse({lookup[k]['base_circuit_id'] for k in dev}&{lookup[k]['base_circuit_id'] for k in hold})
            graph={g['circuit_id']:g for g in b.readl(data/'circuits/circuit_cases.jsonl')}
            files=[]
            for name in ['base','A','E']:
                out=root/name;out.mkdir();preds=[]
                for r in rows:
                    preds.append({'id':r['id'],'question':r['question'],'model_question':r['question']+'\nGive a brief answer in English. Do not include an explanation.',
                        'image_used':r['image'],'image_sha256':graph[r['circuit_id']]['image_sha256'],
                        'output_text':r['answer'],'generation_limit_hit':False})
                file=out/'predictions.jsonl';file.write_text(''.join(json.dumps(p)+'\n' for p in preds),encoding='utf-8')
                b.writej(out/'run.json',{'master_sha256':meta['master_sha256'],'split':'validation','prompt_mode':'direct','max_new_tokens':768})
                files.append(name+'='+str(file))
            judge_dir=root/'fixture_not_a_model';judge_dir.mkdir();(judge_dir/'fixture.safetensors').write_bytes(b'NOT REAL MODEL WEIGHTS')
            args=argparse.Namespace(panel=str(panel),predictions=files,output=str(root/'judged'),
                model_path=str(judge_dir),judge_kind='qwen',dimension='answer',only_ids=None,
                max_input_tokens=12288,max_new_tokens=512,check_only=False,allow_unverified_inputs=False)
            # This fixture echoes known inputs solely to test I/O, quote verification and resume.
            calls=[]
            class FixtureJudge:
                def __init__(self,*a):pass
                def generate(self,system,user,retry=False):
                    calls.append(1);packet=json.loads(user)
                    return json.dumps({'rating':3,'needs_review':False,'reason':'CPU fixture only',
                        'answer_span':packet['candidate_response']}),{'generation_limit_hit':False,'input_tokens':1,'generated_tokens':1}
            original=b.LocalJudge;b.LocalJudge=FixtureJudge
            try:
                b.judge(args);self.assertEqual(len(calls),90)
                b.judge(args);self.assertEqual(len(calls),90)  # no re-inference on resume
                summary=root/'summary';b.summarize(argparse.Namespace(scores=[str(root/'judged/scores.jsonl')],output=str(summary)))
                report=b.readj(summary/'summary.json')
                self.assertFalse(report['full_split'])
                self.assertEqual({v['n'] for v in report['results']},{28})
                self.assertTrue((root/'judged/review.html').is_file())
                # New prediction content must never resume into old judge results.
                file=root/'base/predictions.jsonl';pr=b.readl(file);pr[0]['output_text']='changed fixture'
                file.write_text(''.join(json.dumps(p)+'\n' for p in pr),encoding='utf-8')
                with self.assertRaises(ValueError):b.judge(args)
            finally:b.LocalJudge=original


if __name__=='__main__':unittest.main()
