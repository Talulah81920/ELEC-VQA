"""Pinned download of proposed text judges; does not load a model or use a GPU."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

REPOS={"qwen32":"Qwen/Qwen2.5-32B-Instruct",
       "prometheus7":"prometheus-eval/prometheus-7b-v2.0"}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=REPOS,required=True)
    p.add_argument('--destination',required=True)
    p.add_argument('--revision',help='Use a recorded commit SHA for reproducibility')
    a=p.parse_args()
    from huggingface_hub import HfApi,snapshot_download
    dest=Path(a.destination).expanduser().resolve();repo=REPOS[a.model]
    # Require successful remote metadata retrieval; never certify an offline fallback.
    info=HfApi().model_info(repo,revision=a.revision or 'main',files_metadata=True)
    revision=info.sha
    receipt=dest/'judge_download_receipt.json'
    if receipt.exists():
        old=json.loads(receipt.read_text(encoding='utf-8'))
        if old['repo_id']!=repo or old['revision']!=revision:
            raise ValueError('Destination contains a different snapshot; use its pinned --revision or a new directory')
    dest.mkdir(parents=True,exist_ok=True)
    expected=[f for f in info.siblings if f.rfilename.endswith(('.safetensors','.json','.txt','.model','.tiktoken','.jinja'))]
    if not any(f.rfilename.endswith('.safetensors') for f in expected):raise ValueError('No safetensors files declared')
    record={'repo_id':repo,'revision':revision,'status':'downloading',
            'time_utc':datetime.now(timezone.utc).isoformat()}
    receipt.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)
    snapshot_download(repo_id=repo,revision=revision,local_dir=str(dest),
        allow_patterns=[f.rfilename for f in expected])
    for f in expected:
        local=dest/f.rfilename
        if not local.is_file() or f.size is None or local.stat().st_size!=f.size:
            raise ValueError(f'Missing or size-mismatched snapshot file: {f.rfilename}')
    record.update(status='complete_size_checked',files={f.rfilename:f.size for f in expected},
                  verification='Pinned remote inventory and file sizes; not a full cryptographic weight audit')
    receipt.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(f'Completed {len(expected)} files at {dest}',flush=True)


if __name__=='__main__':main()
