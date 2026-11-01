import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p

def load_config(path="configs/answer_only.yaml"):
    return yaml.safe_load(resolve(path).read_text(encoding="utf-8"))

def write_report(name, report):
    p = ROOT / "logs" / name
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
