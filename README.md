# ELEC-VQA

**Improving Electrical Schematic Understanding with the ELEC-VQA Dataset: A Comparative Study**

[Project webpage](https://talulah81920.github.io/ELEC-VQA/) ·
[Dataset card](docs/DATASET_CARD.md) · [Results](docs/RESULTS.md)

ELEC-VQA is a source-traceable electrical schematic visual question answering
dataset with concise answers, compact evidence and checks over bounded terminal
graphs. Its current sources are substation secondary-circuit CAD drawings.
It supports comparisons between answer-only and evidence-guided adaptation,
alongside general-purpose vision–language baselines.

**Dataset: CC BY 4.0. Original project code: MIT.** See [license scopes](LICENSE_NOTICE.md).

## Dataset

| Item | Count |
|---|---:|
| English question–answer records | 10,201 |
| Image configurations | 1,019 |
| Original local windows | 300 |
| Controlled drawing variants | 719 |
| Source drawings / projects | 44 / 14 |
| Train / validation / test QA | 6,972 / 1,155 / 2,074 |

Original windows can overlap, and controlled variants are not additional
independent original circuit designs. Related questions and edits stay with
their source partition. Primary test accuracy uses 2,064 questions from seven
drawings and two projects; ten T14 controls are reported separately.

Two aligned training views share images, questions and stable identifiers:

- **A — answer only**: a concise final answer.
- **E — evidence + answer**: compact supporting evidence followed by the answer.

Reference annotations are automatically graph-checked, not independently
expert-validated gold. Generated explanations are observable outputs, not
observations of hidden model reasoning.

## Comparative results

The current manuscript includes provisional supplementary ratings for 585
originally unresolved judgments. None has independent human confirmation.
Original non-null Qwen2.5-32B-Instruct scores are unchanged, and the original-only
analysis is retained separately. See [current findings](docs/RESULTS.md),
[original Qwen-only findings](docs/RESULTS_ORIGINAL_QWEN.md), and the
[review queue](results/assistant_review/README.md).

| Full-credit metric | Base | A | E |
|---|---:|---:|---:|
| Direct answer | 37.16% | **74.27%** | 52.42% |
| Explanation-requested answer | 39.00% | **70.16%** | 67.97% |
| Explanation-requested evidence | 32.17% | 46.27% | **64.78%** |
| Same-response answer AND evidence | 31.93% | 46.17% | **64.24%** |

Evidence supervision improves supported output and selected multi-condition
tasks, but does not universally improve answer accuracy. State-invariance and
visual-topology-change pairs expose substantial weaknesses. The website
includes both gains and failures, plus a switch to original Qwen-only scores.
The study uses one training seed; drawing-cluster intervals do not cover judge,
reference, supplementary-rating or training-seed uncertainty.

## Download

```bash
git clone https://github.com/Talulah81920/ELEC-VQA.git
cd ELEC-VQA
```

The complete dataset is included under `dataset/`. Git LFS is not required.
Pretrained models, trained adapters and original DWG files are not included.

## Layout

| Directory | Contents |
|---|---|
| `dataset/` | Images, master QA, terminal graphs, supervision files, change pairs |
| `src/elec_vqa/` | Qwen training and data pipeline |
| `configs/` | Answer-only and evidence–answer training settings |
| `scripts/` | Prediction, validation and offline analysis |
| `evaluation/judge/` | Reference-guided judge and task rubrics |
| `results/` | Recorded predictions, original scores and separate provisional overlay |
| `paper/` | English manuscript, LaTeX source and Chinese reading version |
| `website/` | Static project page and its interaction checks |
| `docs/` | Dataset, results and release documentation |

## Ubuntu 22.04 / H100

```bash
conda create -n elec-vqa python=3.10 -y
conda activate elec-vqa
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/validate_dataset.py
python -m elec_vqa.train --config configs/answer_only.yaml --check-data
CUDA_VISIBLE_DEVICES=0 python -m elec_vqa.train --config configs/answer_only.yaml
CUDA_VISIBLE_DEVICES=0 python -m elec_vqa.train --config configs/evidence_answer.yaml
```

Place an independently downloaded Qwen2.5-VL-7B-Instruct checkpoint at
`models/Qwen2.5-VL-7B-Instruct`, or edit `model_path`. The renamed public package
has passed CPU validation and offline analysis; a fresh GPU rerun of this
packaging is not claimed.

## Recompute recorded results

```bash
# Original Qwen-only analysis
python scripts/analyze_evidence.py
python scripts/analyze_extended.py

# Separate provisional supplementary analysis
python scripts/resolve_unresolved.py
python scripts/analyze_provisional.py

# Evaluation entry point
python evaluation/judge/benchmark.py --help
```

Recomputing the provisional overlay does not convert it into a human-confirmed
evaluation. The dataset supports schematic-understanding research, not autonomous
live electrical operation.

## Project page

The static source is `website/dist/`. GitHub Pages deployment is defined in
[`.github/workflows/pages.yml`](.github/workflows/pages.yml). The page contains
real original-DWG examples, supervision-target views, recorded model responses,
interactive comparisons and resource downloads. See [website notes](website/README.md).

## Licenses and citation

- [Dataset: CC BY 4.0](dataset/LICENSE.md).
- [Original project code: MIT](LICENSE).
- Third-party templates, fonts, dependencies and models retain their own terms.

The manuscript title above is the preliminary citation; author metadata and a
persistent publication identifier will be added when available. No conference
or journal acceptance is claimed.
