# ELEC-VQA
Electrical schematic visual question answering and compact evidence-guided fine-tuning.

## Layout
- `dataset/`: images, master annotations, terminal graphs, fixed split training files and revalidated pairs.
- `src/elec_vqa/`: Qwen training and data pipeline.
- `configs/`: answer-only and evidence-answer settings.
- `scripts/`: prediction and offline analysis.
- `evaluation/judge/`: reference-guided evaluation and fixed task rubrics.
- `results/`: final held-out predictions, automatic judgments, metrics and figures.
- `paper/`: manuscript source and PDF.
- `website/dist/`: static research project page with real-data examples and interactive comparisons.
- `docs/`: data card, reproduction and release notes.

## Ubuntu 22.04 / H100
```bash
conda create -n elec-vqa python=3.10 -y
conda activate elec-vqa
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install matplotlib
python -m pip install -e .
python scripts/validate_dataset.py
python -m elec_vqa.train --config configs/answer_only.yaml --check-data
CUDA_VISIBLE_DEVICES=0 python -m elec_vqa.train --config configs/answer_only.yaml
CUDA_VISIBLE_DEVICES=0 python -m elec_vqa.train --config configs/evidence_answer.yaml
```
Place an independently downloaded Qwen2.5-VL-7B-Instruct checkpoint at `models/Qwen2.5-VL-7B-Instruct`, or edit model_path. The new package has been checked with CPU data validation and analysis; GPU reruns of the renamed package are not claimed.

## Offline results
```bash
python scripts/analyze_evidence.py
python scripts/analyze_extended.py
python evaluation/judge/benchmark.py --help
```
Scores are reference-relative automatic evaluations. Missing scores retain explicit uncertainty bounds. No full human calibration is claimed. Do not use results for autonomous live electrical operation.

The latest manuscript additionally reports the separate provisional overlay in
`results/assistant_review/`. It supplements originally null ratings without
overwriting the Qwen records. These mixed-evaluator scores are pending human
adjudication; they are not described as pure automatic or expert-validated scores.
The website switches between this overlay and the original Qwen-only lower estimates.

## Publication
The manuscript title is *Improving Electrical Schematic Understanding with the ELEC-VQA Dataset: A Comparative Study*. The dataset is licensed under CC BY 4.0 and the original project code under MIT. The owner has confirmed redistribution rights for the schematic images and annotations. Author metadata and the repository URL remain to be supplied. See LICENSE_NOTICE.md for scope and third-party exclusions. This folder is prepared locally; nothing has been uploaded.

## Licenses

Dataset: [CC BY 4.0](dataset/LICENSE.md). Original project code: [MIT](LICENSE). Third-party templates, fonts, dependencies and models retain their respective terms.

## GitHub publication

The intended public repository is `Talulah81920/ELEC-VQA`. The expected project-page URL after deployment is `https://talulah81920.github.io/ELEC-VQA/`; deployment must be confirmed separately. See [GitHub release notes](docs/GITHUB_RELEASE.md). All files in this release fit regular Git limits; Git LFS is not required.
