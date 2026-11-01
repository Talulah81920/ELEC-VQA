# GitHub release

Intended repository: `Talulah81920/ELEC-VQA` (public).

Expected project-page address after a successful Pages deployment:
`https://talulah81920.github.io/ELEC-VQA/`.
The expected address is not a claim that deployment has completed.

## Included

The full dataset, original project code, evaluation outputs and the manuscript
are included. Dataset: CC BY 4.0; original project code: MIT. Publisher templates,
fonts and model outputs retain their applicable terms. Original DWG files and
model weights are not included.

Every release file is below 100 MiB. This release uses ordinary Git instead of
Git LFS. Checksums preserve the release bytes across Windows and Linux.

## Project webpage

The source is `website/dist/`. `.github/workflows/pages.yml` validates the static
interactions and deploys that directory to GitHub Pages after a main-branch
website update. If Pages has not yet been enabled, select **Settings → Pages →
Build and deployment → Source → GitHub Actions** in this repository, then run
the workflow. No long-lived token is stored in the workflow.

## Scientific status

The current manuscript and webpage include provisional reference-relative
supplementary scores for 585 originally null judgments. These are pending human
adjudication. Original non-null Qwen records remain unchanged, and the original
automatic-only sensitivity analysis is retained. Publication of the repository
does not imply independent expert validation of the annotations or scores.
