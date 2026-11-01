# ELEC-VQA research project page

`dist/` is a buildless, responsive website. Open `dist/index.html` for local use,
or serve `dist/` with any static host. No build step or model backend is needed.
The sample browser uses real released records; it is not live model inference.

## Hosting

Repository: https://github.com/Talulah81920/ELEC-VQA

Project page: https://talulah81920.github.io/ELEC-VQA/

The GitHub Pages workflow is `.github/workflows/pages.yml`. Relative resources
work under the repository subpath. The page uses no model backend, account
system, analytics or external JavaScript dependency.

## Content and evidence

- Eight real screenshots from original DWG windows, plus exact reference QA.
- Original A/E case responses, including their failures under direct prompting.
- Interactive comparison of answer, evidence and joint metrics.
- A switch between the supplemented results and original Qwen-only lower estimates.
- Complementary positive and negative controlled-change results.
- Downloadable paper, code, sample data, CSV and Chinese reading bundle.

The 585 supplemented ratings are explicitly provisional and not human-confirmed.
They must not be described as pure Qwen ratings or independent expert labels.
The complete dataset is included in the repository, not in the small website
sample download. The sample contains eight examples only. Author and publication
metadata remain to be supplied.

## Provenance and maintenance

Page structure was informed by Nerfies and DrawingVQA project pages; the page
code and design are original. Third-party website source was not copied.
The dataset sample is CC BY 4.0; original site code is MIT. The manuscript and
third-party paper assets retain their separate terms.

Refresh source data and download bundles with the workspace helper
`python scripts/prepare_website_assets.py` after updating the public manuscript and metrics.
Check `app.js` and `data.js` with `node --check` before publication.
Run `node check-site.mjs` to verify result switching, example navigation,
evidence disclosure, case outputs, image-viewer controls and local resource links.
These CPU-only checks do not replace browser visual validation.
