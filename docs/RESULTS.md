# Current manuscript results: provisional supplementary adjudication

The latest paper and webpage use the provisional overlay in `results/assistant_review/`. Original non-null Qwen scores remain unchanged. The 585 previously unresolved ratings were assigned provisional reference-relative scores: 537 at 1, 22 at 2, and 26 at 3. Of these, 85 recover consistent numerical suggestions from original judge attempts; the remaining assignments use assistant/reference assessment. There is no independent human confirmation. The unchanged automatic-only analysis is retained in RESULTS_ORIGINAL_QWEN.md and results/scores/.

| Protocol / metric | A | E |
|---|---:|---:|
| Direct answer full credit | 74.27% | 52.42% |
| Explanation-requested answer full credit | 70.16% | 67.97% |
| Explanation-requested evidence full credit | 46.27% | 64.78% |
| Same-response joint full credit | 46.17% | 64.24% |
| Full evidence conditional on correct answer | 65.81% | 94.51% |

The joint E-minus-A difference is 18.07 points, with seven-drawing cluster-bootstrap interval [16.04, 20.44]. This interval does not include judge, reference, provisional-rating or training-seed uncertainty. Direct answer accuracy remains stronger for A. Benefits for E are selective and do not eliminate state-invariance and visual-topology failures.

The original-only joint lower estimates were 46.08% for A and 63.76% for E; original unresolved upper bounds were 63.18% and 64.78%. Thus the main joint ordering also holds within those missing-rating bounds, independently of the supplementary assignments. This does not validate the judge or the reference annotations.

Metric precedents: [e-ViL](https://arxiv.org/abs/2105.03761), [GQA](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html), [LAVE](https://ojs.aaai.org/index.php/AAAI/article/view/28212), and [ERASER](https://aclanthology.org/2020.acl-main.408/). No causal sufficiency/comprehensiveness metric is inferred from static outputs.

