# Experimental findings

All figures below use final Qwen2.5-32B-Instruct automatic ratings, without manual overrides. A is answer-only adaptation; E is evidence-answer adaptation of the same 7B backbone. Primary test analysis excludes the separately reported T14 diagnostic: 2,064 questions from seven source drawings. Training uses seed 42 only.

| Metric | A | E |
|---|---:|---:|
| Direct answer full credit | 74.18% | 52.37% |
| Common explanation prompt: answer full credit | 69.91% | 67.93% |
| Common explanation prompt: evidence full credit | 46.17% | 64.29% |
| Same-response answer AND evidence full credit | 46.08% | 63.76% |
| Full evidence conditional on a full-credit answer | 65.90% | 93.87% |

The last metric has unresolved-rating upper bounds of 90.37% and 95.29%, respectively. Unresolved ratings are retained, rather than silently treated as successful or deleted. The joint-rate E-minus-A difference is 17.68 percentage points, with drawing-cluster bootstrap interval [15.17, 20.58]. Only seven clusters limit inferential precision. This interval describes sampling across these sources, not seed variation or judge validity.

## Where evidence helps, and where it fails

Under identical explanation prompting, E exceeds A by 19.39 points on the 98 questions with at least three reference contact-state conditions. Both-members-correct rates on state-flip pairs are 93.90% versus 67.07%; on visual-invariant pairs they are 86.96% versus 78.26%.

The complementary negative findings are substantial: E scores 11.34% versus A's 61.86% on state-invariant pairs, and 4.35% versus 39.13% on visual-flip pairs. Evidence supervision improves explicit supporting output and selected conditional decisions, but does not establish robust topology perception or selective use of irrelevant conditions. Pair subsets share source circuits and questions; they are descriptive, not independent trials. Complexity strata are exploratory and confounded by task composition.

## Metric precedents and boundaries

- [eViL, ICCV 2021](https://arxiv.org/abs/2105.03761): motivates separating task success from explanation quality. Our conditional-support metric uses automatic electrical ratings; it is not eViL's human evaluation.
- [GQA evaluation](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html): motivates reporting consistency beyond aggregate accuracy. Our fixed electrical change pairs require both responses to be correct.
- [LAVE, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28212): supports reference-guided semantic evaluation beyond literal matching. It does not independently validate this judge on electrical schematics.
- [ERASER, ACL 2020](https://aclanthology.org/2020.acl-main.408/): sufficiency and comprehensiveness require input intervention and new predictions. Those causal metrics cannot be recovered from the present static outputs and are not claimed.

Use `results/deep_analysis/` for conditional support, change pairs and complexity figures; `results/final_analysis/` for same-response joint metrics; and `results/analysis/` for overall, task, origin and efficiency summaries. Do not interpret generated explanations as observations of hidden reasoning.
