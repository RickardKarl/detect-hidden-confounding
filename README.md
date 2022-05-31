# Combining observational datasets from multiple environments to detect hidden confounding

### [Link to paper](https://arxiv.org/abs/2205.13935)

### Abstract
A common assumption in causal inference from observational data is the assumption of no hidden confounding. Yet it is, in general, impossible to verify the presence of hidden confounding factors from a single dataset. However, under the assumption of independent causal mechanisms underlying the data generative process, we demonstrate a way to detect unobserved confounders when having multiple observational datasets coming from different environments. We present a theory for testable conditional independencies that are only violated during hidden confounding and examine cases where we break its assumptions: degenerate & dependent mechanisms, and faithfulness violations. Additionally, we propose a procedure to test these independencies and study its empirical finite-sample behavior using simulation studies.

## Experiments

All our experiments can be reproduced using the .ipynb files in the /notebooks folder. Note, experiments with the continous data was run using the scripts in /R-code
