# Detecting hidden confounding in observational data using multiple environments

### [Link to paper](https://arxiv.org/abs/2205.13935)

### Python package

I have also included the method from this paper into an easy-to-use Python package: **[causal-falsify](https://github.com/RickardKarl/causal-falsify)**, please check it out.

### Abstract

A common assumption in causal inference from observational data is that there is no hidden confounding. Yet it is, in general, impossible to verify the presence of hidden confounding factors from a single dataset. Under the assumption of independent causal mechanisms underlying the data generating process, we demonstrate a way to detect unobserved confounders when having multiple observational datasets coming from different environments. We present a theory for testable conditional independencies that are only absent during hidden confounding and examine cases where we violate its assumptions: degenerate & dependent mechanisms, and faithfulness violations. Additionally, we propose a procedure to test these independencies and study its empirical finite-sample behavior using simulation studies and semi-synthetic data based on a real-world dataset. In most cases, our theory correctly predicts the presence of hidden confounding, particularly when the confounding bias is large.

## Experiments

All our experiments can be reproduced using the .ipynb files in the /notebooks folder, except for the experiment from Example 1 which was run with the .R script in the same folder.

See requirements.txt file for required Python packages. The experiments were run in Python 3.10.
