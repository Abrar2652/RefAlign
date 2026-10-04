# RefAlign

Failure attribution for multi-agent LLM runs when at least one successful run of the same task is available. RefAlign selects a successful reference, aligns the failing run to it along a count-consistent action backbone, and returns the earliest divergence in action or content; no LLM call is made at attribution time. The repository also contains the controlled benchmarks, the human-labeled natural failures, the ALFWorld replay evaluation, every result file, and the scripts that regenerate each number. Some stored result files keep the method's earlier working name, PhyloAgent, in their keys.

## Setup

```
pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q
```

## Reproducing

All commands, seeds and model identifiers are listed in `results/revision/REPRODUCE.md`. LLM experiments read an OpenAI-compatible API key from a credentials file outside the repository; the ALFWorld experiments need a separate environment, described in the same file.

## Layout

`refalign/` method and evaluation code. `experiments/` benchmark generators, runners and analysis. `annotation/` natural-failure benchmark, labels and tools. `results/` result files and reports. `data_real/` public Who&When logs. `tests/` test suite.
