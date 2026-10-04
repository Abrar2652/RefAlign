# Reproducing every new result of the revision

All commands run from the repository root with `PYTHONPATH=.`; the system Python 3.10 with numpy/scipy/statsmodels suffices unless the ALFWorld venv is named. Paths in the package are relative; `$HF_HOME` and credentials come from the environment. Outputs land in `results/revision/`.

## 0. Reproduction of the manuscript's headline numbers from saved outputs (no LLM calls)
```
python -m experiments.rev_reproduce            # 19/19 checks against the saved outputs
```

## 1. Leakage-controlled positional baselines (LOFO) on the 32 natural failures (no LLM calls)
```
python -m experiments.rev_positional_baselines # -> positional_baselines.json, used by tab_positional.tex
```

## 2. t03 answer-key correction (76) and rescoring of the saved prompt-intervention rollouts (no LLM calls)
```
python -m experiments.rev_rescore_t03          # -> intervention_rescored_{gpt-4o-mini,qwen2.5-7b}.json
```

## 3. ALFWorld external evaluation
Environment (one-time): `python3 -m venv --without-pip alf_venv && alf_venv/bin/python get-pip.py && PATH=$PWD/alf_venv/bin:$PATH alf_venv/bin/pip install alfworld openai scipy statsmodels` (the `python` symlink on PATH is needed to build fast-downward), then `ALFWORLD_DATA=<dir> alfworld-download`. Exemplars: `experiments/alfworld_react_prompts.json` (public ReAct prompt file).
```
export ALFWORLD_DATA=<dir>; V=alf_venv/bin/python
# development split (40 games x 4 rollouts), used only for pipeline development
$V -m experiments.run_alfworld_collect --split valid_seen --n-games 40 --game-seed 0 --n 4 --model gpt-4o-mini-2024-07-18 --base-url openai --workers 8 --out results/revision/alf_dev_traces.json
# test split (134 games x 8 rollouts); resumable; stopped at 600 episodes by credit exhaustion in our run
$V -m experiments.run_alfworld_collect --split valid_unseen --n-games 0 --game-seed 1 --n 8 --model gpt-4o-mini-2024-07-18 --base-url openai --workers 12 --out results/revision/alf_test_traces.json
# deterministic attribution (free) on the fully collected games
python -m experiments.run_alfworld_attribution --traces results/revision/alf_test_complete_traces.json --out results/revision/alf_test_complete_attribution.json --no-judge
python -m experiments.rev_alfworld_analysis --prefix results/revision/alf_test_complete --seeds 0 1
python -m experiments.rev_tables
# judges, replay jobs, replays, analysis, tables (as executed; resumable)
experiments/rev_alfworld_chain.sh          # or step by step: run_alfworld_attribution -> rev_alfworld_jobs -> run_alfworld_replay -> rev_alfworld_analysis
experiments/rev_replay_resume.sh           # resumes replays after an interruption
experiments/rev_positional_replays.sh      # replays the leave-one-type-out positional selections, re-analyses
```
Seeds: every LLM call carries a seed derived with crc32 from (experiment, game, rollout, step) or (replay, run, step, mode, seed index); game samples use `numpy.random.RandomState(seed)`. The pre-registration is `EXTERNAL_PREREG.md`.

## 4. Earlier follow-up experiments (information-matched judges, reference sensitivity, prompt intervention, GSM8K, BBH)
See `results/new/RESULTS_NEW.md` Section 9 and `experiments/finalize_new_results.sh`.

## 5. Tables, tests, packages
```
python -m experiments.rev_tables                      # tab_positional.tex, tab_intervention_rescored.tex, tab_alf_*.tex
python -m pytest -q                                      # all tests (existing + tests/test_new_experiments.py + tests/test_revision.py)
python -m experiments.rev_package --which overleaf repro
```
Compilation: `tectonic main.tex` and `tectonic supplement.tex` in the Overleaf ZIP directory (pdfTeX/TeX Live also compiles them; pagination differs slightly between engines, see the change report).
