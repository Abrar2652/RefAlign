#!/usr/bin/env bash
# After the test collection finishes: judge attribution -> replay jobs -> replays -> analysis -> tables.
set -euo pipefail
cd .
export ALFWORLD_DATA=/path/to/workspace/alfworld_data
export PYTHONPATH=.
V=/path/to/workspace/alf_venv/bin/python
R=results/revision
until [ -f $R/alf_test_traces_meta.json ]; do sleep 20; done
echo "[chain] collection done $(date -u +%H:%M:%S)"
python3 -m experiments.run_alfworld_attribution --traces $R/alf_test_traces.json --out $R/alf_test_attribution.json --judge gpt-4o-2024-08-06 --workers 12 --judge-seeds 0 1
echo "[chain] attribution done $(date -u +%H:%M:%S)"
python3 -m experiments.rev_alfworld_jobs --attribution $R/alf_test_attribution.json --traces $R/alf_test_traces.json --out $R/alf_test_replay_jobs.json --profile-n 40 --kmax 10 --method-subset-n 100 --methods refalign refalign_s refalign_s_first refalign_s_edit refalign_s_random-s0 judge_aao judge_matched judge_first judge_edit judge_random-s0 last_step penultimate random_step-s0
$V -m experiments.run_alfworld_replay --traces $R/alf_test_traces.json --jobs $R/alf_test_replay_jobs.json --out $R/alf_test_replay.json --seeds 0 1 --workers 16 --model gpt-4o-mini-2024-07-18 --base-url openai
echo "[chain] replay done $(date -u +%H:%M:%S)"
python3 -m experiments.rev_alfworld_analysis --prefix $R/alf_test --seeds 0 1
python3 -m experiments.rev_tables
echo "[chain] all done $(date -u +%H:%M:%S)"
