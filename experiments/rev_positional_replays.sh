#!/usr/bin/env bash
# Replay the leave-one-type-out positional selections (persisted by the analysis), then re-analyse.
set -euo pipefail
export ALFWORLD_DATA=/path/to/workspace/alfworld_data
export PYTHONPATH=.
cd .
V=/path/to/workspace/alf_venv/bin/python
R=results/revision
python3 -m experiments.rev_alfworld_analysis --prefix $R/alf_test --seeds 0 1 > /dev/null
python3 -m experiments.rev_alfworld_jobs --attribution $R/alf_test_attribution.json --traces $R/alf_test_traces.json --out $R/alf_test_replay_jobs_pos.json --profile-n 40 --kmax 10 --method-subset-n 100 --methods lo_mode_step lo_norm_pos
for attempt in 1 2 3 4; do
  $V -m experiments.run_alfworld_replay --traces $R/alf_test_traces.json --jobs $R/alf_test_replay_jobs_pos.json --out $R/alf_test_replay.json --seeds 0 1 --workers 12 --model gpt-4o-mini-2024-07-18 --base-url openai && break
  sleep 5
done
python3 -m experiments.rev_alfworld_analysis --prefix $R/alf_test --seeds 0 1
python3 -m experiments.rev_tables
echo "[pos-replays] all done $(date -u +%H:%M:%S)"
