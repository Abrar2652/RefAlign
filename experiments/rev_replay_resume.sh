#!/usr/bin/env bash
# Resume the ALFWorld replays (resumable; a worker crash breaks the pool, so retry), then analysis + tables.
export ALFWORLD_DATA=/path/to/workspace/alfworld_data
export PYTHONPATH=.
cd .
V=/path/to/workspace/alf_venv/bin/python
R=results/revision
for attempt in 1 2 3 4 5 6 7 8; do
  echo "[replay-resume] attempt $attempt $(date -u +%H:%M:%S)"
  $V -m experiments.run_alfworld_replay --traces $R/alf_test_traces.json --jobs $R/alf_test_replay_jobs.json --out $R/alf_test_replay.json --seeds 0 1 --workers 12 --model gpt-4o-mini-2024-07-18 --base-url openai && break
  sleep 5
done
echo "[replay-resume] replay done $(date -u +%H:%M:%S)"
python3 -m experiments.rev_alfworld_analysis --prefix $R/alf_test --seeds 0 1
python3 -m experiments.rev_tables
echo "[replay-resume] all done $(date -u +%H:%M:%S)"
