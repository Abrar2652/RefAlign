#!/usr/bin/env bash
# Resume the test collection; the collector skips saved runs, so retries are safe.
export ALFWORLD_DATA=/path/to/workspace/alfworld_data
export PYTHONPATH=.
cd .
for attempt in 1 2 3 4 5 6; do
  echo "[resume] attempt $attempt $(date -u +%H:%M:%S)"
  /path/to/workspace/alf_venv/bin/python -m experiments.run_alfworld_collect --split valid_unseen --n-games 0 --game-seed 1 --n 8 --model gpt-4o-mini-2024-07-18 --base-url openai --workers 12 --out results/revision/alf_test_traces.json && break
  sleep 5
done
