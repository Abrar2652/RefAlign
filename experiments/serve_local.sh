#!/usr/bin/env bash
# Launch a local vLLM OpenAI-compatible server for one open model, pinned to the
# cached HF snapshot revision so every call is reproducible.
#   usage: serve_local.sh <gpu> <port> <hf-model-id> <revision> [extra vllm args]
# Logs go to results/new/logs/vllm_<port>.log. Used for every experiment that the
# exhausted OpenAI account could not serve (see results/new/RESULTS_NEW.md).
set -euo pipefail
GPU=$1; PORT=$2; MODEL=$3; REV=$4; shift 4
export HF_HOME=$HF_HOME
export CUDA_VISIBLE_DEVICES=$GPU
export VLLM_LOGGING_LEVEL=WARNING
PY=/path/to/workspace/candor_env/venv/bin/python
LOG=results/new/logs/vllm_${PORT}.log
exec $PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --revision "$REV" --served-model-name "$MODEL" \
  --port "$PORT" --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --seed 0 --generation-config vllm --disable-log-requests "$@" >"$LOG" 2>&1
