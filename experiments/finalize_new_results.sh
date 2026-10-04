#!/usr/bin/env bash
# Finalize: give the gpt-4o-mini runs the brief's file names (local runs -> _qwen2.5-7b),
# merge the per-judge Experiment 1 files, concatenate the API logs, regenerate the LaTeX
# tables and RESULTS_NEW.md, and run the tests. Idempotent.
set -euo pipefail
cd .
R=results/new
# 1) naming: primary = gpt-4o-mini, supplementary = Qwen (only once the gpt-4o-mini outputs exist)
swap () { # base ext  -> <base>.<ext> (Qwen) becomes <base>_qwen2.5-7b.<ext>; <base>_gpt-4o-mini.<ext> becomes <base>.<ext>
  local base=$1 ext=$2
  if [ -e "${base}_gpt-4o-mini.${ext}" ]; then
    [ -e "${base}.${ext}" ] && [ ! -e "${base}_qwen2.5-7b.${ext}" ] && mv "${base}.${ext}" "${base}_qwen2.5-7b.${ext}"
    mv "${base}_gpt-4o-mini.${ext}" "${base}.${ext}"
  fi; }
for b in intervention intervention_traces external_gsm8k external_gsm8k_traces semantic_variant semantic_variant_traces; do swap $R/$b json; done
for b in tab_intervention external_availability; do swap $R/$b tex; done
swap $R/intervention_prompts md
for d in external_pending semantic_pending; do
  if [ -d "annotation/${d}_gpt-4o-mini" ]; then
    [ -d "annotation/$d" ] && [ ! -d "annotation/${d}_qwen2.5-7b" ] && mv "annotation/$d" "annotation/${d}_qwen2.5-7b"
    mv "annotation/${d}_gpt-4o-mini" "annotation/$d"
  fi
done
# distinct LaTeX labels for the supplementary tables
for f in $R/tab_intervention_qwen2.5-7b.tex $R/external_availability_qwen2.5-7b.tex; do
  [ -e "$f" ] && sed -i 's/\\label{tab:\([a-z-]*\)}/\\label{tab:\1-qwen}/' "$f"
done
# 2) Experiment 1 merge (all judges)
PAIRED=$(ls $R/matched_judge_paired_*.json | grep -v "paired.json$")
META=$(ls $R/matched_judge_*.json | grep -v "paired" | grep -v "matched_judge.json$")
python3 -m experiments.run_matched_judge --table-only --merge-from $PAIRED --merge-meta-from $META
# 3) logs, report, tests
cat $R/logs/api_log_*.jsonl > $R/api_log.jsonl
wc -l $R/api_log.jsonl
python3 -m experiments.write_results_new
python3 -m pytest -q 2>&1 | tail -3
