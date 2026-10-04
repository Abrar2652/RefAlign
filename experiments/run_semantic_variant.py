"""Experiment 5: the semantic-error variant (feasibility / behaviour study).

Populations of the semantic Planner->Worker->Verifier system (experiments/
semantic_agents.py) on BIG-Bench Hard tracking_shuffled_objects_three_objects,
whose gold answers are option letters (categories) and whose decisive signal is a
wrong *semantic* state update, not a comparable number. On failing runs with a
reference we compare RefAlign (deterministic; expected to hit the fallback
often), RefAlign+LLM (adjudicator over the top rho=4 divergences), the matched
judge and single-trace AAO. There is NO step ground truth: we report fallback
rates, step distributions and agreement rates as an explicitly labelled proxy,
and export an annotation package (annotation/semantic_pending/).

    python -m experiments.run_semantic_variant --n-tasks 40 --n 14 \
        --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8103/v1 --revision <hash>
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np

from experiments import semantic_agents as SA
from experiments.run_intervention import trace_to_json, json_to_trace
from experiments.run_matched_judge import phylo_reference, det_seed as _det
from experiments.run_external_gsm8k import availability, export_annotation_package
from refalign.contrastive import contrastive_localize, contrastive_llm
from refalign.llm_baselines import all_at_once, matched_judge, parse_judge_answer
from refalign.llm_logged import LoggedLLM
from refalign.types import RunPopulation
from refalign import stats_new as S


def sample_tasks(n_tasks: int, seed: int) -> List[dict]:
    import datasets
    os.environ.setdefault("HF_HOME", "$HF_HOME")
    ds = datasets.load_dataset("lukaemon/bbh", "tracking_shuffled_objects_three_objects", split="test")
    rng = np.random.RandomState(seed)
    idx = sorted(int(i) for i in rng.choice(len(ds), size=n_tasks, replace=False))
    return [{"bbh_index": i, "question": ds[i]["input"], "answer": ds[i]["target"]} for i in idx]


def failure_mode(r: dict) -> str:
    if r["outcome"] == "success":
        return "success"
    ans = [s for s in r["steps"] if s["agent"] == "Worker" and s["content"].startswith("ANSWER")]
    if not ans:
        return "no_answer(turn_cap)"
    if r["final_answer"] in ("None", None, ""):
        return "answer_not_an_option"
    return "wrong_option"


def rollout(llm, qid: str, task: dict, i: int, temperature: float, max_turns: int) -> dict:
    seed = _det("exp5", qid, i)
    v = llm.view(seed=seed, tag=f"exp5|{qid}|r{i:02d}")
    tr, ok = SA.run_once_semantic(v, task["question"], task["answer"], temperature, max_turns)
    rec = trace_to_json(tr, f"{qid}_r{i:02d}", ok)
    rec.update({"seed_base": seed, "seeds_used": v.seeds_used, "n_calls": v.usage.calls,
                "tokens": v.usage.prompt_tokens + v.usage.completion_tokens,
                "response_models": sorted(v.response_models)})
    return rec


def attribute(pops: Dict[str, RunPopulation], judge, judge_name: str, workers: int, store: dict) -> dict:
    per_run, jobs = {}, []
    for q, pop in pops.items():
        if not pop.successes:
            continue
        for f in pop.failures:
            ref, own, how = phylo_reference(pop, f)
            per_run[f.run_id] = {"phylo": {"agent": own.agent, "step": int(own.step),
                                           "kind": own.meta.get("kind") or own.meta.get("fallback"),
                                           "reference": ref.run_id, "reference_how": how},
                                 "n_steps": len(f), "failure_mode": failure_mode(store[f.run_id])}
            jobs.append((pop, f, ref))

    def work(job):
        pop, f, ref = job
        v = judge.view(tag=f"exp5|{judge_name}|phylo+llm|{f.run_id}")
        pl = contrastive_llm(pop, f, v)
        pllm = {"agent": pl.agent, "step": int(pl.step), "fallback": pl.meta.get("fallback"),
                "tokens": sum(v.last_tokens) if v.usage.calls else 0, "llm_called": bool(v.usage.calls)}
        v = judge.view(tag=f"exp5|{judge_name}|aao|{f.run_id}")
        a = all_at_once(f, v)
        _, _, sp, ap_ = parse_judge_answer(v.last_raw, f)
        aao = {"agent": a.agent, "step": int(a.step), "step_parsed": sp, "agent_parsed": ap_, "tokens": sum(v.last_tokens)}
        v = judge.view(tag=f"exp5|{judge_name}|matched-phylo|{f.run_id}")
        m = matched_judge(f, ref, v)
        mj = {"agent": m.agent, "step": int(m.step), "step_parsed": m.meta["step_parsed"],
              "agent_parsed": m.meta["agent_parsed"], "tokens": sum(v.last_tokens), "reference": ref.run_id}
        return f.run_id, pllm, aao, mj

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rid, pllm, aao, mj in ex.map(work, jobs):
            per_run[rid].update({"phylo+llm": pllm, "aao": aao, "matched-phylo": mj})
    rids = sorted(per_run)
    if not rids:
        return {"n_failing_with_reference": 0, "per_run": {}}

    def agree(a, b, key="step"):
        return S.acc_record([per_run[r][a][key] == per_run[r][b][key] for r in rids])
    out = {"n_failing_with_reference": len(rids), "per_run": per_run,
           "phylo_fallback_rate": S.acc_record([per_run[r]["phylo"]["kind"] == "no_sibling" for r in rids]),
           "phylo_llm_fallback_rate": S.acc_record([per_run[r]["phylo+llm"]["fallback"] == "no_sibling" for r in rids]),
           "phylo_kind_distribution": dict(Counter(per_run[r]["phylo"]["kind"] for r in rids)),
           "step_distribution": {m: dict(sorted(Counter(per_run[r][m]["step"] for r in rids).items()))
                                 for m in ("phylo", "phylo+llm", "matched-phylo", "aao")},
           "agent_distribution": {m: dict(Counter(per_run[r][m]["agent"] for r in rids))
                                  for m in ("phylo", "phylo+llm", "matched-phylo", "aao")},
           "last_step_rate": {m: S.acc_record([per_run[r][m]["step"] == per_run[r]["n_steps"] - 1 for r in rids])
                              for m in ("phylo", "phylo+llm", "matched-phylo", "aao")},
           "failure_mode_distribution": dict(Counter(per_run[r]["failure_mode"] for r in rids)),
           "agreement_PROXY_not_accuracy": {
               "note": "agreement between attribution methods on UNLABELED runs; not accuracy",
               "phylo_vs_phylo+llm": agree("phylo", "phylo+llm"),
               "phylo_vs_matched-phylo": agree("phylo", "matched-phylo"),
               "phylo+llm_vs_matched-phylo": agree("phylo+llm", "matched-phylo"),
               "phylo_vs_aao": agree("phylo", "aao"),
               "matched-phylo_vs_aao": agree("matched-phylo", "aao"),
               "phylo+llm_vs_aao": agree("phylo+llm", "aao")},
           "unparsed_step": {"aao": int(sum(not per_run[r]["aao"]["step_parsed"] for r in rids)),
                             "matched-phylo": int(sum(not per_run[r]["matched-phylo"]["step_parsed"] for r in rids))},
           "tokens_mean": {m: float(np.mean([per_run[r][m]["tokens"] for r in rids])) for m in ("phylo+llm", "aao", "matched-phylo")},
           "judge": judge_name}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=40)
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--task-seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8103/v1")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--annot-n", type=int, default=30)
    ap.add_argument("--annot-dir", default="annotation/semantic_pending")
    ap.add_argument("--out", default="results/new/semantic_variant.json")
    ap.add_argument("--traces", default="results/new/semantic_variant_traces.json")
    ap.add_argument("--log", default="results/new/api_log.jsonl")
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.base_url or args.base_url.lower() in ("none", "openai"):
        args.base_url = None            # the OpenAI API itself (key from .env)

    tl = sample_tasks(args.n_tasks, args.task_seed)
    tasks = {f"bbh_tso3_{t['bbh_index']:03d}": t for t in tl}
    store = json.load(open(args.traces)) if os.path.exists(args.traces) else {}
    meta = {"dataset": "lukaemon/bbh tracking_shuffled_objects_three_objects (BIG-Bench Hard), HF datasets",
            "why_this_dataset": "public; gold answers are option letters (categories, not numbers); solving needs a "
                                "multi-step chain of semantic state updates, so a wrong step is a wrong statement, "
                                "not a comparable value -- the regime RefAlign+LLM was proposed for",
            "task_seed": args.task_seed, "bbh_indices": sorted(t["bbh_index"] for t in tasks.values()),
            "model": args.model, "revision": args.revision, "base_url": args.base_url,
            "temperature": args.temperature, "max_tokens": args.max_tokens, "max_worker_turns": args.max_turns,
            "n_rollouts": args.n, "system": "experiments/semantic_agents.run_once_semantic (Planner->Worker[NOTE/ANSWER]->Verifier)",
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if not args.analyse_only:
        llm = LoggedLLM(args.model, base_url=args.base_url, temperature=args.temperature,
                        max_tokens=args.max_tokens, log_path=args.log, tag="exp5", revision=args.revision)
        todo = [(q, i) for q in tasks for i in range(args.n) if f"{q}_r{i:02d}" not in store]
        print(f"[exp5] {len(todo)} rollouts to run", flush=True)
        t0 = time.time()

        def save():
            json.dump(store, open(args.traces, "w"), indent=0)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(rollout, llm, q, tasks[q], i, args.temperature, args.max_turns) for q, i in todo]
            for k, fut in enumerate(as_completed(futs), 1):
                rec = fut.result(); store[rec["run_id"]] = rec
                if k % 50 == 0 or k == len(todo):
                    print(f"[exp5] {k}/{len(todo)} rollouts, {time.time() - t0:.0f}s", flush=True); save()
        save()
        meta["usage_rollouts"] = {"calls": llm.usage.calls, "prompt_tokens": llm.usage.prompt_tokens,
                                  "completion_tokens": llm.usage.completion_tokens, "seconds": round(llm.usage.seconds, 1),
                                  "errors": llm.n_errors, "usd": 0.0 if args.base_url else None}
        meta["response_models"] = sorted({m for r in store.values() for m in r.get("response_models", [])})
    pops = {}
    for q, t in tasks.items():
        runs = [store[f"{q}_r{i:02d}"] for i in range(args.n) if f"{q}_r{i:02d}" in store]
        pops[q] = RunPopulation(q, [json_to_trace(r, q, t["question"]) for r in runs], t["question"])
    avail = availability({q: {**t, "gsm8k_test_index": t["bbh_index"]} for q, t in tasks.items()}, pops, args.n)
    judge_name = args.model.split("/")[-1]
    judge = LoggedLLM(args.model, base_url=args.base_url, temperature=0.0, max_tokens=128,
                      log_path=args.log, tag=f"exp5-judge|{judge_name}", revision=args.revision)
    attr = attribute(pops, judge, judge_name, args.workers, store)
    meta["judge"] = {"name": judge_name, "model": args.model, "temperature": 0.0, "max_tokens": 128,
                     "usage": {"calls": judge.usage.calls, "prompt_tokens": judge.usage.prompt_tokens,
                               "completion_tokens": judge.usage.completion_tokens, "errors": judge.n_errors}}
    pkg = export_annotation_package({q: {**t, "gsm8k_test_index": t["bbh_index"]} for q, t in tasks.items()},
                                    pops, store, args.annot_dir, args.annot_n, args.task_seed,
                                    "BBH tracking_shuffled_objects_three_objects")
    meta["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    res = {"design": meta, "availability": avail, "attribution": attr, "annotation_package": pkg,
           "status": "feasibility/behaviour study on UNLABELED runs; no accuracy is claimed"}
    json.dump(res, open(args.out, "w"), indent=1, default=float)
    a = avail
    print(f"\n=== semantic variant: {a['n_tasks']} tasks x {a['rollouts_per_task']} ===")
    print(f"  mixed tasks {a['tasks_with_success_and_failure']['c']}/{a['n_tasks']}, all-success {a['tasks_all_success']}, all-fail {a['tasks_all_failure']}, "
          f"run success {a['success_rate_runs']['acc']:.3f}; failing w/ ref {a['failing_runs_with_reference']}")
    if attr.get("n_failing_with_reference"):
        print(f"  phylo fallback {attr['phylo_fallback_rate']['c']}/{attr['phylo_fallback_rate']['n']}; kinds {attr['phylo_kind_distribution']}")
        print(f"  step dists: {attr['step_distribution']}")
        print(f"  last-step rate: { {m: round(v['acc'],2) for m, v in attr['last_step_rate'].items()} }")
        print(f"  AGREEMENT PROXY: { {k: round(v['acc'],2) for k, v in attr['agreement_PROXY_not_accuracy'].items() if isinstance(v, dict)} }")
        print(f"  unparsed {attr['unparsed_step']}")
    print(f"  annotation package: {pkg['n_runs']} runs / {pkg['n_tasks']} tasks -> {pkg['dir']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
