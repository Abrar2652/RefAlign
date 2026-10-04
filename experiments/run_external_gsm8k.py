"""Experiment 4A: traces not generated for this work -- GSM8K.

Runs the existing Planner->Worker->Verifier system (experiments/real_agents.py)
on randomly sampled GSM8K test problems, many rollouts per problem at the same
temperature as the human-benchmark collection, and reports:

  * reference availability: fraction of tasks with >=1 success AND >=1 failure,
    rollouts to first success, overall success rate (label-free, quantifies the
    applicability condition of populated-trace attribution);
  * RefAlign, single-trace AAO and the matched judge on every failing run that
    has a reference. GSM8K gives only success/failure, so there is NO
    decisive-step ground truth: agreement between methods is reported as an
    explicitly labelled proxy, never as accuracy;
  * an annotation package (annotation/external_pending/) for a random subset of
    failing runs in the exact format annotate.py consumes (first-success
    reference), schema-validated and loadable by load_annotated_benchmark.

    python -m experiments.run_external_gsm8k --n-tasks 60 --n 14 \
        --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8103/v1 --revision <hash>
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments import real_agents as RA
from experiments.run_intervention import trace_to_json, json_to_trace, failure_mode
from experiments.run_matched_judge import phylo_reference, det_seed as _det
from refalign.contrastive import contrastive_localize
from refalign.data.schema import load_annotated_benchmark
from refalign.llm_baselines import all_at_once, matched_judge, parse_judge_answer
from refalign.llm_logged import LoggedLLM
from refalign.types import RunPopulation, Trace
from refalign import stats_new as S


def gsm8k_answer(ans: str) -> Optional[float]:
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", ans)
    return float(m.group(1).replace(",", "")) if m else None


def sample_tasks(n_tasks: int, seed: int) -> List[dict]:
    import datasets
    os.environ.setdefault("HF_HOME", "$HF_HOME")
    ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
    rng = np.random.RandomState(seed)
    idx = sorted(int(i) for i in rng.choice(len(ds), size=n_tasks, replace=False))
    out = []
    for i in idx:
        ex = ds[i]
        out.append({"gsm8k_test_index": i, "question": ex["question"], "answer": gsm8k_answer(ex["answer"]),
                    "answer_text": ex["answer"]})
    return out


def rollout(llm, tag: str, qid: str, task: dict, i: int, temperature: float, max_turns: int) -> dict:
    seed = _det(tag, qid, i)
    v = llm.view(seed=seed, tag=f"{tag}|{qid}|r{i:02d}")
    tr, ok, _, _ = RA.run_once(v, task["question"], float(task["answer"]), temperature=temperature,
                               max_worker_turns=max_turns)
    rec = trace_to_json(tr, f"{qid}_r{i:02d}", ok)
    rec.update({"seed_base": seed, "seeds_used": v.seeds_used, "n_calls": v.usage.calls,
                "tokens": v.usage.prompt_tokens + v.usage.completion_tokens,
                "response_models": sorted(v.response_models)})
    return rec


def collect(llm, tag: str, tasks: Dict[str, dict], n: int, temperature: float, max_turns: int,
            workers: int, store: dict, save_cb=None) -> None:
    todo = [(q, i) for q in tasks for i in range(n) if f"{q}_r{i:02d}" not in store]
    print(f"[{tag}] {len(todo)} rollouts to run", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(rollout, llm, tag, q, tasks[q], i, temperature, max_turns) for q, i in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            store[rec["run_id"]] = rec
            if k % 50 == 0 or k == len(todo):
                print(f"[{tag}] {k}/{len(todo)} rollouts, {time.time() - t0:.0f}s", flush=True)
                if save_cb:
                    save_cb()


def populations(tasks: Dict[str, dict], store: dict, n: int) -> Dict[str, RunPopulation]:
    pops = {}
    for q, t in tasks.items():
        runs = [store[f"{q}_r{i:02d}"] for i in range(n) if f"{q}_r{i:02d}" in store]
        traces = [json_to_trace(r, q, t["question"]) for r in runs]
        pops[q] = RunPopulation(q, traces, t["question"])
    return pops


def availability(tasks: Dict[str, dict], pops: Dict[str, RunPopulation], n: int) -> dict:
    both, only_s, only_f, first_success = 0, 0, 0, []
    n_succ_runs = n_runs = 0
    per_task = {}
    for q, pop in pops.items():
        ns, nf = len(pop.successes), len(pop.failures)
        n_succ_runs += ns; n_runs += len(pop.runs)
        if ns and nf:
            both += 1
        elif ns:
            only_s += 1
        else:
            only_f += 1
        fs = next((i for i, r in enumerate(pop.runs) if r.success), None)
        first_success.append(None if fs is None else fs + 1)
        per_task[q] = {"n_success": ns, "n_failure": nf, "rollouts_to_first_success": None if fs is None else fs + 1,
                       "gsm8k_test_index": tasks[q]["gsm8k_test_index"]}
    fs_vals = [x for x in first_success if x is not None]
    return {"n_tasks": len(pops), "rollouts_per_task": n,
            "tasks_with_success_and_failure": S.acc_record([bool(p.successes and p.failures) for p in pops.values()]),
            "tasks_all_success": only_s, "tasks_all_failure": only_f, "tasks_mixed": both,
            "success_rate_runs": S.acc_record([r.success for p in pops.values() for r in p.runs]),
            "rollouts_to_first_success": {"n_tasks_with_success": len(fs_vals),
                                          "mean": float(np.mean(fs_vals)) if fs_vals else None,
                                          "median": float(np.median(fs_vals)) if fs_vals else None,
                                          "max": int(max(fs_vals)) if fs_vals else None,
                                          "histogram": dict(sorted(Counter(fs_vals).items()))},
            "failing_runs_with_reference": sum(len(p.failures) for p in pops.values() if p.successes),
            "failing_runs_without_reference": sum(len(p.failures) for p in pops.values() if not p.successes),
            "per_task": per_task}


def attribute(pops: Dict[str, RunPopulation], judge, judge_name: str, workers: int, store: dict) -> dict:
    """RefAlign, AAO and matched-phylo on every failing run with a reference.
    Returns per-run predictions + agreement rates (a proxy; there are no labels)."""
    per_run = {}
    jobs = []
    for q, pop in pops.items():
        if not pop.successes:
            continue
        for f in pop.failures:
            ref, own, how = phylo_reference(pop, f)
            per_run[f.run_id] = {"phylo": {"agent": own.agent, "step": int(own.step),
                                           "kind": own.meta.get("kind") or own.meta.get("fallback"),
                                           "reference": ref.run_id, "reference_how": how},
                                 "n_steps": len(f), "failure_mode": failure_mode(store[f.run_id])}
            if judge is not None:
                jobs.append((f, ref))

    def work(job):
        f, ref = job
        v = judge.view(tag=f"exp4|{judge_name}|aao|{f.run_id}")
        a = all_at_once(f, v)
        _, _, sp, ap_ = parse_judge_answer(v.last_raw, f)
        aao = {"agent": a.agent, "step": int(a.step), "step_parsed": sp, "agent_parsed": ap_,
               "tokens": sum(v.last_tokens)}
        v2 = judge.view(tag=f"exp4|{judge_name}|matched-phylo|{f.run_id}")
        m = matched_judge(f, ref, v2)
        mj = {"agent": m.agent, "step": int(m.step), "step_parsed": m.meta["step_parsed"],
              "agent_parsed": m.meta["agent_parsed"], "tokens": sum(v2.last_tokens), "reference": ref.run_id}
        return f.run_id, aao, mj

    if jobs:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for rid, aao, mj in ex.map(work, jobs):
                per_run[rid]["aao"] = aao
                per_run[rid]["matched-phylo"] = mj
    rids = sorted(per_run)
    out = {"n_failing_with_reference": len(rids), "per_run": per_run,
           "phylo_fallback_rate": S.acc_record([per_run[r]["phylo"]["kind"] == "no_sibling" for r in rids]) if rids else None,
           "phylo_kind_distribution": dict(Counter(per_run[r]["phylo"]["kind"] for r in rids)),
           "phylo_step_distribution": dict(sorted(Counter(per_run[r]["phylo"]["step"] for r in rids).items())),
           "phylo_agent_distribution": dict(Counter(per_run[r]["phylo"]["agent"] for r in rids)),
           "failure_mode_distribution": dict(Counter(per_run[r]["failure_mode"] for r in rids))}
    if judge is not None and rids:
        def agree(a, b, key):
            return S.acc_record([per_run[r][a][key] == per_run[r][b][key] for r in rids])
        out["agreement_PROXY_not_accuracy"] = {
            "note": "agreement between two attribution methods on UNLABELED runs; this is not accuracy",
            "phylo_vs_matched-phylo": {"step": agree("phylo", "matched-phylo", "step"),
                                       "agent": agree("phylo", "matched-phylo", "agent"),
                                       "step_within_1": S.acc_record([abs(per_run[r]["phylo"]["step"] - per_run[r]["matched-phylo"]["step"]) <= 1 for r in rids])},
            "phylo_vs_aao": {"step": agree("phylo", "aao", "step"), "agent": agree("phylo", "aao", "agent")},
            "matched-phylo_vs_aao": {"step": agree("matched-phylo", "aao", "step"), "agent": agree("matched-phylo", "aao", "agent")},
            "judge": judge_name,
            "aao_unparsed_step": int(sum(1 for r in rids if not per_run[r]["aao"]["step_parsed"])),
            "matched_unparsed_step": int(sum(1 for r in rids if not per_run[r]["matched-phylo"]["step_parsed"])),
            "tokens_mean": {"aao": float(np.mean([per_run[r]["aao"]["tokens"] for r in rids])),
                            "matched-phylo": float(np.mean([per_run[r]["matched-phylo"]["tokens"] for r in rids]))}}
    return out


def export_annotation_package(tasks: Dict[str, dict], pops: Dict[str, RunPopulation], store: dict,
                              out_dir: str, n_runs: int, seed: int, source: str) -> dict:
    """Random subset of failing runs WITH a reference, written per task in the
    annotate.py trace format (annotators see successes[0] as the reference)."""
    cands = sorted(f.run_id for p in pops.values() if p.successes for f in p.failures)
    rng = np.random.RandomState(seed)
    chosen = set(rng.choice(cands, size=min(n_runs, len(cands)), replace=False).tolist()) if cands else set()
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):
        if old.endswith(".json"):
            os.remove(os.path.join(out_dir, old))
    written = {}
    for q, pop in pops.items():
        fails = [r for r in pop.failures if r.run_id in chosen]
        if not fails:
            continue
        runs = [pop.successes[0]] + fails          # first success first: annotate.py shows successes[0]
        obj = {"query_id": q, "query": tasks[q]["question"], "ground_truth": str(tasks[q]["answer"]),
               "source": source, "gsm8k_test_index": tasks[q]["gsm8k_test_index"],
               "note": "UNLABELED external package; reference = first successful rollout; "
                       "all other successful rollouts are omitted so the annotator view matches annotate.py",
               "runs": [{k: v for k, v in store[r.run_id].items()
                         if k in ("run_id", "outcome", "final_answer", "steps")} for r in runs]}
        for r in obj["runs"]:                       # annotate.py schema: index/agent/content per step
            r["steps"] = [{"index": s["index"], "agent": s["agent"], "content": s["content"]} for s in r["steps"]]
        with open(os.path.join(out_dir, f"{q}.json"), "w") as fh:
            json.dump(obj, fh, indent=1)
        written[q] = [r.run_id for r in fails]
    # schema validation: loadable by the benchmark loader, outcomes well-formed, indices contiguous
    pops_back, _ = load_annotated_benchmark(out_dir, out_dir + "/__no_labels__")
    n_fail_back = sum(len(p.failures) for p in pops_back)
    assert n_fail_back == len(chosen), (n_fail_back, len(chosen))
    for f in os.listdir(out_dir):
        obj = json.load(open(os.path.join(out_dir, f)))
        assert obj["runs"][0]["outcome"] == "success"
        for r in obj["runs"]:
            assert r["outcome"] in ("success", "failure")
            assert [s["index"] for s in r["steps"]] == list(range(len(r["steps"])))
            assert all({"index", "agent", "content"} <= set(s) for s in r["steps"])
    readme = (f"# Unlabeled external annotation package ({source})\n\n"
              f"{len(chosen)} failing runs from {len(written)} tasks, each file holding the FIRST successful rollout "
              f"(the reference annotators see, as in `annotation/annotate.py`) followed by the sampled failing runs. "
              f"No decisive-step labels exist yet; do not report accuracy on these runs until they are labeled.\n\n"
              f"Annotate with:\n\n```\npython -m annotation.annotate --annotator annotator_XX --traces {out_dir} "
              f"--out annotation/annotations_external\n```\n\nSelection: uniform random subset (seed {seed}) of failing runs "
              f"whose task has at least one successful rollout.\n")
    open(os.path.join(out_dir, "README.md"), "w").write(readme)
    return {"dir": out_dir, "n_runs": len(chosen), "n_tasks": len(written), "runs": written, "seed": seed}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=60)
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--task-seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8103/v1")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--judge-model", default=None, help="defaults to --model")
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-name", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-turns", type=int, default=8,
                    help="max Worker turns; GSM8K needs more than the 5 used for the one-step word problems")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--annot-n", type=int, default=40)
    ap.add_argument("--annot-dir", default="annotation/external_pending")
    ap.add_argument("--out", default="results/new/external_gsm8k.json")
    ap.add_argument("--traces", default="results/new/external_gsm8k_traces.json")
    ap.add_argument("--tex", default="results/new/external_availability.tex")
    ap.add_argument("--log", default="results/new/api_log.jsonl")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--tex-only", action="store_true", help="rebuild the LaTeX table from --out without any calls")
    args = ap.parse_args(argv)
    if not args.base_url or args.base_url.lower() in ("none", "openai"):
        args.base_url = None            # the OpenAI API itself (key from .env)
    if args.tex_only:
        write_tex(json.load(open(args.out)), args.tex)
        print(f"wrote {args.tex}")
        return 0

    tasks_list = sample_tasks(args.n_tasks, args.task_seed)
    tasks = {f"gsm8k_{t['gsm8k_test_index']:04d}": t for t in tasks_list}
    assert all(t["answer"] is not None for t in tasks.values())
    store = json.load(open(args.traces)) if os.path.exists(args.traces) else {}
    meta = {"dataset": "openai/gsm8k main test (HF datasets)", "task_seed": args.task_seed,
            "gsm8k_test_indices": sorted(t["gsm8k_test_index"] for t in tasks.values()),
            "model": args.model, "revision": args.revision, "base_url": args.base_url,
            "temperature": args.temperature, "max_tokens": args.max_tokens, "max_worker_turns": args.max_turns,
            "n_rollouts": args.n, "system": "experiments/real_agents.run_once (Planner->Worker->Verifier), unchanged prompts",
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    judge_model = args.judge_model or args.model
    judge_name = args.judge_name or judge_model.split("/")[-1]
    if not args.analyse_only:
        llm = LoggedLLM(args.model, base_url=args.base_url, temperature=args.temperature,
                        max_tokens=args.max_tokens, log_path=args.log, tag="exp4", revision=args.revision)

        def save():
            json.dump(store, open(args.traces, "w"), indent=0)
        collect(llm, "exp4", tasks, args.n, args.temperature, args.max_turns, args.workers, store, save)
        save()
        meta["usage_rollouts"] = {"calls": llm.usage.calls, "prompt_tokens": llm.usage.prompt_tokens,
                                  "completion_tokens": llm.usage.completion_tokens,
                                  "seconds": round(llm.usage.seconds, 1), "errors": llm.n_errors, "usd": 0.0 if args.base_url else None}
        meta["response_models"] = sorted({m for r in store.values() for m in r.get("response_models", [])})
    pops = populations(tasks, store, args.n)
    avail = availability(tasks, pops, args.n)
    judge = LoggedLLM(judge_model, base_url=args.judge_base_url or args.base_url, temperature=0.0,
                      max_tokens=128, log_path=args.log, tag=f"exp4-judge|{judge_name}", revision=args.revision)
    attr = attribute(pops, judge, judge_name, args.workers, store)
    meta["judge"] = {"name": judge_name, "model": judge_model, "temperature": 0.0, "max_tokens": 128,
                     "usage": {"calls": judge.usage.calls, "prompt_tokens": judge.usage.prompt_tokens,
                               "completion_tokens": judge.usage.completion_tokens, "errors": judge.n_errors}}
    pkg = export_annotation_package(tasks, pops, store, args.annot_dir, args.annot_n, args.task_seed, "GSM8K test")
    meta["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    res = {"design": meta, "availability": avail, "attribution": attr, "annotation_package": pkg}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1, default=float)
    write_tex(res, args.tex)
    a = avail
    print(f"\n=== GSM8K reference availability ({a['n_tasks']} tasks x {a['rollouts_per_task']} rollouts) ===")
    print(f"  tasks with >=1 success and >=1 failure: {a['tasks_with_success_and_failure']['c']}/{a['n_tasks']} "
          f"[{a['tasks_with_success_and_failure']['wilson_lo']:.2f},{a['tasks_with_success_and_failure']['wilson_hi']:.2f}]  "
          f"all-success {a['tasks_all_success']}  all-failure {a['tasks_all_failure']}")
    print(f"  run success rate {a['success_rate_runs']['c']}/{a['success_rate_runs']['n']} = {a['success_rate_runs']['acc']:.3f}; "
          f"rollouts to first success mean {a['rollouts_to_first_success']['mean']}, median {a['rollouts_to_first_success']['median']}, hist {a['rollouts_to_first_success']['histogram']}")
    print(f"  failing runs with reference {a['failing_runs_with_reference']}, without {a['failing_runs_without_reference']}")
    print(f"  phylo kinds {attr['phylo_kind_distribution']} steps {attr['phylo_step_distribution']}")
    if "agreement_PROXY_not_accuracy" in attr:
        g = attr["agreement_PROXY_not_accuracy"]
        print(f"  AGREEMENT PROXY (unlabeled): phylo~matched step {g['phylo_vs_matched-phylo']['step']['acc']:.2f} "
              f"(+-1 {g['phylo_vs_matched-phylo']['step_within_1']['acc']:.2f}), phylo~aao {g['phylo_vs_aao']['step']['acc']:.2f}, "
              f"matched~aao {g['matched-phylo_vs_aao']['step']['acc']:.2f}; unparsed aao {g['aao_unparsed_step']} matched {g['matched_unparsed_step']}")
    print(f"  annotation package: {pkg['n_runs']} runs / {pkg['n_tasks']} tasks -> {pkg['dir']}")
    print(f"wrote {args.out}, {args.traces}, {args.tex}")
    return 0


def write_tex(res: dict, path: str) -> None:
    a, d = res["availability"], res["design"]
    m = a["tasks_with_success_and_failure"]
    s = a["success_rate_runs"]
    fs = a["rollouts_to_first_success"]
    attr = res["attribution"]
    g = attr.get("agreement_PROXY_not_accuracy")
    judge_model = (d.get("judge") or {}).get("model", "")
    n_pkg = res["annotation_package"]["n_runs"]
    rows = [
        f"tasks sampled (GSM8K test, seed {d['task_seed']}) & {a['n_tasks']} \\\\",
        f"rollouts per task & {a['rollouts_per_task']} \\\\",
        f"tasks with $\\ge 1$ success and $\\ge 1$ failure & {m['c']}/{m['n']} = {m['acc']:.2f} [{m['wilson_lo']:.2f}, {m['wilson_hi']:.2f}] \\\\",
        f"tasks with all rollouts successful & {a['tasks_all_success']} \\\\",
        f"tasks with no successful rollout (no reference) & {a['tasks_all_failure']} \\\\",
        f"run-level success rate & {s['c']}/{s['n']} = {s['acc']:.2f} [{s['wilson_lo']:.2f}, {s['wilson_hi']:.2f}] \\\\",
        f"rollouts to first success (mean / median / max) & {fs['mean']:.2f} / {fs['median']:.0f} / {fs['max']} \\\\" if fs["mean"] is not None else "rollouts to first success & -- \\\\",
        f"failing runs with a reference / without & {a['failing_runs_with_reference']} / {a['failing_runs_without_reference']} \\\\",
        f"RefAlign fallback rate on failing runs with a reference & {attr['phylo_fallback_rate']['c']}/{attr['phylo_fallback_rate']['n']} \\\\" if attr.get("phylo_fallback_rate") else "",
    ]
    if g:
        rows += [f"\\midrule\n\\multicolumn{{2}}{{l}}{{\\emph{{agreement between methods (unlabeled runs; a proxy, not accuracy)}}}} \\\\",
                 f"RefAlign vs.\\ matched judge, same step / within $\\pm1$ & {g['phylo_vs_matched-phylo']['step']['acc']:.2f} [{g['phylo_vs_matched-phylo']['step']['wilson_lo']:.2f}, {g['phylo_vs_matched-phylo']['step']['wilson_hi']:.2f}] / {g['phylo_vs_matched-phylo']['step_within_1']['acc']:.2f} \\\\",
                 f"RefAlign vs.\\ single-trace AAO, same step & {g['phylo_vs_aao']['step']['acc']:.2f} [{g['phylo_vs_aao']['step']['wilson_lo']:.2f}, {g['phylo_vs_aao']['step']['wilson_hi']:.2f}] \\\\",
                 f"matched judge vs.\\ single-trace AAO, same step & {g['matched-phylo_vs_aao']['step']['acc']:.2f} [{g['matched-phylo_vs_aao']['step']['wilson_lo']:.2f}, {g['matched-phylo_vs_aao']['step']['wilson_hi']:.2f}] \\\\"]
    tex = f"""% Experiment 4A: reference availability on GSM8K (external tasks).
\\begin{{table}}[t]
\\centering\\small
\\begin{{tabular}}{{lc}}
\\toprule
quantity & value \\\\
\\midrule
{chr(10).join(r for r in rows if r)}
\\bottomrule
\\end{{tabular}}
\\caption{{Reference availability of the Planner$\\to$Worker$\\to$Verifier system on {a['n_tasks']} randomly
sampled GSM8K test problems ({a['rollouts_per_task']} rollouts each, {d['model'].split('/')[-1]}, temperature {d['temperature']},
fixed seeds; problem ids in \\texttt{{results/new/external\\_gsm8k.json}}). Intervals are 95\\% Wilson.
GSM8K provides only the final answer, so there is no decisive-step ground truth: the agreement rows compare
two attribution methods on unlabeled failing runs and must not be read as accuracy; the judge is
\\texttt{{{judge_model}}} at temperature 0. An unlabeled annotation package for
{n_pkg} failing runs is exported to \\texttt{{annotation/external\\_pending/}}.}}
\\label{{tab:external-availability}}
\\end{{table}}
"""
    open(path, "w").write(tex)


if __name__ == "__main__":
    sys.exit(main())
