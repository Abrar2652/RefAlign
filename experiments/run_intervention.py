"""Experiment 3: from attribution to a measured improvement (intervention).

The human labels concentrate at the Worker's computation step (27/32 at step 3).
We regenerate every task family under three prompt conditions with identical
sampling (same model, temperature, seeds) and ask whether intervening at the
attributed step lowers the failure rate while an equally long intervention
elsewhere (the Planner) does not:

  * baseline  : the original prompts of experiments/real_agents.py
  * targeted  : ONE verification instruction appended to the Worker prompt
  * placebo   : an equally long instruction appended to the Planner prompt only

Failure rate per task and pooled with 95% Wilson intervals; two-sided Fisher
exact tests baseline vs targeted and baseline vs placebo (Holm family = the two
comparisons per task); Newcombe CI on the failure-rate difference. RefAlign is
re-run on the new failures to see whether its decisive-step distribution shifts.

    python -m experiments.run_intervention --model Qwen/Qwen2.5-7B-Instruct \
        --base-url http://localhost:8103/v1 --revision <hash> --tasks all
"""
from __future__ import annotations

import argparse
import datetime
import difflib
import json
import os
import re
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from unittest import mock

import numpy as np

from experiments import real_agents as RA
from annotation.collect_traces import TASKS as CANDIDATE_TASKS
from refalign.contrastive import contrastive_localize
from refalign.data.schema import _action_of
from refalign.llm_logged import LoggedLLM
from refalign.types import RunPopulation, Step, Trace
from refalign import stats_new as S

RETAINED = [0, 5, 7, 9]           # real_t00, t05, t07, t09 (human benchmark families)
CONDITIONS = ["baseline", "targeted", "placebo"]

TARGETED_INSTRUCTION = (
    "Before giving the final value, re-read the question and confirm every stated "
    "operation, including any final percentage, multiplication or division, has been "
    "applied explicitly.")
PLACEBO_INSTRUCTION = (
    "Before stating the plan, re-read the question and list every stated quantity, then describe every operation, including any final percentage or division, in the required order.")


def det_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


def prompts_for(cond: str) -> Dict[str, str]:
    planner, worker = RA.PLANNER_SYS, RA.WORKER_SYS
    if cond == "targeted":
        worker = worker + "\n" + TARGETED_INSTRUCTION
    elif cond == "placebo":
        planner = planner + "\n" + PLACEBO_INSTRUCTION
    return {"PLANNER_SYS": planner, "WORKER_SYS": worker, "VERIFIER_SYS": RA.VERIFIER_SYS}


def trace_to_json(tr: Trace, rid: str, ok: bool) -> dict:
    return {"run_id": rid, "outcome": "success" if ok else "failure",
            "final_answer": tr.final_answer,
            "steps": [{"index": s.index, "agent": s.agent, "action": s.action, "content": s.content}
                      for s in tr.steps]}


def json_to_trace(r: dict, qid: str, query: str) -> Trace:
    steps = [Step(s["index"], s["agent"], s.get("action") or _action_of(s["agent"], s["content"]), s["content"])
             for s in r["steps"]]
    return Trace(r["run_id"], qid, steps, r["outcome"] == "success", query=query,
                 final_answer=str(r.get("final_answer")))


def failure_mode(r: dict) -> str:
    """Heuristic, label-free description of HOW a failing run failed (for the
    decisive-step shift analysis; not a ground-truth fault type)."""
    if r["outcome"] == "success":
        return "success"
    ans = [s for s in r["steps"] if s["agent"] == "Worker" and s["content"].startswith("ANSWER")]
    if not ans:
        return "no_answer(turn_cap)"
    a = ans[-1]["content"]
    if re.search(r"CALC|[-+*/]\s*\d", a[len("ANSWER"):]):
        return "answer_is_expression"
    if r["final_answer"] in ("None", None, ""):
        return "answer_not_numeric"
    return "wrong_number"


def evaluated_final(rec: dict) -> Optional[float]:
    """Secondary scoring: if the Worker's ANSWER carries an arithmetic expression
    (the 'ANSWER CALC 0.0714 * 350' pattern), evaluate it instead of taking its
    last number as run_once does. Label-free; reported next to the harness score."""
    ans = [s for s in rec["steps"] if s["agent"] == "Worker" and s["content"].startswith("ANSWER")]
    if not ans:
        return None
    arg = ans[-1]["content"][len("ANSWER"):].strip()
    arg = re.sub(r"^(Worker\s+)?CALC\s*:?\s*", "", arg, flags=re.I)
    v = RA._calc(arg.split("=")[0])
    return v if v is not None else RA._last_num(arg)


def is_fail_harness(rec: dict, gold: float) -> bool:
    return rec["outcome"] == "failure"


def is_fail_evaluated(rec: dict, gold: float) -> bool:
    v = evaluated_final(rec)
    return v is None or abs(v - gold) >= 1e-6


def rollout(llm, cond: str, task_idx: int, i: int, temperature: float, max_turns: int) -> dict:
    task, answer = CANDIDATE_TASKS[task_idx]
    seed = det_seed("exp3", cond, task_idx, i)
    v = llm.view(seed=seed, tag=f"exp3|{cond}|t{task_idx:02d}|r{i:02d}")
    tr, ok, _, _ = RA.run_once(v, task, float(answer), temperature=temperature, max_worker_turns=max_turns)
    rid = f"int_{cond}_t{task_idx:02d}_r{i:02d}"
    rec = trace_to_json(tr, rid, ok)
    rec.update({"seed_base": seed, "seeds_used": v.seeds_used, "n_calls": v.usage.calls,
                "tokens": v.usage.prompt_tokens + v.usage.completion_tokens,
                "response_models": sorted(v.response_models)})
    return rec


def run_condition(llm, cond: str, tasks: List[int], n: int, temperature: float, max_turns: int,
                  workers: int, store: dict, save_cb=None) -> None:
    """All rollouts of one condition, with the condition's prompts patched in for the
    whole batch (module globals are read at call time, so conditions never overlap)."""
    P = prompts_for(cond)
    todo = [(t, i) for t in tasks for i in range(n) if f"int_{cond}_t{t:02d}_r{i:02d}" not in store]
    print(f"[{cond}] {len(todo)} rollouts to run", flush=True)
    t0 = time.time()
    with mock.patch.object(RA, "PLANNER_SYS", P["PLANNER_SYS"]), \
            mock.patch.object(RA, "WORKER_SYS", P["WORKER_SYS"]):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(rollout, llm, cond, t, i, temperature, max_turns): (t, i) for t, i in todo}
            for k, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                store[rec["run_id"]] = rec
                if k % 20 == 0 or k == len(todo):
                    print(f"[{cond}] {k}/{len(todo)} rollouts, {time.time() - t0:.0f}s", flush=True)
                    if save_cb:
                        save_cb()


def rates(store: dict, tasks: List[int], n: int, is_fail) -> dict:
    out = {"per_task": {}, "pooled": {}}
    tot = {c: [0, 0] for c in CONDITIONS}
    for t in tasks:
        gold = float(CANDIDATE_TASKS[t][1])
        row = {}
        for c in CONDITIONS:
            runs = [store[f"int_{c}_t{t:02d}_r{i:02d}"] for i in range(n) if f"int_{c}_t{t:02d}_r{i:02d}" in store]
            fl = [is_fail(r, gold) for r in runs]
            f = int(sum(fl))
            row[c] = {"fail": f, "n": len(runs), **{k: v for k, v in S.acc_record(fl).items() if k != "c"}}
            tot[c][0] += f; tot[c][1] += len(runs)
        b = row["baseline"]
        pv = {}
        for c in ("targeted", "placebo"):
            fi = S.fisher_2x2(b["fail"], b["n"], row[c]["fail"], row[c]["n"])
            row[f"baseline_vs_{c}"] = {**fi, **S.diff_ci(b["fail"], b["n"], row[c]["fail"], row[c]["n"])}
            pv[c] = fi["p_value"]
        adj = S.holm(pv)
        for c in ("targeted", "placebo"):
            row[f"baseline_vs_{c}"]["p_holm"] = adj[c]["p_holm"]
            row[f"baseline_vs_{c}"]["holm_family"] = f"task t{t:02d}: baseline vs targeted, baseline vs placebo (m=2)"
        out["per_task"][f"t{t:02d}"] = row
    pooled = {}
    for c in CONDITIONS:
        f, nn = tot[c]
        lo, hi = S.wilson(f, nn)
        pooled[c] = {"fail": f, "n": nn, "acc": (f / nn if nn else 0.0), "wilson_lo": lo, "wilson_hi": hi}
    pv = {}
    for c in ("targeted", "placebo"):
        fi = S.fisher_2x2(tot["baseline"][0], tot["baseline"][1], tot[c][0], tot[c][1])
        pooled[f"baseline_vs_{c}"] = {**fi, **S.diff_ci(tot["baseline"][0], tot["baseline"][1], tot[c][0], tot[c][1])}
        pv[c] = fi["p_value"]
    adj = S.holm(pv)
    for c in ("targeted", "placebo"):
        pooled[f"baseline_vs_{c}"]["p_holm"] = adj[c]["p_holm"]
        pooled[f"baseline_vs_{c}"]["holm_family"] = "pooled: baseline vs targeted, baseline vs placebo (m=2)"
    # targeted vs placebo (secondary, single test)
    pooled["targeted_vs_placebo"] = {**S.fisher_2x2(tot["targeted"][0], tot["targeted"][1], tot["placebo"][0], tot["placebo"][1]),
                                     **S.diff_ci(tot["targeted"][0], tot["targeted"][1], tot["placebo"][0], tot["placebo"][1])}
    out["pooled"] = pooled
    return out


def analyse(store: dict, tasks: List[int], n: int) -> dict:
    out = {**rates(store, tasks, n, is_fail_harness), "phylo": {}, "failure_modes": {},
           "scoring_note": "per_task/pooled use the harness definition of failure (run_once: last number of the "
                           "ANSWER line vs gold). evaluated_answer_scoring re-scores each run by evaluating an "
                           "arithmetic expression carried in the ANSWER line before comparing (label-free "
                           "sensitivity analysis; not the system's own success criterion).",
           "evaluated_answer_scoring": rates(store, tasks, n, is_fail_evaluated)}
    # harness-vs-evaluated disagreement per condition
    out["scoring_disagreement"] = {}
    for c in CONDITIONS:
        dis = tot_n = 0
        for t in tasks:
            gold = float(CANDIDATE_TASKS[t][1])
            for i in range(n):
                k = f"int_{c}_t{t:02d}_r{i:02d}"
                if k in store:
                    tot_n += 1
                    dis += is_fail_harness(store[k], gold) != is_fail_evaluated(store[k], gold)
        out["scoring_disagreement"][c] = {"runs_scored_differently": dis, "n": tot_n}

    # RefAlign on the new failures, per condition (label-free: distribution only)
    for c in CONDITIONS:
        steps, agents, kinds, modes, ref_avail = Counter(), Counter(), Counter(), Counter(), {"with_ref": 0, "without_ref": 0}
        per_run = {}
        for t in tasks:
            task, _ = CANDIDATE_TASKS[t]
            runs = [store[f"int_{c}_t{t:02d}_r{i:02d}"] for i in range(n) if f"int_{c}_t{t:02d}_r{i:02d}" in store]
            traces = [json_to_trace(r, f"int_t{t:02d}", task) for r in runs]
            pop = RunPopulation(f"int_t{t:02d}", traces, task)
            for r, tr in zip(runs, traces):
                if tr.success:
                    continue
                modes[failure_mode(r)] += 1
                if not pop.successes:
                    ref_avail["without_ref"] += 1
                    continue
                ref_avail["with_ref"] += 1
                p = contrastive_localize(pop, tr)
                steps[p.step] += 1; agents[p.agent] += 1
                kinds[p.meta.get("kind") or p.meta.get("fallback")] += 1
                per_run[tr.run_id] = {"agent_pred": p.agent, "step_pred": int(p.step),
                                      "kind": p.meta.get("kind") or p.meta.get("fallback"),
                                      "sibling": p.meta.get("sibling"), "failure_mode": failure_mode(r)}
        out["phylo"][c] = {"n_failures": sum(modes.values()), "reference_availability": ref_avail,
                           "step_distribution": dict(sorted(steps.items())),
                           "agent_distribution": dict(agents), "kind_distribution": dict(kinds),
                           "per_run": per_run}
        out["failure_modes"][c] = dict(modes)
    return out


def write_tex(res: dict, path: str, model_label: str) -> None:
    tex = _tex_panel(res, model_label, "intervention",
                     "harness scoring (run\\_once: last number of the ANSWER line vs.\\ gold, as in the human-benchmark collection)")
    tex += "\n" + _tex_panel(res["evaluated_answer_scoring"] | {"per_task": res["evaluated_answer_scoring"]["per_task"]}, model_label,
                              "intervention-evaluated",
                              "evaluated-answer scoring (an arithmetic expression in the ANSWER line is evaluated before comparison; label-free sensitivity analysis)")
    open(path, "w").write(tex)


def _tex_panel(res: dict, model_label: str, label: str, scoring: str) -> str:
    rows = []
    for t, row in res["per_task"].items():
        tag = f"\\texttt{{{t}}}" + ("$^\\dagger$" if int(t[1:]) in RETAINED else "")
        cells = [f"{row[c]['fail']}/{row[c]['n']}" for c in CONDITIONS]
        bt, bp = row["baseline_vs_targeted"], row["baseline_vs_placebo"]
        rows.append(f"{tag} & {' & '.join(cells)} & {bt['diff']:+.2f} [{bt['ci_lo']:+.2f}, {bt['ci_hi']:+.2f}] & "
                    f"{S.fmt_p(bt['p_value'])} / {S.fmt_p(bt['p_holm'])} & "
                    f"{bp['diff']:+.2f} [{bp['ci_lo']:+.2f}, {bp['ci_hi']:+.2f}] & {S.fmt_p(bp['p_value'])} / {S.fmt_p(bp['p_holm'])} \\\\")
    P = res["pooled"]
    bt, bp = P["baseline_vs_targeted"], P["baseline_vs_placebo"]
    pooled = (f"pooled & " + " & ".join(f"{P[c]['fail']}/{P[c]['n']} [{P[c]['wilson_lo']:.2f}, {P[c]['wilson_hi']:.2f}]" for c in CONDITIONS) +
              f" & {bt['diff']:+.2f} [{bt['ci_lo']:+.2f}, {bt['ci_hi']:+.2f}] & {S.fmt_p(bt['p_value'])} / {S.fmt_p(bt['p_holm'])} & "
              f"{bp['diff']:+.2f} [{bp['ci_lo']:+.2f}, {bp['ci_hi']:+.2f}] & {S.fmt_p(bp['p_value'])} / {S.fmt_p(bp['p_holm'])} \\\\")
    tex = f"""% Experiment 3: intervention at the attributed step vs a placebo elsewhere.
\\begin{{table*}}[t]
\\centering\\scriptsize
\\begin{{tabular}}{{lccccccc}}
\\toprule
 & \\multicolumn{{3}}{{c}}{{failures / rollouts}} & \\multicolumn{{2}}{{c}}{{targeted $-$ baseline}} & \\multicolumn{{2}}{{c}}{{placebo $-$ baseline}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-6}} \\cmidrule(lr){{7-8}}
task & baseline & targeted (Worker) & placebo (Planner) & $\\Delta$ fail rate [95\\% CI] & $p$ / $p_{{\\mathrm{{Holm}}}}$ & $\\Delta$ fail rate [95\\% CI] & $p$ / $p_{{\\mathrm{{Holm}}}}$ \\\\
\\midrule
{chr(10).join(rows)}
\\midrule
{pooled}
\\bottomrule
\\end{{tabular}}
\\caption{{Failure rates of the Planner$\\to$Worker$\\to$Verifier system under the original prompts, a
one-sentence verification instruction added to the Worker prompt (the step the human labels
attribute), and an equally long instruction added to the Planner prompt (placebo). All three
conditions were regenerated with identical sampling ({model_label}, temperature $1.0$,
{P['baseline']['n'] // max(1, len(res['per_task']))} rollouts per task and condition, fixed seeds).
$\\dagger$: the four families of the human benchmark. Pooled rows give 95\\% Wilson intervals.
$\\Delta$ is the difference in failure rate with a Newcombe score interval; $p$ is a two-sided
Fisher exact test and $p_{{\\mathrm{{Holm}}}}$ the Holm--Bonferroni value within the family of the two
comparisons of that row ($m{{=}}2$). Scoring: {scoring}.}}
\\label{{tab:{label}}}
\\end{{table*}}
"""
    return tex


def write_prompts_md(path: str, res_meta: dict) -> None:
    base = prompts_for("baseline")
    parts = ["# Experiment 3 prompt conditions (exact text)\n",
             f"Agent model: `{res_meta['model']}` (revision `{res_meta['revision']}`), temperature {res_meta['temperature']}, "
             f"max_tokens {res_meta['max_tokens']}, {res_meta['n_rollouts']} rollouts per task per condition, seeds `det_seed('exp3', condition, task_idx, i)`.\n",
             f"Targeted instruction ({len(TARGETED_INSTRUCTION)} chars, {len(TARGETED_INSTRUCTION.split())} words) is appended to `WORKER_SYS`; "
             f"placebo instruction ({len(PLACEBO_INSTRUCTION)} chars, {len(PLACEBO_INSTRUCTION.split())} words) is appended to `PLANNER_SYS`. The Verifier prompt is never changed.\n"]
    for cond in ("targeted", "placebo"):
        P = prompts_for(cond)
        for key in ("PLANNER_SYS", "WORKER_SYS"):
            if P[key] != base[key]:
                diff = difflib.unified_diff(base[key].splitlines(), P[key].splitlines(),
                                            fromfile=f"baseline/{key}", tofile=f"{cond}/{key}", lineterm="")
                parts.append(f"## {cond}: diff of `{key}`\n\n```diff\n" + "\n".join(diff) + "\n```\n")
    parts.append("## Full baseline prompts\n")
    for key, val in base.items():
        parts.append(f"### `{key}`\n\n```\n{val}\n```\n")
    open(path, "w").write("\n".join(parts))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8103/v1")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="the human-benchmark collection used collect_traces --temperature 1.0")
    ap.add_argument("--max-tokens", type=int, default=256, help="APILLM default used by the original collection")
    ap.add_argument("--max-turns", type=int, default=5, help="run_once default max_worker_turns")
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--tasks", default="retained", help="'retained' (t00,t05,t07,t09), 'all' (10 candidates) or comma list")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="results/new/intervention.json")
    ap.add_argument("--traces", default="results/new/intervention_traces.json")
    ap.add_argument("--tex", default="results/new/tab_intervention.tex")
    ap.add_argument("--prompts-md", default="results/new/intervention_prompts.md")
    ap.add_argument("--log", default="results/new/api_log.jsonl")
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.base_url or args.base_url.lower() in ("none", "openai"):
        args.base_url = None            # the OpenAI API itself (key from .env)

    tasks = RETAINED if args.tasks == "retained" else (list(range(len(CANDIDATE_TASKS))) if args.tasks == "all"
                                                      else [int(x) for x in args.tasks.split(",")])
    assert abs(len(TARGETED_INSTRUCTION) - len(PLACEBO_INSTRUCTION)) <= 6, \
        (len(TARGETED_INSTRUCTION), len(PLACEBO_INSTRUCTION))
    store = json.load(open(args.traces)) if os.path.exists(args.traces) else {}
    meta = {"model": args.model, "revision": args.revision, "base_url": args.base_url,
            "temperature": args.temperature, "max_tokens": args.max_tokens, "max_worker_turns": args.max_turns,
            "n_rollouts": args.n, "tasks": tasks, "retained_families": RETAINED,
            "task_text": {f"t{t:02d}": CANDIDATE_TASKS[t] for t in tasks},
            "prompt_lengths": {"targeted_chars": len(TARGETED_INSTRUCTION), "placebo_chars": len(PLACEBO_INSTRUCTION)},
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if not args.analyse_only:
        llm = LoggedLLM(args.model, base_url=args.base_url, temperature=args.temperature,
                        max_tokens=args.max_tokens, log_path=args.log, tag="exp3", revision=args.revision)

        def save():
            json.dump(store, open(args.traces, "w"), indent=0)
        for cond in CONDITIONS:
            run_condition(llm, cond, tasks, args.n, args.temperature, args.max_turns, args.workers, store, save)
            save()
        meta["usage"] = {"calls": llm.usage.calls, "prompt_tokens": llm.usage.prompt_tokens,
                         "completion_tokens": llm.usage.completion_tokens, "seconds": round(llm.usage.seconds, 1),
                         "errors": llm.n_errors, "usd": 0.0 if args.base_url else None}
        meta["response_models"] = sorted({m for r in store.values() for m in r.get("response_models", [])})
    meta["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    res = {"design": meta, **analyse(store, tasks, args.n)}
    json.dump(res, open(args.out, "w"), indent=1, default=float)
    write_tex(res, args.tex, args.model.split("/")[-1].replace("_", "\\_"))
    write_prompts_md(args.prompts_md, meta)
    print("\n=== failures / rollouts ===")
    for t, row in res["per_task"].items():
        print(f"  {t}: " + "  ".join(f"{c}={row[c]['fail']}/{row[c]['n']}" for c in CONDITIONS) +
              f"  p(t)={row['baseline_vs_targeted']['p_value']:.3f} p(p)={row['baseline_vs_placebo']['p_value']:.3f}")
    P = res["pooled"]
    print("  pooled: " + "  ".join(f"{c}={P[c]['fail']}/{P[c]['n']} [{P[c]['wilson_lo']:.2f},{P[c]['wilson_hi']:.2f}]" for c in CONDITIONS))
    print(f"  baseline vs targeted: diff={P['baseline_vs_targeted']['diff']:+.3f} p={P['baseline_vs_targeted']['p_value']:.4f} holm={P['baseline_vs_targeted']['p_holm']:.4f}")
    print(f"  baseline vs placebo : diff={P['baseline_vs_placebo']['diff']:+.3f} p={P['baseline_vs_placebo']['p_value']:.4f} holm={P['baseline_vs_placebo']['p_holm']:.4f}")
    for c in CONDITIONS:
        ph = res["phylo"][c]
        print(f"  phylo[{c}]: n_fail={ph['n_failures']} ref={ph['reference_availability']} steps={ph['step_distribution']} kinds={ph['kind_distribution']} modes={res['failure_modes'][c]}")
    print(f"wrote {args.out}, {args.traces}, {args.tex}, {args.prompts_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
