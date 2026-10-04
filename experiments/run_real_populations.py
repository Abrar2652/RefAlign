"""RefAlign on REAL multi-agent LLM traces.

Builds real-trace populations (experiments/real_agents.py) — many runs per task
from an actual Planner->Worker->Verifier system, with AEGIS-injected faults for
ground truth — then runs RefAlign and the baselines, reporting agent/step
accuracy overall AND broken down by fault type. The breakdown surfaces the key
honest finding: RefAlign's (agent, action) abstraction catches *structural*
faults (premature answer, wrong tool) but not *content* faults (a calc with the
right action but wrong arithmetic), where a content-reading LLM judge is needed.

Usage: python -m experiments.run_real_populations --tasks 12 --runs 8 --out results/real_pop.json
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np

from refalign.llm import APILLM
from refalign.attribute import AttributionConfig, attribute_corpus
from refalign.baselines import run_baseline
from refalign.llm_baselines import all_at_once
from refalign import evaluate as ev
from experiments.real_agents import build_population, TASKS


def _serialize(pops, anns):
    return {
        "annotations": {rid: {"agent": a.responsible_agent, "step": a.decisive_step,
                              "fault_type": a.meta.get("fault_type")} for rid, a in anns.items()},
        "populations": [{"query_id": p.query_id, "query": p.query,
                         "runs": [{"run_id": r.run_id, "success": r.success,
                                   "steps": [{"agent": s.agent, "action": s.action,
                                              "content": s.content} for s in r.steps]}
                                  for r in p.runs]} for p in pops],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--n_fault", type=int, default=4)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default="results/real_pop.json")
    ap.add_argument("--corpus_out", default="results/real_pop_corpus.json")
    args = ap.parse_args(argv)

    gen = APILLM(provider="openai", model=args.model, temperature=args.temperature)
    pops, anns = [], {}
    for ti in range(min(args.tasks, len(TASKS))):
        pop, a = build_population(gen, ti, n_runs=args.runs, n_fault=args.n_fault,
                                  temperature=args.temperature)
        pops.append(pop)
        anns.update(a)
        n_succ = len(pop.successes)
        print(f"  task{ti:02d}: {len(pop.runs)} runs ({n_succ} healthy), "
              f"{len(a)} labeled faults  [{gen.usage.calls} calls, {gen.usage.seconds:.0f}s]",
              flush=True)
    json.dump(_serialize(pops, anns), open(args.corpus_out, "w"), indent=1, default=str)

    # attributors
    named = {"RefAlign (full)": attribute_corpus(pops, AttributionConfig())}
    for b in ["random", "last_step", "most_active_agent", "longest_agent"]:
        # baselines expect populations; run over failures, keep only labeled ones
        named[b] = run_baseline(b, pops, seed=0)
    judge = APILLM(provider="openai", model=args.model, temperature=0.0)
    aao = {}
    for pop in pops:
        for f in pop.failures:
            if f.run_id in anns:
                aao[f.run_id] = all_at_once(f, judge)
    named["llm_all_at_once"] = aao

    # overall leaderboard (labeled faults only)
    rows = ev.leaderboard({k: {r: p for r, p in v.items() if r in anns}
                           for k, v in named.items()}, anns, tolerances=(1,), B=2000)
    print(f"\n=== RefAlign on REAL agent traces ({len(anns)} labeled faults) ===")
    print(ev.format_leaderboard(rows, tolerances=(1,)))

    # per-fault-type agent/step accuracy for RefAlign vs the LLM judge
    by_type = defaultdict(list)
    for rid, a in anns.items():
        by_type[a.meta.get("fault_type", "?")].append(rid)
    print("\nper-fault-type accuracy (agent / step):")
    print(f"  {'fault_type':12s} {'n':>3s}  {'RefAlign':>18s}  {'llm_all_at_once':>18s}")
    ftype_res = {}
    for ft, rids in sorted(by_type.items()):
        def acc(preds):
            a = np.mean([preds[r].agent == anns[r].responsible_agent for r in rids if r in preds])
            s = np.mean([preds[r].step == anns[r].decisive_step for r in rids if r in preds])
            return a, s
        pa, ps = acc(named["RefAlign (full)"])
        la, ls = acc(aao)
        print(f"  {ft:12s} {len(rids):3d}  {pa:.2f} / {ps:.2f}          {la:.2f} / {ls:.2f}")
        ftype_res[ft] = {"n": len(rids), "phylo": [float(pa), float(ps)],
                         "llm": [float(la), float(ls)]}

    usage = {"gen_calls": gen.usage.calls, "gen_tokens": gen.usage.prompt_tokens + gen.usage.completion_tokens,
             "judge_calls": judge.usage.calls, "seconds": gen.usage.seconds + judge.usage.seconds}
    out = {"n_faults": len(anns), "model": args.model, "usage": usage,
           "leaderboard": [{"name": r.name, "summary": r.summary, "ci": r.ci} for r in rows],
           "by_fault_type": ftype_res}
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\ncost: {usage}\nwrote {args.out} and {args.corpus_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
