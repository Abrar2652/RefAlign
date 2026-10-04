"""Gap-closing experiments: multi-culprit agent attribution, fault-type coverage,
Who&When SOTA-adjacent LLM baselines on the same benchmark, and solution diversity.
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np

from refalign.llm import APILLM
from refalign.contrastive import contrastive_localize, ContrastiveConfig
from refalign.llm_baselines import all_at_once, step_by_step, binary_search
from refalign.baselines import run_baseline
from experiments.bench import build_multiculprit, build_diverse


def _acc(preds, anns, rids=None):
    rids = rids or [r for r in preds if r in anns]
    a = np.mean([preds[r].agent == anns[r].responsible_agent for r in rids]) if rids else 0
    s = np.mean([preds[r].step == anns[r].decisive_step for r in rids]) if rids else 0
    return float(a), float(s), len(rids)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_length", type=int, default=12)
    ap.add_argument("--tasks", type=int, default=6)
    ap.add_argument("--n_fault", type=int, default=8)
    ap.add_argument("--contrastive_lengths", type=int, nargs="+", default=[5, 12, 24])
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--out", default="results/gaps.json")
    args = ap.parse_args(argv)
    out = {}

    # ---- multi-culprit: contrastive across lengths (free) ----
    out["multiculprit_contrastive"] = {}
    for L in args.contrastive_lengths:
        pops, anns = [], {}
        for t in range(args.tasks):
            p, a = build_multiculprit(t, L, seed=0, n_fault=args.n_fault)
            pops.append(p); anns.update(a)
        preds = {f.run_id: contrastive_localize(p, f) for p in pops for f in p.failures}
        a, s, n = _acc(preds, anns)
        bytype = defaultdict(list)
        for r in preds:
            if r in anns: bytype[anns[r].meta["fault_type"]].append(r)
        per = {k: _acc(preds, anns, v)[1] for k, v in bytype.items()}
        out["multiculprit_contrastive"][f"L{L}"] = {"agent": a, "step": s, "n": n,
                                                     "per_fault_step": per}
        print(f"[multiculprit] L={L}: agent={a:.2f} step={s:.2f} n={n} per-type={ {k:round(v,2) for k,v in per.items()} }", flush=True)

    # ---- multi-culprit at one length: contrastive vs LLM baselines + heuristics ----
    L = args.llm_length
    pops, anns = [], {}
    for t in range(args.tasks):
        p, a = build_multiculprit(t, L, seed=0, n_fault=args.n_fault)
        pops.append(p); anns.update(a)
    fails = [f for p in pops for f in p.failures if f.run_id in anns]
    judge = APILLM(provider="openai", model=args.model, temperature=0.0)

    methods = {"contrastive (no LLM)": {f.run_id: contrastive_localize(
                    [p for p in pops if f in p.runs][0], f) for f in fails}}
    for name in ["most_active_agent", "last_step"]:
        methods[name] = run_baseline(name, pops, seed=0)
    print(f"[LLM baselines] running on L={L}, {len(fails)} faults ...", flush=True)
    for name, fn in [("all_at_once", all_at_once), ("binary_search", binary_search),
                     ("step_by_step", step_by_step)]:
        methods[name] = {f.run_id: fn(f, judge) for f in fails}
        print(f"  {name}: {judge.usage.calls} calls, {judge.usage.seconds:.0f}s", flush=True)

    row = {}
    for name, pr in methods.items():
        a, s, n = _acc(pr, anns)
        row[name] = {"agent": a, "step": s, "n": n}
    row["_tokens"] = judge.usage.prompt_tokens + judge.usage.completion_tokens
    row["_calls"] = judge.usage.calls
    out["multiculprit_vs_llm"] = {"length": L, "methods": row}
    print(f"\n=== multi-culprit L={L}: contrastive vs LLM baselines ===")
    print(f"  {'method':22s} {'agent':>6s} {'step':>6s}")
    for name in ["contrastive (no LLM)", "all_at_once", "binary_search", "step_by_step",
                 "most_active_agent", "last_step"]:
        r = row[name]; print(f"  {name:22s} {r['agent']:6.2f} {r['step']:6.2f}")

    # ---- solution diversity: content-aware vs edit sibling ----
    out["diversity"] = {}
    for k in [8, 15]:
        pops, anns = [], {}
        for t in range(args.tasks):
            p, a = build_diverse(t, k, seed=0); pops.append(p); anns.update(a)
        for mode in ["edit", "content"]:
            cfg = ContrastiveConfig(sibling=mode)
            preds = {f.run_id: contrastive_localize(p, f, cfg) for p in pops for f in p.failures}
            _, s, n = _acc(preds, anns)
            out["diversity"][f"k{k}_{mode}"] = {"step": s, "n": n}
    print("\n=== solution diversity (step-acc) ===")
    for k in [8, 15]:
        print(f"  k={k}: edit={out['diversity'][f'k{k}_edit']['step']:.2f}  "
              f"content-aware={out['diversity'][f'k{k}_content']['step']:.2f}")

    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\nLLM cost: {row['_calls']} calls, {row['_tokens']} tokens\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
