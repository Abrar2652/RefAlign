"""Run attribution baselines on the REAL Who&When benchmark.

Heuristic baselines (no LLM) + the Who&When LLM baselines (All-at-Once,
Step-by-Step, Binary-Search) via an API model. RefAlign itself is NOT run here:
Who&When provides one failing log per query, so there is no multi-run population
to build a phylogeny over (see run_real_populations.py for RefAlign on real
multi-run traces).

Usage:
    python -m experiments.run_real_whoandwhen --subset algo --n 20 --model gpt-4o-mini \
        --methods all_at_once binary_search --out results/real_ww_algo.json
"""
import argparse
import json
import sys
import numpy as np

from refalign.data.schema import load_who_and_when_parquet
from refalign.types import RunPopulation
from refalign.baselines import (random_baseline, first_step_baseline,
                                  last_step_baseline, most_active_agent_baseline,
                                  longest_agent_baseline)
from refalign.llm import APILLM
from refalign.llm_baselines import LLM_METHODS
from refalign import evaluate as ev

HEUR = {"random": random_baseline, "first_step": first_step_baseline,
        "last_step": last_step_baseline, "most_active_agent": most_active_agent_baseline,
        "longest_agent": longest_agent_baseline}

PARQUET = {"algo": "data_real/Algorithm-Generated.parquet",
           "hand": "data_real/Hand-Crafted.parquet"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["algo", "hand"], default="algo")
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--methods", nargs="*", default=["all_at_once", "binary_search"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    traces, anns = load_who_and_when_parquet(PARQUET[args.subset],
                                             subset=(args.n or None))
    print(f"loaded {len(traces)} {args.subset} traces", flush=True)

    named = {}
    # heuristic baselines (single-trace; wrap each as a 1-run population)
    rng = np.random.RandomState(0)
    for name, fn in HEUR.items():
        named[name] = {t.run_id: fn(RunPopulation(t.query_id, [t]), t, rng)
                       for t in traces}

    # LLM baselines
    usage = {}
    if args.methods:
        llm = APILLM(provider=args.provider, model=args.model)
        for m in args.methods:
            preds = {}
            for i, t in enumerate(traces):
                preds[t.run_id] = LLM_METHODS[m](t, llm)
                if (i + 1) % 20 == 0:
                    print(f"  {m}: {i+1}/{len(traces)}  ({llm.usage.calls} calls, "
                          f"{llm.usage.seconds:.0f}s)", flush=True)
            named[m] = preds
        usage = {"calls": llm.usage.calls, "prompt_tokens": llm.usage.prompt_tokens,
                 "completion_tokens": llm.usage.completion_tokens,
                 "seconds": llm.usage.seconds, "model": args.model}

    rows = ev.leaderboard(named, anns, tolerances=(1,), B=2000)
    print(f"\n=== Real Who&When ({args.subset}, n={len(traces)}, model={args.model}) ===")
    print(ev.format_leaderboard(rows, tolerances=(1,)))
    if usage:
        print(f"\nLLM cost: {usage['calls']} calls, "
              f"{usage['prompt_tokens']}+{usage['completion_tokens']} tokens, "
              f"{usage['seconds']:.0f}s")

    if args.out:
        out = {"subset": args.subset, "n": len(traces), "model": args.model,
               "usage": usage,
               "leaderboard": [{"name": r.name, "summary": r.summary, "ci": r.ci}
                               for r in rows]}
        json.dump(out, open(args.out, "w"), indent=2, default=float)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
