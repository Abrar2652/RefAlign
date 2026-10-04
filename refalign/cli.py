"""RefAlign command-line entry point.

    python -m refalign.cli demo        [--regime algo]
    python -m refalign.cli ablations   [--regime algo --seeds 0 1 2 --out r.json]
    python -m refalign.cli evaluate    [--who-when PATH]
    python -m refalign.cli signal      [--regime algo]

`ablations` is the headline command: it runs the full method, all baselines, and
the four component ablations over multi-seed corpora, then prints Who&When
metrics with 95% bootstrap CIs and paired significance tests vs the full method.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

import numpy as np

from .data.synth import SyntheticConfig, generate_corpus
from .data.schema import load_who_and_when
from .attribute import AttributionConfig, attribute_corpus
from .baselines import run_baseline
from .encode import encode_population
from .evostat import failure_signal_permutation
from . import evaluate as ev
from .ablation import run_suite


# ---------------------------------------------------------------------------
def cmd_demo(args) -> int:
    pops, anns = generate_corpus(SyntheticConfig(n_queries=3, runs_per_query=8,
                                                 regime=args.regime, seed=0))
    cfg = AttributionConfig()
    preds = attribute_corpus(pops, cfg)
    sc = ev.score(preds, anns)
    print(f"RefAlign demo on synthetic '{args.regime}' corpus")
    print(f"  populations={len(pops)}  failing runs={sc.n}")
    print(f"  {json.dumps(sc.summary())}")
    for rid in sorted(preds)[:8]:
        p, g = preds[rid], anns[rid]
        ok_a = "OK" if p.agent == g.responsible_agent else "  "
        ok_s = "OK" if p.step == g.decisive_step else "  "
        print(f"  {rid}: pred(agent={p.agent:10s} step={p.step:2d}) "
              f"gt(agent={g.responsible_agent:10s} step={g.decisive_step:2d}) "
              f"[{ok_a} agent | {ok_s} step] {p.meta.get('kind','')}")
    return 0


def _pooled_scored(suite, name) -> ev.Scored:
    return ev.score(suite.named_preds[name], suite.anns)


def cmd_ablations(args) -> int:
    seeds = list(args.seeds)
    suite = run_suite(regime=args.regime, seeds=seeds, n_queries=args.queries,
                      runs_per_query=args.runs)

    # ---- main leaderboard: full method + baselines ----
    main_methods = ["RefAlign (full)"] + [m for m in suite.named_preds
                                            if suite.groups[m] == "baseline"]
    rows = ev.leaderboard({m: suite.named_preds[m] for m in main_methods},
                          suite.anns, tolerances=(1, 2))
    print(f"\n=== RefAlign vs baselines — regime={args.regime}, "
          f"seeds={seeds}, {int(rows[0].summary['n'])} failing runs ===")
    print(ev.format_leaderboard(rows, tolerances=(1, 2)))

    # ---- ablation groups with paired comparison vs full ----
    full = _pooled_scored(suite, "RefAlign (full)")
    results: Dict[str, dict] = {"leaderboard": [r.__dict__ for r in rows],
                                "ablations": {}}
    order = ["a_alignment", "b_tree", "c_geometry", "d_evostat", "localizer"]
    titles = {"a_alignment": "(a) alignment", "b_tree": "(b) tree inference",
              "c_geometry": "(c) geometry vs sequence distance",
              "d_evostat": "(d) evolutionary rule", "localizer": "localizer contrast"}
    step_pvals: Dict[str, float] = {}   # family for Holm correction
    for grp in order:
        members = ["RefAlign (full)"] + [m for m, g in suite.groups.items()
                                           if g == grp]
        grows = ev.leaderboard({m: suite.named_preds[m] for m in members},
                               suite.anns, tolerances=(1,))
        print(f"\n--- Ablation {titles[grp]} ---")
        print(ev.format_leaderboard(grows, tolerances=(1,)))
        print("  paired vs full (Δ step-acc, McNemar p, Cohen's h):")
        grp_out = []
        for m in members:
            if m == "RefAlign (full)":
                continue
            other = _pooled_scored(suite, m)
            ca = ev.compare(full, other, "agent")
            cs = ev.compare(full, other, "step")
            step_pvals[m] = cs["mcnemar_p_value"]
            print(f"    vs {m:24s}  Δstep={cs['diff']:+.3f} "
                  f"(p={cs['mcnemar_p_value']:.3f}, h={cs['cohens_h']:+.2f})")
            grp_out.append({"method": m, "agent": ca, "step": cs})
        results["ablations"][grp] = grp_out

    # ---- Holm-Bonferroni correction over the whole ablation family ----
    holm = ev.holm_bonferroni(step_pvals)
    print("\n--- Holm-Bonferroni corrected step-acc comparisons vs full ---")
    for name, h in sorted(holm.items(), key=lambda kv: kv[1]["p_holm"]):
        mark = "*" if h["reject"] else " "
        print(f"  {mark} {name:24s} p_raw={h['p_raw']:.3f}  p_holm={h['p_holm']:.3f}")
    results["holm_step"] = holm

    # ---- phylogenetic signal of the failure trait (characterisation) ----
    sig = _signal_summary(args.regime, seeds[0], args.queries, args.runs)
    print(f"\n--- Phylogenetic signal of the failure trait (seed {seeds[0]}) ---")
    print(f"  populations tested={sig['n']}  significant clustering (p<0.05): "
          f"{sig['frac_significant']:.2f}  median p={sig['median_p']:.3f}")
    results["failure_signal"] = sig

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(_jsonable(results), fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


def _signal_summary(regime, seed, n_queries, runs_per_query) -> dict:
    pops, _ = generate_corpus(SyntheticConfig(n_queries=min(n_queries, 20),
                                              runs_per_query=runs_per_query,
                                              regime=regime, seed=seed))
    ps: List[float] = []
    for pop in pops:
        alpha, seqs = encode_population(pop)
        r = failure_signal_permutation(pop, alpha, seqs, n_perm=500, seed=seed)
        if not np.isnan(r["p_value"]):
            ps.append(r["p_value"])
    ps_arr = np.array(ps) if ps else np.array([1.0])
    return {"n": len(ps), "frac_significant": float((ps_arr < 0.05).mean()),
            "median_p": float(np.median(ps_arr))}


def cmd_signal(args) -> int:
    sig = _signal_summary(args.regime, 0, 20, 10)
    print(f"Phylogenetic signal ({args.regime}): {json.dumps(sig)}")
    return 0


def cmd_evaluate(args) -> int:
    if args.who_when:
        pops, anns = load_who_and_when(args.who_when)
        print(f"loaded Who&When: {len(pops)} queries, {len(anns)} annotations")
    else:
        pops, anns = generate_corpus(SyntheticConfig(regime=args.regime, seed=0))
    named = {"RefAlign (full)": attribute_corpus(pops, AttributionConfig())}
    for b in ["last_step", "most_active_agent", "edit_nn"]:
        named[b] = run_baseline(b, pops, seed=0)
    rows = ev.leaderboard(named, anns, tolerances=(1, 2))
    print(ev.format_leaderboard(rows, tolerances=(1, 2)))
    return 0


# ---------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="refalign", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="quick end-to-end on a synthetic corpus")
    d.add_argument("--regime", default="algo", choices=["algo", "hand"])
    d.set_defaults(func=cmd_demo)

    a = sub.add_parser("ablations", help="full method + baselines + 4 ablations")
    a.add_argument("--regime", default="algo", choices=["algo", "hand"])
    a.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a.add_argument("--queries", type=int, default=30)
    a.add_argument("--runs", type=int, default=10)
    a.add_argument("--out", default=None)
    a.set_defaults(func=cmd_ablations)

    e = sub.add_parser("evaluate", help="leaderboard on synthetic or Who&When")
    e.add_argument("--regime", default="algo", choices=["algo", "hand"])
    e.add_argument("--who-when", default=None, help="path to Who&When JSON dir/file")
    e.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("signal", help="phylogenetic signal of the failure trait")
    s.add_argument("--regime", default="algo", choices=["algo", "hand"])
    s.set_defaults(func=cmd_signal)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
