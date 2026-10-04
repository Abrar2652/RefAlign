"""Experiment 2 (LLM-free): sensitivity of RefAlign's predictions to the choice
of successful reference, on the four human-labeled real families.

For every failing run we force each available successful run to be the reference
(by monkeypatching ``_nearest_sibling``), record the predicted (agent, step), and
compare accuracy under the nearest reference (the method's own choice), the first
reference (``successes[0]``, the run the human annotators saw), the best and the
worst reference, and the mean over references (a uniformly random reference).
We also test whether references that follow the *same arithmetic path* as the
failing run recover the human step (the paper's explanation of the ``t07``
off-by-one), and correlate reference-to-failing distances with correctness.

    python -m experiments.run_reference_sensitivity
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple
from unittest import mock

import numpy as np
from scipy import stats

from refalign import contrastive as C
from refalign.contrastive import ContrastiveConfig, contrastive_localize, _encode_actions, \
    _nearest_sibling, _sibling_distance
from refalign.data.schema import load_annotated_benchmark
from refalign.types import Annotation, RunPopulation, Trace
from refalign import stats_new as S

_CALC = re.compile(r"CALC\s*(.+?)\s*=", re.I)


def load_human(traces="annotation/traces_scale", labels="annotation/annotations_human/adjudicated",
               perrun="results/human_perrun.json") -> Tuple[List[RunPopulation], Dict[str, Annotation]]:
    """Populations from the released traces; gold from ``human_perrun.json`` (the
    brief's source of truth), cross-checked against the adjudicated directory."""
    pops, adj = load_annotated_benchmark(traces, labels)
    anns: Dict[str, Annotation] = {}
    for r in json.load(open(perrun))["rows"]:
        anns[r["run_id"]] = Annotation(r["task"], r["run_id"], r["human_agent"], int(r["human_step"]))
    for rid, a in anns.items():
        if rid in adj and (adj[rid].responsible_agent, adj[rid].decisive_step) != (a.responsible_agent, a.decisive_step):
            raise RuntimeError(f"gold mismatch for {rid}: perrun {a} vs adjudicated {adj[rid]}")
    return pops, anns


def calc_exprs(tr: Trace) -> List[str]:
    """Normalised CALC expressions (left of '=') of the Worker's steps."""
    out = []
    for s in tr.steps:
        if s.agent != "Worker":
            continue
        m = _CALC.search(s.content or "")
        if m:
            e = re.sub(r"\s+", "", m.group(1))
            e = re.sub(r"(\d+)\.0+(?!\d)", r"\1", e)     # 90.0 -> 90
            out.append(e)
    return out


def path_overlap(f: Trace, r: Trace) -> float:
    """Jaccard overlap of the two runs' CALC-expression sets (1 = same arithmetic path)."""
    a, b = set(calc_exprs(f)), set(calc_exprs(r))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def forced_prediction(pop: RunPopulation, f: Trace, ref: Trace):
    """Run the unmodified contrastive_localize with ``ref`` forced as the sibling."""
    with mock.patch.object(C, "_nearest_sibling", lambda failing, successes, alpha, seqs, cfg=None: ref):
        return contrastive_localize(pop, f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/new/reference_sensitivity.json")
    ap.add_argument("--tex", default="results/new/tab_reference_sensitivity.tex")
    ap.add_argument("--same-path-threshold", type=float, default=0.5,
                    help="Jaccard overlap of CALC expressions at/above which a reference "
                         "counts as following the failing run's arithmetic path")
    args = ap.parse_args(argv)

    pops, anns = load_human()
    cfg_edit, cfg_content = ContrastiveConfig(sibling="edit"), ContrastiveConfig(sibling="content")
    out = {"design": {"families": [p.query_id for p in pops], "n_failing": len(anns),
                      "gold": "results/human_perrun.json (== annotations_human/adjudicated)",
                      "reference_forcing": "monkeypatch refalign.contrastive._nearest_sibling; "
                                           "contrastive_localize otherwise unmodified (default config)",
                      "same_path_definition": f"Jaccard overlap of normalised Worker CALC expressions "
                                              f">= {args.same_path_threshold}"},
           "families": {}, "per_run": {}, "pooled": {}}
    pooled_pairs = []      # (family, run, ref, step_ok, agent_ok, edit, content, overlap, same_path)
    pooled_runs = {"nearest": [], "first": [], "best": [], "worst": [], "mean": [], "invariant": [],
                   "nearest_agent": [], "first_agent": []}

    for pop in pops:
        S_ = pop.successes
        alpha, seqs = _encode_actions(pop)
        fam = {"n_success": len(S_), "n_fail": len(pop.failures), "first_reference": S_[0].run_id,
               "runs": {}}
        for f in pop.failures:
            ann = anns[f.run_id]
            own = contrastive_localize(pop, f)
            nearest = own.meta.get("sibling") or _nearest_sibling(f, S_, alpha, seqs, cfg_content).run_id
            refs = {}
            for r in S_:
                p = forced_prediction(pop, f, r)
                ov = path_overlap(f, r)
                refs[r.run_id] = {
                    "agent": p.agent, "step": int(p.step), "kind": p.meta.get("kind") or p.meta.get("fallback"),
                    "agent_correct": bool(p.agent == ann.responsible_agent),
                    "step_correct": bool(p.step == ann.decisive_step),
                    "edit_distance": float(_sibling_distance(f, r, alpha, seqs, cfg_edit)),
                    "content_distance": float(_sibling_distance(f, r, alpha, seqs, cfg_content)),
                    "path_overlap": ov, "same_path": bool(ov >= args.same_path_threshold)}
                pooled_pairs.append((pop.query_id, f.run_id, r.run_id, refs[r.run_id]["step_correct"],
                                    refs[r.run_id]["agent_correct"], refs[r.run_id]["edit_distance"],
                                    refs[r.run_id]["content_distance"], ov, refs[r.run_id]["same_path"]))
            # sanity: forcing the nearest reference reproduces the method's own prediction
            assert (refs[nearest]["agent"], refs[nearest]["step"]) == (own.agent, own.step), (f.run_id, own, refs[nearest])
            steps = [v["step"] for v in refs.values()]
            sc = [v["step_correct"] for v in refs.values()]
            ac = [v["agent_correct"] for v in refs.values()]
            rec = {"gold": [ann.responsible_agent, ann.decisive_step], "nearest_reference": nearest,
                   "own_prediction": [own.agent, int(own.step)],
                   "invariant_step": len(set(steps)) == 1, "distinct_steps": sorted(set(steps)),
                   "step_correct_nearest": refs[nearest]["step_correct"],
                   "step_correct_first": refs[S_[0].run_id]["step_correct"],
                   "step_correct_best": any(sc), "step_correct_worst": all(sc),
                   "step_correct_mean": float(np.mean(sc)),
                   "agent_correct_nearest": refs[nearest]["agent_correct"],
                   "agent_correct_first": refs[S_[0].run_id]["agent_correct"],
                   "agent_correct_mean": float(np.mean(ac)),
                   "n_same_path_refs": int(sum(v["same_path"] for v in refs.values())),
                   "step_correct_same_path": [v["step_correct"] for v in refs.values() if v["same_path"]],
                   "step_correct_other_path": [v["step_correct"] for v in refs.values() if not v["same_path"]],
                   "references": refs}
            fam["runs"][f.run_id] = rec
            out["per_run"][f.run_id] = rec
            pooled_runs["nearest"].append(rec["step_correct_nearest"]); pooled_runs["first"].append(rec["step_correct_first"])
            pooled_runs["best"].append(rec["step_correct_best"]); pooled_runs["worst"].append(rec["step_correct_worst"])
            pooled_runs["mean"].append(rec["step_correct_mean"]); pooled_runs["invariant"].append(rec["invariant_step"])
            pooled_runs["nearest_agent"].append(rec["agent_correct_nearest"]); pooled_runs["first_agent"].append(rec["agent_correct_first"])
        R = fam["runs"].values()
        fam["summary"] = {
            "frac_invariant_step": S.acc_record([r["invariant_step"] for r in R]),
            "step_acc_nearest": S.acc_record([r["step_correct_nearest"] for r in R]),
            "step_acc_first": S.acc_record([r["step_correct_first"] for r in R]),
            "step_acc_best": S.acc_record([r["step_correct_best"] for r in R]),
            "step_acc_worst": S.acc_record([r["step_correct_worst"] for r in R]),
            "step_acc_mean_random_ref": float(np.mean([r["step_correct_mean"] for r in R])),
            "agent_acc_nearest": S.acc_record([r["agent_correct_nearest"] for r in R]),
            "agent_acc_first": S.acc_record([r["agent_correct_first"] for r in R]),
            "n_runs_with_same_path_ref": int(sum(r["n_same_path_refs"] > 0 for r in R)),
            "same_path_pairs": S.acc_record([x for r in R for x in r["step_correct_same_path"]]),
            "other_path_pairs": S.acc_record([x for r in R for x in r["step_correct_other_path"]]),
            "nearest_is_first": S.acc_record([r["nearest_reference"] == S_[0].run_id for r in R]),
        }
        out["families"][pop.query_id] = fam
        s = fam["summary"]
        print(f"{pop.query_id}: succ={len(S_)} fail={len(pop.failures)} invariant={s['frac_invariant_step']['c']}/{s['frac_invariant_step']['n']} "
              f"step nearest={s['step_acc_nearest']['acc']:.2f} first={s['step_acc_first']['acc']:.2f} "
              f"best={s['step_acc_best']['acc']:.2f} worst={s['step_acc_worst']['acc']:.2f} "
              f"rand={s['step_acc_mean_random_ref']:.2f} | same-path refs: runs={s['n_runs_with_same_path_ref']} "
              f"pairs={s['same_path_pairs']['c']}/{s['same_path_pairs']['n']} other={s['other_path_pairs']['c']}/{s['other_path_pairs']['n']}")

    # pooled over the 32 runs
    P = out["pooled"]
    for k in ["nearest", "first", "best", "worst", "invariant", "nearest_agent", "first_agent"]:
        P[f"step_acc_{k}" if "agent" not in k and k != "invariant" else k] = S.acc_record(pooled_runs[k])
    P["step_acc_mean_random_ref"] = float(np.mean(pooled_runs["mean"]))
    P["mcnemar_nearest_vs_first_step"] = S.mcnemar_exact(pooled_runs["nearest"], pooled_runs["first"])
    P["mcnemar_nearest_vs_first_agent"] = S.mcnemar_exact(pooled_runs["nearest_agent"], pooled_runs["first_agent"])
    # distance / path correlations over all (failing, reference) pairs
    sc = np.array([p[3] for p in pooled_pairs], float)
    corr = {}
    for name, idx in [("edit_distance", 5), ("content_distance", 6), ("path_overlap", 7)]:
        x = np.array([p[idx] for p in pooled_pairs], float)
        if np.std(x) > 0 and np.std(sc) > 0:
            pb = stats.pointbiserialr(sc, x)
            sp = stats.spearmanr(x, sc)
            corr[name] = {"point_biserial_r": float(pb.statistic), "p": float(pb.pvalue),
                          "spearman_rho": float(sp.statistic), "spearman_p": float(sp.pvalue), "n_pairs": int(x.size)}
        else:
            corr[name] = {"note": "zero variance", "n_pairs": int(x.size)}
    P["correlation_with_step_correct"] = corr
    same = [p[3] for p in pooled_pairs if p[8]]
    other = [p[3] for p in pooled_pairs if not p[8]]
    P["same_path_pairs"] = S.acc_record(same)
    P["other_path_pairs"] = S.acc_record(other)
    if same and other:
        P["fisher_same_vs_other_path"] = S.fisher_2x2(int(sum(same)), len(same), int(sum(other)), len(other))
    P["n_pairs"] = len(pooled_pairs)
    print(f"POOLED n=32: invariant={P['invariant']['c']}/32 step nearest={P['step_acc_nearest']['acc']:.3f} "
          f"first={P['step_acc_first']['acc']:.3f} best={P['step_acc_best']['acc']:.3f} worst={P['step_acc_worst']['acc']:.3f} "
          f"rand={P['step_acc_mean_random_ref']:.3f}; McNemar nearest vs first p={P['mcnemar_nearest_vs_first_step']['p_value']:.3f}")
    print("correlations:", json.dumps(corr, indent=1))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    _write_tex(out, args.tex)
    print(f"wrote {args.out} and {args.tex}")
    return 0


def _write_tex(out, path):
    rows = []
    for fam, d in out["families"].items():
        s = d["summary"]
        rows.append(f"\\texttt{{{fam.replace('real_', '')}}} & {d['n_success']} & {d['n_fail']} & "
                    f"{s['frac_invariant_step']['c']}/{s['frac_invariant_step']['n']} & "
                    f"{s['step_acc_nearest']['acc']:.2f} & {s['step_acc_first']['acc']:.2f} & "
                    f"{s['step_acc_mean_random_ref']:.2f} & {s['step_acc_best']['acc']:.2f} & {s['step_acc_worst']['acc']:.2f} & "
                    f"{s['same_path_pairs']['c']}/{s['same_path_pairs']['n']} & {s['other_path_pairs']['c']}/{s['other_path_pairs']['n']} \\\\")
    P = out["pooled"]
    tot_s = sum(d["n_success"] for d in out["families"].values())
    rows.append("\\midrule")
    rows.append(f"pooled & {tot_s} & 32 & {P['invariant']['c']}/32 & {P['step_acc_nearest']['acc']:.2f} & "
                f"{P['step_acc_first']['acc']:.2f} & {P['step_acc_mean_random_ref']:.2f} & {P['step_acc_best']['acc']:.2f} & "
                f"{P['step_acc_worst']['acc']:.2f} & {P['same_path_pairs']['c']}/{P['same_path_pairs']['n']} & "
                f"{P['other_path_pairs']['c']}/{P['other_path_pairs']['n']} \\\\")
    mc = P["mcnemar_nearest_vs_first_step"]
    lo, hi = P["step_acc_nearest"]["wilson_lo"], P["step_acc_nearest"]["wilson_hi"]
    tex = f"""% Experiment 2: reference sensitivity on the human benchmark (LLM-free).
\\begin{{table}}[t]
\\centering\\small
\\begin{{tabular}}{{lcccccccccc}}
\\toprule
 & & & step & \\multicolumn{{5}}{{c}}{{exact-step accuracy under reference}} & \\multicolumn{{2}}{{c}}{{pairs correct}} \\\\
\\cmidrule(lr){{5-9}} \\cmidrule(lr){{10-11}}
family & \\#succ & \\#fail & invariant & nearest & first & random & best & worst & same path & other path \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{Sensitivity of RefAlign's exact-step prediction to the choice of successful
reference on the $n{{=}}32$ human-labeled failing runs. For each failing run every
successful run of its family is forced as the reference. ``nearest'' is the method's
own content-aware choice, ``first'' the reference shown to the annotators
(\\texttt{{successes[0]}}), ``random'' the mean over references, ``best''/``worst'' the
fraction of runs for which some/every reference recovers the human step. ``pairs
correct'' counts (failing run, reference) pairs by whether the reference shares the
failing run's arithmetic path (Jaccard overlap of CALC expressions $\\ge 0.5$).
Pooled nearest-reference accuracy {P['step_acc_nearest']['acc']:.2f} (95\\% Wilson [{lo:.2f}, {hi:.2f}]);
exact McNemar nearest vs.\\ first: $b{{=}}{mc['b']}$, $c{{=}}{mc['c']}$, $p{{=}}{mc['p_value']:.3f}$ (single test, no correction).}}
\\label{{tab:reference-sensitivity}}
\\end{{table}}
"""
    open(path, "w").write(tex)


if __name__ == "__main__":
    sys.exit(main())
