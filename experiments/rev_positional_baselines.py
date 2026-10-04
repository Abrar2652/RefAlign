"""Leakage-controlled positional baselines on the human-labeled natural failures.

Leave-one-task-family-out (LOFO) over the four families: every label-dependent
choice is fitted on the other three families and applied to the held-out one.
Baselines: most frequent decisive-step index; most frequent responsible agent;
joint (agent, step) mode; last step; penultimate step; normalized position
(mean of step/(len-1) over training runs, rounded). Invalid predicted positions
(>= trace length) are clamped to the last step; this rule is prespecified here
and reported. Metrics: exact step, exact agent, joint; pooled and family-macro.
RefAlign (no fitting) and the post-hoc constant step-3 diagnostic are listed
beside them. Writes results/revision/positional_baselines.json.

    python -m experiments.rev_positional_baselines
"""
import json
import os
from collections import Counter

import numpy as np

from refalign.contrastive import contrastive_localize
from refalign import stats_new as S
from experiments.run_reference_sensitivity import load_human

OUT = "results/revision/positional_baselines.json"
INVALID_RULE = "predicted step >= len(trace) is clamped to len(trace)-1; predicted step < 0 is clamped to 0"


def clamp(step, n):
    return max(0, min(int(step), n - 1))


def fit(train):
    """train: list of (trace, annotation). Returns fitted parameters."""
    steps = Counter(a.decisive_step for _, a in train)
    agents = Counter(a.responsible_agent for _, a in train)
    joint = Counter((a.responsible_agent, a.decisive_step) for _, a in train)
    pos = [a.decisive_step / max(1, len(t) - 1) for t, a in train]
    tie = lambda c: sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]   # deterministic tie-break
    return {"mode_step": tie(steps), "mode_agent": tie(agents), "mode_joint": tie(joint), "mean_norm_pos": float(np.mean(pos))}


def predict(name, params, trace):
    n = len(trace)
    if name == "mode_step":
        s = clamp(params["mode_step"], n); return trace.agent_at(s), s
    if name == "mode_agent":
        return params["mode_agent"], None
    if name == "mode_joint":
        ag, s = params["mode_joint"]; return ag, clamp(s, n)
    if name == "last_step":
        return trace.agent_at(n - 1), n - 1
    if name == "penultimate_step":
        s = clamp(n - 2, n); return trace.agent_at(s), s
    if name == "norm_pos":
        s = clamp(round(params["mean_norm_pos"] * (n - 1)), n); return trace.agent_at(s), s
    if name == "mode_agent+mode_step":
        s = clamp(params["mode_step"], n); return params["mode_agent"], s
    raise KeyError(name)


METHODS = ["mode_step", "mode_agent", "mode_joint", "mode_agent+mode_step", "norm_pos", "last_step", "penultimate_step"]


def main():
    pops, anns = load_human()
    fams = {p.query_id: [(f, anns[f.run_id]) for f in p.failures] for p in pops}
    per_run = {}
    fits = {}
    for held in fams:
        train = [x for q, L in fams.items() if q != held for x in L]
        params = fit(train)
        fits[held] = params
        for tr, ann in fams[held]:
            rec = per_run.setdefault(tr.run_id, {"family": held, "gold_agent": ann.responsible_agent, "gold_step": ann.decisive_step, "n_steps": len(tr)})
            for m in METHODS:
                ag, st = predict(m, params, tr)
                rec[m] = {"agent": ag, "step": st,
                          "agent_correct": bool(ag == ann.responsible_agent),
                          "step_correct": bool(st is not None and st == ann.decisive_step),
                          "joint_correct": bool(ag == ann.responsible_agent and st is not None and st == ann.decisive_step)}
            p = contrastive_localize([pp for pp in pops if pp.query_id == held][0], tr)
            rec["refalign"] = {"agent": p.agent, "step": int(p.step), "agent_correct": bool(p.agent == ann.responsible_agent),
                               "step_correct": bool(p.step == ann.decisive_step),
                               "joint_correct": bool(p.agent == ann.responsible_agent and p.step == ann.decisive_step)}
            rec["posthoc_const_step3"] = {"agent": "Worker", "step": 3, "agent_correct": ann.responsible_agent == "Worker",
                                          "step_correct": ann.decisive_step == 3, "joint_correct": ann.responsible_agent == "Worker" and ann.decisive_step == 3}
    names = METHODS + ["refalign", "posthoc_const_step3"]
    summary = {}
    for m in names:
        pooled = {k: S.acc_record([per_run[r][m][k] for r in per_run]) for k in ("step_correct", "agent_correct", "joint_correct")}
        by_fam = {q: {k: S.acc_record([per_run[t.run_id][m][k] for t, _ in L]) for k in ("step_correct", "agent_correct", "joint_correct")} for q, L in fams.items()}
        macro = {k: float(np.mean([by_fam[q][k]["acc"] for q in fams])) for k in ("step_correct", "agent_correct", "joint_correct")}
        summary[m] = {"pooled": pooled, "by_family": by_fam, "family_macro": macro}
    # paired tests vs RefAlign on exact step (single family of 7 comparisons, Holm)
    tests = {}
    for m in METHODS:
        x = [per_run[r]["refalign"]["step_correct"] for r in per_run]; y = [per_run[r][m]["step_correct"] for r in per_run]
        tests[m] = S.mcnemar_exact(x, y)
    adj = S.holm({m: tests[m]["p_value"] for m in tests})
    for m in tests:
        tests[m]["p_holm"] = adj[m]["p_holm"]; tests[m]["family"] = "RefAlign vs each of the 7 LOFO baselines, exact step (m=7)"
    out = {"design": {"evaluation": "leave-one-task-family-out over 4 families; all label-dependent parameters fitted on the other 3 families",
                      "invalid_position_rule": INVALID_RULE, "n_runs": len(per_run), "families": {q: len(L) for q, L in fams.items()},
                      "fits_per_held_out_family": fits, "tie_break": "most frequent; ties broken by string order (deterministic)",
                      "posthoc_const_step3": "retrospective diagnostic using the labels themselves; not a leakage-controlled baseline"},
           "summary": summary, "tests_vs_refalign_step": tests, "per_run": per_run}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"{'method':24s} {'step pooled':>14s} {'agent pooled':>14s} {'joint pooled':>14s} {'step macro':>10s} {'agent macro':>11s} {'joint macro':>11s}")
    for m in names:
        s = summary[m]
        print(f"{m:24s} {s['pooled']['step_correct']['c']:>3d}/32={s['pooled']['step_correct']['acc']:.3f} {s['pooled']['agent_correct']['c']:>3d}/32={s['pooled']['agent_correct']['acc']:.3f} "
              f"{s['pooled']['joint_correct']['c']:>3d}/32={s['pooled']['joint_correct']['acc']:.3f} {s['family_macro']['step_correct']:>10.3f} {s['family_macro']['agent_correct']:>11.3f} {s['family_macro']['joint_correct']:>11.3f}")
    print("fits:", json.dumps(fits))
    print("McNemar RefAlign vs baselines (step):", {m: (t['b'], t['c'], round(t['p_value'], 3), round(t['p_holm'], 3)) for m, t in tests.items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
