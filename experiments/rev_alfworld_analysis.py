"""Analysis of the ALFWorld external evaluation: funnel, replay-verified labels,
positional baselines (leave-one-task-type-out), method accuracy, replay outcomes,
reference-selection effects, costs, and paired tests. Writes a JSON, LaTeX
snippets and a Markdown summary under results/revision/.

    PYTHONPATH=. python -m experiments.rev_alfworld_analysis --prefix results/revision/alf_test
Expects <prefix>_traces.json, <prefix>_traces_meta.json, <prefix>_attribution.json,
<prefix>_replay.json (optional), <prefix>_replay_jobs.json (optional).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

from refalign import stats_new as S

TASK_TYPES = ["pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
              "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"]
SHORT = {"pick_and_place_simple": "place", "look_at_obj_in_light": "examine", "pick_clean_then_place_in_recep": "clean",
         "pick_heat_then_place_in_recep": "heat", "pick_cool_then_place_in_recep": "cool", "pick_two_obj_and_place": "two"}
REF_METHODS = ["refalign", "refalign_s", "refalign_s_first", "refalign_s_edit", "refalign_s_random", "judge_aao", "judge_matched",
               "judge_first", "judge_edit", "judge_random"]
POS_METHODS = ["lo_mode_step", "lo_norm_pos", "penultimate", "last_step", "random_step"]
PRIMARY_FAMILY = [("refalign_s", "judge_matched"), ("refalign_s", "lo_norm_pos"), ("refalign_s", "random_step"), ("refalign_s", "refalign")]


def seeds_of(att, prefix):
    return sorted({k for r in att.values() for k in r if k.startswith(prefix + "-s")})


def pred_step(rec, m, seed_keys):
    """Prediction for method m on one run; seed-averaged methods return a list."""
    if m.endswith("random") or m == "random_step":
        ks = [k for k in seed_keys if k.startswith(m + "-s")] if m != "random_step" else [k for k in seed_keys if k.startswith("random_step-s")]
        return [int(rec[k]["step"]) for k in ks if k in rec]
    return [int(rec[m]["step"])] if m in rec else []


def labels_from_replay(replay, kmax, seeds, rule="strict"):
    """Earliest repairable step per profiled run from expert-override and no-edit
    replays at each grid step k:
      strict : all expert replays succeed AND no no-edit replay succeeds
      relaxed: expert success fraction >= 0.5 AND no-edit success fraction <= 0.5
    Steps where the no-edit control is missing are not eligible under either rule."""
    pe, pn = defaultdict(dict), defaultdict(dict)
    for key, r in replay.items():
        d = pe if r["mode"] == "expert" else pn if r["mode"] == "noedit" else None
        if d is not None:
            d[r["run_id"]].setdefault(r["step"], []).append(bool(r["won"]))
    labels = {}
    for rid, steps in pe.items():
        ks = sorted(k for k in steps if 1 <= k <= kmax and k in pn.get(rid, {}))
        prof = {k: {"p_expert": float(np.mean(steps[k])), "p_noedit": float(np.mean(pn[rid][k])), "n_e": len(steps[k]), "n_n": len(pn[rid][k])} for k in ks}
        if rule == "strict":
            rep = [k for k in ks if prof[k]["p_expert"] == 1.0 and prof[k]["p_noedit"] == 0.0 and prof[k]["n_e"] >= len(seeds) and prof[k]["n_n"] >= len(seeds)]
        else:
            rep = [k for k in ks if prof[k]["p_expert"] >= 0.5 and prof[k]["p_noedit"] <= 0.5]
        labels[rid] = {"profile": prof, "earliest": (rep[0] if rep else None), "repairable": rep, "n_profiled": len(ks),
                       "resample_recoverable": [k for k in ks if prof[k]["p_noedit"] > 0.5]}
    return labels


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args(argv)
    P = args.prefix
    store = json.load(open(f"{P}_traces.json"))
    meta = json.load(open(f"{P}_traces_meta.json"))
    att = json.load(open(f"{P}_attribution.json"))
    replay = json.load(open(f"{P}_replay.json")) if os.path.exists(f"{P}_replay.json") else {}
    jobs = json.load(open(f"{P}_replay_jobs.json")) if os.path.exists(f"{P}_replay_jobs.json") else {"jobs": [], "profile_runs": [], "kmax": 12, "not_replayable": []}
    seed_keys = sorted({k for r in att.values() for k in r if "-s" in k})
    out = {"funnel": meta["funnel"], "design": {k: v for k, v in meta.items() if k not in ("games", "funnel")}}
    rids = sorted(att)
    ttype = {rid: att[rid]["task_type"] for rid in rids}

    # ---- replay-verified labels -------------------------------------------
    labels = labels_from_replay(replay, jobs.get("kmax", 12), args.seeds, rule="strict")
    labels_relaxed = labels_from_replay(replay, jobs.get("kmax", 12), args.seeds, rule="relaxed")
    profiled = [rid for rid in jobs.get("profile_runs", []) if rid in labels]
    labelled = [rid for rid in profiled if labels[rid]["earliest"] is not None]
    labelled_relaxed = [rid for rid in profiled if labels_relaxed[rid]["earliest"] is not None]
    out["labels"] = {"rule": "strict: expert override succeeds on every seed and the no-edit control succeeds on none; relaxed: expert >= 0.5 and no-edit <= 0.5",
                     "n_profiled": len(profiled), "n_with_earliest_repairable_strict": len(labelled), "n_with_earliest_repairable_relaxed": len(labelled_relaxed),
                     "n_unrepairable_within_kmax_strict": len(profiled) - len(labelled), "kmax": jobs.get("kmax"),
                     "earliest_distribution_strict": dict(sorted(Counter(labels[r]["earliest"] for r in labelled).items())),
                     "earliest_distribution_relaxed": dict(sorted(Counter(labels_relaxed[r]["earliest"] for r in labelled_relaxed).items())),
                     "n_runs_resample_recoverable_at_step1": sum(1 for r in profiled if 1 in labels[r]["resample_recoverable"]),
                     "profiles": {r: labels[r] for r in profiled}}

    # ---- leave-one-task-type-out positional baselines on labelled runs ------
    fits = {}
    for held in TASK_TYPES:
        train = [r for r in labelled if ttype[r] != held]
        if not train:
            continue
        steps = Counter(labels[r]["earliest"] for r in train)
        mode = sorted(steps.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        npos = float(np.mean([labels[r]["earliest"] / max(1, att[r]["n_steps"] - 1) for r in train]))
        fits[held] = {"mode_step": mode, "mean_norm_pos": npos, "n_train": len(train)}
    for rid in rids:
        f = fits.get(ttype[rid])
        n = att[rid]["n_steps"]
        if f:
            att[rid]["lo_mode_step"] = {"step": max(0, min(f["mode_step"], n - 1)), "agent": "Actor"}
            att[rid]["lo_norm_pos"] = {"step": max(0, min(int(round(f["mean_norm_pos"] * (n - 1))), n - 1)), "agent": "Actor"}
    out["positional_fits_lofo"] = fits
    att_disk = json.load(open(f"{P}_attribution.json"))
    for rid in rids:
        for m in ("lo_mode_step", "lo_norm_pos"):
            if m in att[rid]:
                att_disk[rid][m] = {**att[rid][m], "note": "fitted leave-one-task-type-out on replay labels of the other task types"}
    json.dump(att_disk, open(f"{P}_attribution.json", "w"), indent=0)

    # ---- P1: exact-step agreement with the earliest repairable step ---------
    def acc_table(subset):
        tab = {}
        for m in REF_METHODS + POS_METHODS:
            corr, per_type = [], defaultdict(list)
            for rid in subset:
                preds = pred_step(att[rid], m, seed_keys)
                if not preds:
                    continue
                c = float(np.mean([p == labels[rid]["earliest"] for p in preds]))
                corr.append(c); per_type[ttype[rid]].append(c)
            if corr:
                k = int(round(sum(corr)))
                tab[m] = {"n": len(corr), "c_mean": float(sum(corr)), "acc": float(np.mean(corr)),
                          "wilson": S.wilson(k, len(corr)), "by_type": {SHORT[t]: (float(np.mean(v)), len(v)) for t, v in per_type.items()},
                          "macro": float(np.mean([np.mean(v) for v in per_type.values()]))}
        return tab
    out["P1_exact_step_vs_earliest_repairable"] = acc_table(labelled)
    _lab_strict = labels
    labels = labels_relaxed
    out["P1_relaxed_rule_sensitivity"] = acc_table(labelled_relaxed)
    labels = _lab_strict
    # within-one and "within repairable set" (secondary)
    sec = {}
    for m in REF_METHODS + POS_METHODS:
        w1, inset = [], []
        for rid in labelled:
            preds = pred_step(att[rid], m, seed_keys)
            if not preds:
                continue
            w1.append(float(np.mean([abs(p - labels[rid]["earliest"]) <= 1 for p in preds])))
            inset.append(float(np.mean([p in labels[rid]["repairable"] for p in preds])))
        if w1:
            sec[m] = {"within_1": float(np.mean(w1)), "in_repairable_set": float(np.mean(inset)), "n": len(w1)}
    out["secondary"] = sec
    # paired tests (primary family, Holm m=4) on single-prediction methods; random methods use seed 0
    def vec(m):
        v = {}
        for rid in labelled:
            preds = pred_step(att[rid], m, seed_keys)
            if preds:
                v[rid] = (preds[0] == labels[rid]["earliest"])
        return v
    tests = {}
    pv = {}
    for a, b in PRIMARY_FAMILY:
        va, vb = vec(a), vec(b)
        common = sorted(set(va) & set(vb))
        if len(common) >= 1:
            t = S.mcnemar_exact([va[r] for r in common], [vb[r] for r in common]); t["n"] = len(common)
            tests[f"{a}_vs_{b}"] = t; pv[f"{a}_vs_{b}"] = t["p_value"]
    adj = S.holm(pv) if pv else {}
    for k in tests:
        tests[k]["p_holm"] = adj[k]["p_holm"]; tests[k]["family"] = "prespecified primary family (m=%d)" % len(pv)
    out["P1_tests"] = tests
    # game-level cluster bootstrap for refalign_s and judge_matched P1
    rng = np.random.RandomState(0)
    games = sorted({att[r]["game_id"] for r in labelled})
    boot = {}
    for m in ["refalign_s", "judge_matched", "lo_norm_pos", "penultimate"]:
        v = vec(m)
        by_g = defaultdict(list)
        for r, ok in v.items():
            by_g[att[r]["game_id"]].append(ok)
        if len(by_g) >= 6:
            vals = []
            gl = sorted(by_g)
            for _ in range(2000):
                pick = rng.choice(len(gl), len(gl), replace=True)
                xs = [x for i in pick for x in by_g[gl[i]]]
                vals.append(float(np.mean(xs)))
            boot[m] = {"n_games": len(gl), "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}
    out["P1_game_cluster_bootstrap"] = boot

    # ---- replay experiments at the method-selected steps -------------------
    def replay_rate(m, mode):
        """Per (run, seed) success at the step selected by m, with the paired no-edit control
        at the same step. A step-0 selection (the Planner) admits no expert substitution: it is
        replayed as a fresh plan + policy ('replan'), identical for treatment and control, so it
        is reported separately and excluded from the substitution rates."""
        won, ctrl, pairs, n_not, n_step0 = [], [], [], 0, 0
        won0 = []
        for rid in rids:
            preds = pred_step(att[rid], m, seed_keys)
            if not preds:
                continue
            st = preds[0]
            if st > att[rid]["n_actor_steps"]:
                n_not += 1; won += [False] * len(args.seeds); ctrl += [None] * len(args.seeds); continue
            if st == 0:
                n_step0 += 1
                for s_ in args.seeds:
                    k = f"{rid}|0|replan|{s_}"
                    if k in replay:
                        won0.append(bool(replay[k]["won"]))
                continue
            for s_ in args.seeds:
                k1 = f"{rid}|{st}|{mode}|{s_}"; k0 = f"{rid}|{st}|noedit|{s_}"
                if k1 in replay:
                    w = bool(replay[k1]["won"]); c = bool(replay[k0]["won"]) if k0 in replay else None
                    won.append(w); ctrl.append(c)
                    if c is not None:
                        pairs.append((w, c))
        n = len(won)
        if n == 0 and not won0:
            return None
        cv = [c for c in ctrl if c is not None]
        r = {"n_episodes_substitution": n, "won": int(sum(won)), "rate": (float(np.mean(won)) if n else None),
             "wilson": (S.wilson(int(sum(won)), n) if n else None),
             "control_rate": float(np.mean(cv)) if cv else None, "n_not_replayable": n_not,
             "improvement_over_noedit": ((float(np.mean(won)) - float(np.mean(cv))) if (cv and n) else None),
             "n_runs_step0_selected": n_step0, "replan_episodes_at_step0": len(won0),
             "replan_success_rate_step0": (float(np.mean(won0)) if won0 else None)}
        if pairs:
            r["mcnemar_vs_noedit"] = S.mcnemar_exact([p[0] for p in pairs], [p[1] for p in pairs])
        return r
    out["replay_A_expert_at_selected_step"] = {m: replay_rate(m, "expert") for m in REF_METHODS + POS_METHODS}
    out["replay_B_repair_at_selected_step"] = {m: replay_rate(m, "repair") for m in REF_METHODS + POS_METHODS}
    out["replay_A_expert_at_selected_step"] = {k: v for k, v in out["replay_A_expert_at_selected_step"].items() if v}
    out["replay_B_repair_at_selected_step"] = {k: v for k, v in out["replay_B_repair_at_selected_step"].items() if v}
    # oracle location (earliest repairable) replay A = 1.0 by construction on labelled runs; note it
    # repair-model diagnostics
    rep = [r for r in replay.values() if r["mode"] == "repair"]
    out["repair_model"] = {"n": len(rep), "valid_proposals": int(sum(bool(r.get("repair_valid")) for r in rep)),
                           "same_as_original": int(sum(r.get("repair_action") == r.get("original_action") for r in rep)),
                           "won": int(sum(r["won"] for r in rep))}
    # ---- reference selection effects (deterministic variant) ---------------
    chg = {}
    for alt in ["refalign_s_first", "refalign_s_edit"] + [k for k in seed_keys if k.startswith("refalign_s_random-s")]:
        changed = [rid for rid in rids if alt in att[rid] and att[rid][alt]["step"] != att[rid]["refalign_s"]["step"]]
        acc_c = [att[rid]["refalign_s"]["step"] == labels[rid]["earliest"] for rid in changed if rid in labels and labels[rid]["earliest"] is not None]
        acc_alt = [att[rid][alt]["step"] == labels[rid]["earliest"] for rid in changed if rid in labels and labels[rid]["earliest"] is not None]
        chg[alt] = {"n_changed": len(changed), "n": len(rids), "frac_changed": len(changed) / max(1, len(rids)),
                    "acc_refalign_s_on_changed": (float(np.mean(acc_c)) if acc_c else None), "acc_alt_on_changed": (float(np.mean(acc_alt)) if acc_alt else None),
                    "n_changed_labelled": len(acc_c),
                    "by_type": {SHORT[t]: sum(1 for r in changed if ttype[r] == t) / max(1, sum(1 for r in rids if ttype[r] == t)) for t in TASK_TYPES}}
    out["reference_selection"] = chg
    jchg = {}
    for alt in ["judge_first", "judge_edit"] + [k for k in seed_keys if k.startswith("judge_random-s")]:
        changed = [rid for rid in rids if alt in att[rid] and "judge_matched" in att[rid] and att[rid][alt]["step"] != att[rid]["judge_matched"]["step"]]
        jchg[alt] = {"n_changed": len(changed), "n": sum(1 for rid in rids if "judge_matched" in att[rid])}
    out["reference_selection_judge"] = jchg
    # pool-size sensitivity: number of successes per eligible failure
    out["pool_size"] = dict(sorted(Counter(att[r]["n_successes"] for r in rids).items()))
    # ---- costs ------------------------------------------------------------
    tok = {m: float(np.mean([att[r][m]["tokens"] for r in rids if m in att[r] and "tokens" in att[r][m]])) for m in ["judge_aao", "judge_matched"] if any(m in att[r] for r in rids)}
    by_game = defaultdict(list)
    for r in store.values():
        by_game[r["game_id"]].append(r)
    first_succ = [next((i for i, r in enumerate(sorted(v, key=lambda x: x["rollout"])) if r["outcome"] == "success"), None) for v in by_game.values()]
    fs = [x + 1 for x in first_succ if x is not None]
    out["costs"] = {"collection_tokens": sum(r["tokens"] for r in store.values()), "collection_calls": sum(r["n_llm_calls"] for r in store.values()),
                    "tokens_per_episode": float(np.mean([r["tokens"] for r in store.values()])),
                    "rollouts_to_first_success": {"mean": float(np.mean(fs)) if fs else None, "median": float(np.median(fs)) if fs else None, "n_games_with_success": len(fs)},
                    "attribution_tokens_per_run": tok, "refalign_tokens": 0,
                    "replay_tokens": sum(r["tokens"] for r in replay.values()), "replay_episodes": len(replay)}
    out["abstention"] = {"refalign_s_abstain": sum(att[r]["refalign_s"]["abstain"] for r in rids), "refalign_numeric_fallback": sum(att[r]["refalign"]["kind"] == "no_sibling" for r in rids), "n": len(rids),
                         "refalign_s_kinds": dict(Counter(att[r]["refalign_s"]["kind"] for r in rids)), "refalign_kinds": dict(Counter(att[r]["refalign"]["kind"] for r in rids))}
    json.dump(out, open(f"{P}_analysis.json", "w"), indent=1, default=float)
    # ---- console summary ----
    f = out["funnel"]
    print(f"FUNNEL games {f['games']} runs {f['runs']} succ {f['successes']} fail {f['failures']} mixed-games {f['games_mixed']} all-fail {f['games_all_failure']} eligible-failures {f['reference_eligible_failures']} no-ref {f['failures_without_reference']} cap {f['hit_cap']}")
    print(f"LABELS profiled {out['labels']['n_profiled']} earliest-repairable (strict) {out['labels']['n_with_earliest_repairable_strict']} dist {out['labels']['earliest_distribution_strict']}")
    print("P1 exact step vs earliest repairable:")
    for m, v in out["P1_exact_step_vs_earliest_repairable"].items():
        print(f"  {m:20s} {v['c_mean']:.1f}/{v['n']} = {v['acc']:.3f} [{v['wilson'][0]:.2f},{v['wilson'][1]:.2f}] macro {v['macro']:.3f} by-type {v['by_type']}")
    print("tests:", {k: (v['b'], v['c'], round(v['p_value'], 3), round(v['p_holm'], 3)) for k, v in tests.items()})
    for key in ["replay_A_expert_at_selected_step", "replay_B_repair_at_selected_step"]:
        print(key)
        for m, v in out[key].items():
            rate = f"{v['rate']:.3f}" if v['rate'] is not None else "--"
            print(f"  {m:20s} won {v['won']}/{v['n_episodes_substitution']} = {rate} ctrl {v['control_rate']} impr {v['improvement_over_noedit']} step0-runs {v['n_runs_step0_selected']} replan-rate {v['replan_success_rate_step0']} not-replayable {v['n_not_replayable']}")
    print("reference selection (refalign_s):", {k: (v['n_changed'], v['n'], v['acc_refalign_s_on_changed'], v['acc_alt_on_changed']) for k, v in chg.items()})
    print("abstention:", out["abstention"]); print("costs:", out["costs"])
    print("wrote", f"{P}_analysis.json")


if __name__ == "__main__":
    main()
