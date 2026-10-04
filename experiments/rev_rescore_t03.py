"""Rescore the saved prompt-intervention rollouts with the corrected t03 answer key.

Task t03 ("A rectangle is 14 m by 9 m. A 2 m wide path runs around its inside
edge...") has area 14*9 - (14-4)*(9-4) = 126 - 50 = 76, not 84. Model outputs
are NOT regenerated: each saved run's committed final answer is compared with
the corrected key, the harness outcome is recomputed, and the full analysis
(Fisher/Holm/Newcombe, RefAlign on failures, evaluated-answer scoring) is
rerun on the corrected outcomes. Original files under results/new are left
untouched; outputs go to results/revision/intervention_rescored_<model>.json.

    python -m experiments.rev_rescore_t03
"""
import json
import os
import re
from unittest import mock

from experiments import run_intervention as RI
from annotation.collect_traces import TASKS as CANDIDATE_TASKS

CORRECTED = {3: 76.0}
OUT_DIR = "results/revision"


def rescore(store, tasks_fixed):
    changed = []
    for rid, rec in store.items():
        t = int(re.search(r"_t(\d\d)_r\d\d$", rid).group(1))
        gold = tasks_fixed[t][1]
        fa = rec.get("final_answer")
        try:
            ok = fa not in (None, "None", "") and abs(float(fa) - float(gold)) < 1e-6
        except ValueError:
            ok = False
        new = "success" if ok else "failure"
        if new != rec["outcome"]:
            changed.append((rid, rec["outcome"], new))
            rec["outcome_old_key"] = rec["outcome"]
            rec["outcome"] = new
    return changed


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks_fixed = list(CANDIDATE_TASKS)
    for t, v in CORRECTED.items():
        tasks_fixed[t] = (tasks_fixed[t][0], v)
    assert tasks_fixed[3][1] == 76.0
    report = {}
    for label, traces, old in [("gpt-4o-mini", "results/new/intervention_traces.json", "results/new/intervention.json"),
                               ("qwen2.5-7b", "results/new/intervention_traces_qwen2.5-7b.json", "results/new/intervention_qwen2.5-7b.json")]:
        store = json.load(open(traces))
        changed = rescore(store, tasks_fixed)
        tasks = sorted({int(re.search(r"_t(\d\d)_r\d\d$", r).group(1)) for r in store})
        n = max(int(r.rsplit("_r", 1)[1]) for r in store) + 1
        with mock.patch.object(RI, "CANDIDATE_TASKS", tasks_fixed):
            res = RI.analyse(store, tasks, n)
        design = json.load(open(old))["design"]
        design.update({"answer_key_correction": {"t03": {"old": 84, "new": 76, "reason": "14*9-(14-4)*(9-4)=76"}},
                       "rescored_from": traces, "runs_relabelled": changed, "note": "model outputs not regenerated; outcomes recomputed from saved final answers"})
        res["design"] = design
        out = f"{OUT_DIR}/intervention_rescored_{label}.json"
        json.dump(res, open(out, "w"), indent=1, default=float)
        json.dump(store, open(f"{OUT_DIR}/intervention_traces_rescored_{label}.json", "w"), indent=0)
        P = res["pooled"]
        bt, bp = P["baseline_vs_targeted"], P["baseline_vs_placebo"]
        print(f"[{label}] relabelled {len(changed)} runs: {[c[0] for c in changed]}")
        print(f"[{label}] pooled fail: baseline {P['baseline']['fail']}/140 targeted {P['targeted']['fail']}/140 placebo {P['placebo']['fail']}/140 | "
              f"b-vs-t diff {bt['diff']:+.3f} [{bt['ci_lo']:+.3f},{bt['ci_hi']:+.3f}] p={bt['p_value']:.3f} holm={bt['p_holm']:.3f} | "
              f"b-vs-p diff {bp['diff']:+.3f} [{bp['ci_lo']:+.3f},{bp['ci_hi']:+.3f}] p={bp['p_value']:.3f} holm={bp['p_holm']:.3f}")
        print(f"[{label}] t03 row: {res['per_task']['t03']['baseline']['fail']}/{res['per_task']['t03']['targeted']['fail']}/{res['per_task']['t03']['placebo']['fail']}; wrote {out}")
        report[label] = {"relabelled": changed, "pooled": {c: P[c]["fail"] for c in ["baseline", "targeted", "placebo"]}}
    json.dump(report, open(f"{OUT_DIR}/intervention_rescore_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
