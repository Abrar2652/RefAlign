"""Dump per-run predictions on the human-annotated benchmark to a JSON the figure
script can read (keeps figures reproducible from data files).

For every human-labeled failing run: the task, the human gold (agent, step), and
each method's predicted (agent, step). Powers the error-concentration and
step-error-distribution figures.
"""
import json
import numpy as np

from refalign.data.schema import load_annotated_benchmark
from refalign.contrastive import contrastive_localize
from refalign.baselines import run_baseline

pops, anns = load_annotated_benchmark("annotation/traces_scale",
                                      "annotation/annotations_human/adjudicated")
phylo = {f.run_id: contrastive_localize(p, f) for p in pops for f in p.failures}
base = {b: run_baseline(b, pops, seed=0) for b in ["most_active_agent", "last_step", "random"]}

rows = []
for rid in sorted(anns):
    g = anns[rid]
    task = rid.rsplit("_r", 1)[0]
    rec = {"run_id": rid, "task": task,
           "human_agent": g.responsible_agent, "human_step": g.decisive_step,
           "phylo_agent": phylo[rid].agent, "phylo_step": phylo[rid].step}
    for b, preds in base.items():
        rec[f"{b}_step"] = preds[rid].step
        rec[f"{b}_agent"] = preds[rid].agent
    rows.append(rec)

out = {"n": len(rows), "rows": rows}
json.dump(out, open("results/human_perrun.json", "w"), indent=1)
# quick sanity summary
for m, sk in [("phylo", "phylo_step"), ("most_active_agent", "most_active_agent_step"),
              ("last_step", "last_step_step"), ("random", "random_step")]:
    err = [abs(r[sk] - r["human_step"]) for r in rows]
    print(f"{m:18s} |pred-human| mean={np.mean(err):.2f}  exact={np.mean([e==0 for e in err]):.2f}  <=1={np.mean([e<=1 for e in err]):.2f}")
print("wrote results/human_perrun.json  (n=%d)" % len(rows))
