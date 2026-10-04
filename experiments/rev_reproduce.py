"""Evidence inventory and reproduction of the manuscript's headline numbers from
saved outputs (no LLM calls). Writes results/revision/reproduction.json and
prints a claim-by-claim check. Nothing under results/new is modified.

    python -m experiments.rev_reproduce
"""
import json
import os
from collections import Counter

from refalign.contrastive import contrastive_localize, ContrastiveConfig
from experiments.chain_agents import build_population
from experiments.bench import build_diverse, build_multiculprit
from experiments.run_reference_sensitivity import load_human

R = "results/new"
OUT = "results/revision/reproduction.json"
J = lambda p: json.load(open(p))


def check(name, observed, expected, note=""):
    ok = observed == expected
    print(f"[{'OK ' if ok else 'XX '}] {name}: observed {observed!r} expected {expected!r} {note}")
    return {"claim": name, "observed": observed, "expected": expected, "ok": ok, "note": note}


def main():
    rows = []
    # 1. 24/24 at each controlled length (recomputed from code, and from the saved matched-judge file)
    per_len = {}
    for L in [5, 12, 24, 40]:
        pops, anns = [], {}
        for t in range(6):
            p, a = build_population(None, t, L, seed=0, noise=True); pops.append(p); anns.update(a)
        preds = {f.run_id: contrastive_localize(p, f) for p in pops for f in p.failures}
        per_len[L] = sum(preds[r].step == anns[r].decisive_step for r in anns)
    rows.append(check("RefAlign 24/24 at L=5,12,24,40 (recomputed)", per_len, {5: 24, 12: 24, 24: 24, 40: 24},
                      "source: experiments/chain_agents.build_population seed 0 noise=True; evaluator: step == decisive_step"))
    mj = J(f"{R}/matched_judge.json")
    saved = {L: mj["summary"]["gpt-4o"][f"L{L}"]["phylo"]["step"]["c"] for L in [5, 12, 24, 40]}
    rows.append(check("RefAlign 24/24 at each length (saved matched_judge.json, phylo rows)", saved, {5: 24, 12: 24, 24: 24, 40: 24}))
    # 2. natural failures: 32 runs, 4 families
    pops, anns = load_human()
    fam = {p.query_id: len(p.failures) for p in pops}
    rows.append(check("natural failures = 32 runs from 4 families", (sum(fam.values()), len(fam)), (32, 4), f"per family {fam}"))
    # 3. RefAlign 15/32 exact step (recomputed) and from human_perrun.json
    preds = {f.run_id: contrastive_localize(p, f) for p in pops for f in p.failures}
    c_step = sum(preds[r].step == anns[r].decisive_step for r in anns)
    c_agent = sum(preds[r].agent == anns[r].responsible_agent for r in anns)
    rows.append(check("RefAlign exact step on human benchmark (recomputed)", f"{c_step}/32", "15/32"))
    hp = J(f"{R}/../human_perrun.json")["rows"]
    rows.append(check("RefAlign exact step (results/human_perrun.json)", f"{sum(r['phylo_step'] == r['human_step'] for r in hp)}/32", "15/32"))
    rows.append(check("RefAlign exact agent on human benchmark (recomputed)", f"{c_agent}/32", "17/32", "0.53 in the manuscript"))
    # 4. GPT-4o with RefAlign reference: 0.69 = 22/32
    P = J(f"{R}/matched_judge_paired.json")
    g = P["gpt-4o"]["human_n32"]
    c = sum(r["step_correct"] for r in g["matched-phylo"].values())
    rows.append(check("reference-conditioned GPT-4o exact step", f"{c}/32 = {c/32:.3f}", "22/32 = 0.688"))
    rows.append(check("GPT-4o single-trace AAO exact step (human)", f"{sum(r['step_correct'] for r in g['aao'].values())}/32", "16/32"))
    rows.append(check("GPT-4o matched-first exact step (human)", f"{sum(r['step_correct'] for r in g['matched-first'].values())}/32", "22/32"))
    # 5-6. all labels Worker; 27 at step 3
    agents = Counter(a.responsible_agent for a in anns.values()); steps = Counter(a.decisive_step for a in anns.values())
    rows.append(check("all human-labeled culprit agents are Workers", dict(agents), {"Worker": 32}))
    rows.append(check("labels at zero-based step 3", steps.get(3, 0), 27, f"full distribution {dict(sorted(steps.items()))}"))
    # post-hoc constant step-3 diagnostic
    const3 = sum(1 for a in anns.values() if a.decisive_step == 3)
    rows.append(check("POST-HOC diagnostic: constant 'step 3' prediction", f"{const3}/32 = {const3/32:.3f}", "27/32 = 0.844",
                      "not an independently selected baseline; leave-one-family-out baselines are in positional_baselines.json"))
    # 7. invariance across references
    rs = J(f"{R}/reference_sensitivity.json")
    rows.append(check("prediction invariant across references on all 32 natural failures", f"{rs['pooled']['invariant']['c']}/32", "32/32"))
    rows.append(check("nearest and first reference coincide", sum(1 for r in rs["per_run"].values() if r["nearest_reference"] == rs["families"][r["gold"] and [k for k, f in rs["families"].items() if any(rid in f["runs"] for rid in [list(rs["per_run"].keys())[0]])][0]]["first_reference"]) if False else sum(1 for fam in rs["families"].values() for r in fam["runs"].values() if r["nearest_reference"] == fam["first_reference"]), 9))
    # 8. non-numeric study: 347 reference-available failures, all fallback
    sv = J(f"{R}/semantic_variant.json")
    a = sv["attribution"]
    rows.append(check("non-numeric study: fallback on all reference-available failures", f"{a['phylo_fallback_rate']['c']}/{a['phylo_fallback_rate']['n']}", "347/347"))
    st = J(f"{R}/semantic_variant_traces.json")
    n_note = sum(any(s["action"] == "note" for s in r["steps"]) for r in st.values())
    rows.append(check("non-numeric study: runs containing a NOTE step", f"{n_note}/{len(st)}", "22/560"))
    # reorderable-sum and multi-culprit sanity
    div = {}
    for k in [8, 15]:
        pops_, anns_ = [], {}
        for t in range(6):
            p, a_ = build_diverse(t, k, seed=0); pops_.append(p); anns_.update(a_)
        for mode in ["edit", "content"]:
            pr = {f.run_id: contrastive_localize(p, f, ContrastiveConfig(sibling=mode)) for p in pops_ for f in p.failures}
            div[f"k{k}_{mode}"] = sum(pr[r].step == anns_[r].decisive_step for r in anns_)
    rows.append(check("reorderable sum (content-aware / edit) k=8,15", div, {"k8_edit": 13, "k8_content": 34, "k15_edit": 13, "k15_content": 36}, "n=36 each"))
    # t03 answer key state in code
    from annotation.collect_traces import TASKS
    rows.append(check("t03 answer key in annotation/collect_traces.py", TASKS[3][1], 76, "manuscript uses the corrected key 76; 84 is superseded"))
    # intervention counts in saved JSON (old key) vs manuscript (corrected key)
    iv = J(f"{R}/intervention.json")["pooled"]
    rows.append(check("saved intervention.json pooled failures (OLD key 84)", (iv["baseline"]["fail"], iv["targeted"]["fail"], iv["placebo"]["fail"]), (60, 75, 75),
                      "manuscript reports 52/74/73 after correcting t03 to 76; rescoring is in intervention_rescored_*.json"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"rows": rows, "n_ok": sum(r["ok"] for r in rows), "n": len(rows)}, open(OUT, "w"), indent=1, default=str)
    print(f"{sum(r['ok'] for r in rows)}/{len(rows)} checks passed; wrote {OUT}")


if __name__ == "__main__":
    main()
