"""Attribution methods on ALFWorld populations (no replay here).

Input: a traces JSON from run_alfworld_collect. For every reference-eligible
failing run (its game has >= 1 success) the script records, per method, the
predicted (agent, step) plus metadata:
  refalign      the unchanged numeric RefAlign (refalign.contrastive)
  refalign_s    structured comparator variant (refalign.structured)
  first_ref / random_ref-s{k} / edit_ref : RefAlign-S with other reference rules
  judge_aao     single-trace judge (gpt-4o-2024-08-06 unless overridden)
  judge_matched reference-conditioned judge, reference = RefAlign-S's reference
  judge_first / judge_random-s{k} / judge_edit : same judge, other references
  last_step, penultimate, random_step-s{k}
Positional baselines that need labels (mode step, normalized position) are
fitted later, leave-one-task-type-out, in the analysis script once replay labels
exist. Runs under the system python (no TextWorld needed).

    PYTHONPATH=. python -m experiments.run_alfworld_attribution --traces results/revision/alf_test_traces.json \
        --out results/revision/alf_test_attribution.json --judge gpt-4o-2024-08-06
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from refalign.contrastive import contrastive_localize, ContrastiveConfig, _encode_actions, _nearest_sibling
from refalign.structured import structured_localize, nearest_reference, StructuredConfig
from refalign.llm_baselines import all_at_once, matched_judge, parse_judge_answer, strict_step_parsed
from refalign.llm_logged import LoggedLLM
from refalign.types import Step, Trace, RunPopulation


def det_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


def load_pops(traces_path):
    store = json.load(open(traces_path))
    by = {}
    for rid, r in store.items():
        by.setdefault(r["game_id"], []).append(r)
    pops = {}
    for gid, runs in by.items():
        trs = [Trace(r["run_id"], gid, [Step(s["index"], s["agent"], s["action"], s["content"]) for s in r["steps"]],
                     r["outcome"] == "success", query=r["goal"], final_answer=r["final_answer"]) for r in runs]
        pops[gid] = RunPopulation(gid, trs, runs[0]["goal"])
    return store, pops


def pred_rec(p, ref_id=None, **extra):
    return {"agent": p.agent, "step": int(p.step), "reference": ref_id, "kind": p.meta.get("kind") or p.meta.get("fallback"),
            "abstain": bool(p.meta.get("abstain", False)), **extra}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default="gpt-4o-2024-08-06")
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="seeds for the free deterministic random-reference / random-step variants")
    ap.add_argument("--judge-seeds", type=int, nargs="+", default=[0, 1], help="seeds for the paid random-reference judge variant")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    store, pops = load_pops(args.traces)
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    scfg = StructuredConfig()
    jobs = []
    for gid, pop in pops.items():
        if not pop.successes:
            continue
        for f in pop.failures:
            rec = out.get(f.run_id, {"game_id": gid, "task_type": store[f.run_id]["task_type"], "n_steps": len(f),
                                     "n_actor_steps": len(f) - 1, "n_successes": len(pop.successes)})
            # deterministic methods (recomputed every time; cheap)
            p = contrastive_localize(pop, f)
            rec["refalign"] = pred_rec(p, p.meta.get("sibling"))
            ref_s = nearest_reference(f, pop, scfg)
            ps = structured_localize(pop, f, scfg, reference=ref_s)
            rec["refalign_s"] = pred_rec(ps, ref_s.run_id, required=ps.meta.get("required"), n_candidates=ps.meta.get("n_candidates", 0))
            ref_first = pop.successes[0]
            rec["refalign_s_first"] = pred_rec(structured_localize(pop, f, scfg, reference=ref_first), ref_first.run_id)
            alpha, seqs = _encode_actions(pop)
            ref_edit = _nearest_sibling(f, pop.successes, alpha, seqs, ContrastiveConfig(sibling="edit"))
            rec["refalign_s_edit"] = pred_rec(structured_localize(pop, f, scfg, reference=ref_edit), ref_edit.run_id)
            for s in args.seeds:
                rng = np.random.RandomState(det_seed("alf-randref", f.run_id, s))
                rr = pop.successes[int(rng.randint(len(pop.successes)))]
                rec[f"refalign_s_random-s{s}"] = pred_rec(structured_localize(pop, f, scfg, reference=rr), rr.run_id)
                rng2 = np.random.RandomState(det_seed("alf-randstep", f.run_id, s))
                st = int(rng2.randint(1, len(f)))            # uniform over Actor steps 1..len-1
                rec[f"random_step-s{s}"] = {"agent": f.agent_at(st), "step": st, "reference": None}
            rec["last_step"] = {"agent": f.agent_at(len(f) - 1), "step": len(f) - 1, "reference": None}
            rec["penultimate"] = {"agent": f.agent_at(max(0, len(f) - 2)), "step": max(0, len(f) - 2), "reference": None}
            rec["reference_pool"] = [s_.run_id for s_ in pop.successes]
            out[f.run_id] = rec
            if not args.no_judge:
                jobs.append((f, pop, ref_s, ref_first, ref_edit))
    print(f"{len(out)} reference-eligible failing runs over {len({r['game_id'] for r in out.values()})} games; judge jobs {len(jobs)}", flush=True)
    if jobs:
        log = args.log or args.out.replace(".json", "_api_log.jsonl")
        llm = LoggedLLM(args.judge, base_url=args.judge_base_url, temperature=0.0, max_tokens=128, log_path=log, tag=f"alf-judge|{args.judge}")

        def work(job):
            f, pop, ref_s, ref_first, ref_edit = job
            res = {}
            v = llm.view(tag=f"alf-judge|aao|{f.run_id}")
            if "judge_aao" not in out[f.run_id]:
                a = all_at_once(f, v)
                ag, st, sp, ap_ = parse_judge_answer(v.last_raw, f)
                if sp and not strict_step_parsed(v.last_raw):
                    a.agent, a.step = ag, st
                res["judge_aao"] = {"agent": a.agent, "step": int(a.step), "step_parsed": sp, "tokens": sum(v.last_tokens), "raw": v.last_raw[:200]}
            refs = {"judge_matched": ref_s, "judge_first": ref_first, "judge_edit": ref_edit}
            for s in args.judge_seeds:
                rng = np.random.RandomState(det_seed("alf-randref", f.run_id, s))
                refs[f"judge_random-s{s}"] = pop.successes[int(rng.randint(len(pop.successes)))]
            for name, ref in refs.items():
                if name in out[f.run_id]:
                    continue
                v = llm.view(tag=f"alf-judge|{name}|{f.run_id}")
                m = matched_judge(f, ref, v)
                res[name] = {"agent": m.agent, "step": int(m.step), "reference": ref.run_id, "step_parsed": m.meta["step_parsed"],
                             "tokens": sum(v.last_tokens), "raw": m.meta["raw"][:200]}
            return f.run_id, res

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for k, (rid, res) in enumerate(ex.map(work, jobs), 1):
                out[rid].update(res)
                if k % 25 == 0 or k == len(jobs):
                    json.dump(out, open(args.out, "w"), indent=0)
                    print(f"judge {k}/{len(jobs)}: {llm.usage.calls} calls, {llm.usage.prompt_tokens + llm.usage.completion_tokens} tokens", flush=True)
    json.dump(out, open(args.out, "w"), indent=0)
    # quick descriptive summary
    kinds = Counter(r["refalign_s"]["kind"] for r in out.values()); abst = sum(r["refalign_s"]["abstain"] for r in out.values())
    print("refalign_s kinds", dict(kinds), "abstain", abst, "/", len(out))
    print("refalign (numeric) kinds", dict(Counter(r["refalign"]["kind"] for r in out.values())))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
