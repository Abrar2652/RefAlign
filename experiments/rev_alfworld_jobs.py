"""Build replay jobs for ALFWorld failing runs.

Two job families:
  profile : expert override at a step grid (every Actor step up to --kmax) for a
            random subset of reference-eligible failures (labels: earliest
            repairable step); modes = ['expert'].
  methods : the step selected by each attribution method (from the attribution
            JSON) for every reference-eligible failure; modes = ['expert',
            'repair', 'noedit'] so that localization validity (A), practical
            repair (B) and the resampling control share identical prefixes.
Steps are trace indices (step 0 = Planner). A method step beyond the run's last
Actor step cannot be replayed and is recorded as 'not_replayable' (counted as a
failed intervention downstream).

    PYTHONPATH=. python -m experiments.rev_alfworld_jobs --attribution results/revision/alf_test_attribution.json \
        --traces results/revision/alf_test_traces.json --out results/revision/alf_replay_jobs.json \
        --profile-n 60 --kmax 12
"""
import argparse
import json
import zlib

import numpy as np

METHODS = ["refalign", "refalign_s", "refalign_s_first", "refalign_s_edit", "refalign_s_random-s0",
           "judge_aao", "judge_matched", "judge_first", "judge_edit", "judge_random-s0",
           "last_step", "penultimate", "random_step-s0"]


def det_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile-n", type=int, default=60)
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--method-subset-n", type=int, default=0, help="replay method-selected steps on a seeded random subset of eligible failures (0 = all)")
    args = ap.parse_args(argv)
    att = json.load(open(args.attribution))
    store = json.load(open(args.traces))
    rids = sorted(att)
    rng = np.random.RandomState(args.seed)
    prof = set(rng.choice(rids, size=min(args.profile_n, len(rids)), replace=False).tolist()) if args.profile_n else set()
    # method-replay subset: the profiled runs first, then more eligible failures up to the requested size
    if args.method_subset_n and args.method_subset_n < len(rids):
        rest = [r for r in rids if r not in prof]
        extra = rng.choice(rest, size=max(0, args.method_subset_n - len(prof)), replace=False).tolist() if len(prof) < args.method_subset_n else []
        msub = set(prof) | set(extra)
    else:
        msub = set(rids)
    jobs = {}
    not_replayable = []
    for rid in rids:
        n_act = len(store[rid]["actions"])
        for m in (args.methods if rid in msub else []):
            if m not in att[rid]:
                continue
            st = int(att[rid][m]["step"])
            if st > n_act:
                not_replayable.append((rid, m, st)); continue
            key = (rid, st)
            jobs.setdefault(key, {"run_id": rid, "step": st, "modes": set(), "selected_by": []})
            jobs[key]["modes"] |= {"expert", "repair", "noedit"} if st > 0 else {"replan"}
            jobs[key]["selected_by"].append(m)
        if rid in prof:
            for k in range(1, min(n_act, args.kmax) + 1):
                key = (rid, k)
                jobs.setdefault(key, {"run_id": rid, "step": k, "modes": set(), "selected_by": []})
                jobs[key]["modes"] |= {"expert", "noedit"}; jobs[key]["selected_by"].append("profile")
    out = [{**j, "modes": sorted(j["modes"])} for j in jobs.values()]
    n_eps = sum(len(j["modes"]) for j in out)
    json.dump({"jobs": out, "profile_runs": sorted(prof), "method_subset": sorted(msub), "not_replayable": not_replayable, "kmax": args.kmax}, open(args.out, "w"), indent=0)
    print(f"{len(out)} (run, step) jobs, {n_eps} episodes per seed; profile runs {len(prof)}; method subset {len(msub)}; not replayable {len(not_replayable)}")
    return 0


if __name__ == "__main__":
    main()
