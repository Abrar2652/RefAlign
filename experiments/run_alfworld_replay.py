"""Controlled replay on ALFWorld failing runs (requires the ALFWorld venv).

For a failing run f with Actor actions a_1..a_K (trace steps 1..K; step 0 is the
Planner), a replay at Actor step k re-executes the deterministic prefix
a_1..a_{k-1} in the simulator and then:
  mode 'expert'  : applies the environment's expert action at step k (a known
                   valid correction), then the ORIGINAL policy model continues
                   with fresh seeds until success or the step budget (max 30
                   Actor steps in total, as in collection);
  mode 'repair'  : a repair model proposes the action at step k from the prefix
                   only (same prompt and budget for every attribution method),
                   then the policy continues;
  mode 'noedit'  : no override; the policy resamples from step k (control for
                   recovery by resampling alone).
A prediction at trace step 0 (the Planner) is replayed as a fresh plan plus
policy continuation from step 1 (mode 'replan'), which is a resample of the plan,
not an expert-verified correction; it is flagged.

Input: --traces (collection JSON) and --jobs (JSON list of {run_id, step, modes,
label}) produced by the analysis script (method-selected steps plus a profile
grid). Output keyed by run_id|step|mode|seed with {won, n_policy_steps, calls,
tokens, actions}. Episodes run in separate processes (TextWorld is not
thread-safe).

    ALFWORLD_DATA=... PYTHONPATH=. alf_venv/bin/python -m experiments.run_alfworld_replay \
        --traces results/revision/alf_test_traces.json --jobs results/revision/alf_replay_jobs.json \
        --out results/revision/alf_replay.json --seeds 0 1 --model gpt-4o-mini-2024-07-18 --base-url openai
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments.alfworld_env import AlfEnv
from experiments.alfworld_agents import run_episode, parse_action, actor_prompt, ACTOR_SYS, PLANNER_SYS, exemplars
from refalign.llm_logged import LoggedLLM

REPAIR_SYS = ("You are a repair assistant for a household robot in a text environment. The robot's next action at this "
              "point of the run is suspected to be the decisive mistake that made the run fail. Using only the task, the "
              "plan, the history so far and the admissible commands, choose the single best command for this step. "
              "Output exactly:\n  ACTION: <command>")

_LLM = None


def det_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


def _init(kw):
    global _LLM
    kw = dict(kw); kw["log_path"] = f"{kw['log_path']}.p{os.getpid()}.jsonl"
    _LLM = LoggedLLM(**kw)


def _replay(job):
    rec, step, mode, seed_idx, max_steps = job
    actions = rec["actions"]
    seed = det_seed("replay", rec["run_id"], step, mode, seed_idx)
    v = _LLM.view(seed=seed, tag=f"alf-replay|{mode}|{rec['run_id']}|s{step}|r{seed_idx}")
    env = AlfEnv(expert=True)
    t0 = time.time()
    try:
        if mode == "replan" or step == 0:
            ep = run_episode(env, v, rec["game_file"], max_steps=max_steps, plan=None, task_type=rec.get('task_type'))
            mode_used = "replan"
        else:
            k = step                         # Actor step k = trace step k
            prefix = actions[:k - 1]
            if mode == "expert":
                ep = run_episode(env, v, rec["game_file"], max_steps=max_steps, prefix_actions=prefix,
                                 override_at=k, override_action=None, plan=rec["plan"], task_type=rec.get("task_type"))
            elif mode == "noedit":
                ep = run_episode(env, v, rec["game_file"], max_steps=max_steps, prefix_actions=prefix,
                                 override_at=None, plan=rec["plan"], task_type=rec.get("task_type"))
            elif mode == "repair":
                # the repair model sees only the prefix (re-executed to obtain the state)
                obs, info = env.reset(rec["game_file"])
                history = []
                for a in prefix:
                    adm = info["admissible_commands"]
                    a2 = a if a in adm else parse_action(a, adm)[0]
                    obs, done, info = env.step(a2); history.append((a2, obs))
                adm = list(info["admissible_commands"])
                prompt = actor_prompt(env.goal, rec["plan"], history, obs, adm) + \
                    f"\n(The robot previously chose '{actions[k - 1]}' here; it is suspected to be the mistake.)"
                raw = v.chat(REPAIR_SYS + exemplars(rec.get('task_type')), prompt)
                fix, valid = parse_action(raw, adm)
                env.close(); env = AlfEnv(expert=True)
                ep = run_episode(env, v, rec["game_file"], max_steps=max_steps, prefix_actions=prefix,
                                 override_at=k, override_action=fix, plan=rec["plan"], task_type=rec.get("task_type"))
                ep["repair_action"] = fix; ep["repair_valid"] = valid; ep["original_action"] = actions[k - 1]
            else:
                raise ValueError(mode)
            mode_used = mode
    finally:
        env.close()
    return {"key": f"{rec['run_id']}|{step}|{mode}|{seed_idx}", "run_id": rec["run_id"], "step": step, "mode": mode_used,
            "seed_idx": seed_idx, "won": ep["won"], "n_actor_steps": ep["n_actor_steps"], "actions": ep["actions"],
            "sources": ep.get("sources"), "override_action": (ep["actions"][step - 1] if 0 < step <= len(ep["actions"]) else None),
            "repair_action": ep.get("repair_action"), "repair_valid": ep.get("repair_valid"), "original_action": ep.get("original_action"),
            "calls": v.usage.calls, "tokens": v.usage.prompt_tokens + v.usage.completion_tokens, "wall_s": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--jobs", required=True, help="JSON list of {run_id, step, modes:[...]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N jobs")
    args = ap.parse_args(argv)
    if not args.base_url or args.base_url.lower() in ("none", "openai"):
        args.base_url = None
    store = json.load(open(args.traces))
    jobs_in = json.load(open(args.jobs))
    if isinstance(jobs_in, dict):
        jobs_in = jobs_in["jobs"]
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    log = args.out.replace(".json", "_api_log.jsonl")
    kw = dict(model=args.model, base_url=args.base_url, temperature=args.temperature, max_tokens=args.max_tokens, log_path=log, tag="alf-replay")
    todo = []
    for j in jobs_in:
        rec = store[j["run_id"]]
        if "game_file" not in rec:
            raise KeyError("traces must carry game_file (collector >= this revision)")
        for mode in j["modes"]:
            for s in args.seeds:
                key = f"{j['run_id']}|{j['step']}|{mode}|{s}"
                if key not in out:
                    todo.append((rec, int(j["step"]), mode, s, args.max_steps))
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} replay episodes to run ({len(jobs_in)} jobs x modes x {len(args.seeds)} seeds)", flush=True)
    t0 = time.time(); tok = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(kw,)) as ex:
        futs = [ex.submit(_replay, j) for j in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result(); out[r["key"]] = r; tok += r["tokens"]
            if k % 25 == 0 or k == len(todo):
                json.dump(out, open(args.out, "w"), indent=0)
                won = sum(x["won"] for x in out.values())
                print(f"{k}/{len(todo)} replays, {time.time() - t0:.0f}s, won {won}/{len(out)}, {tok} tokens this session", flush=True)
    json.dump(out, open(args.out, "w"), indent=0)
    import glob
    with open(log, "a") as fh:
        for pf in sorted(glob.glob(log + ".p*.jsonl")):
            fh.write(open(pf).read()); os.remove(pf)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
