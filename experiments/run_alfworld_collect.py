"""Collect Planner->Actor rollout populations on ALFWorld games.

    ALFWORLD_DATA=... PYTHONPATH=. \
    /path/to/workspace/alf_venv/bin/python -m experiments.run_alfworld_collect \
        --split valid_seen --n-games 40 --game-seed 0 --n 10 \
        --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8103/v1 \
        --out results/revision/alf_dev_traces.json

Every episode is seeded (crc32('alf|<split>|<game_id>|<rollout>')) with one seed
per LLM call; every call is logged by LoggedLLM. The output stores, per run, the
typed trace, the action list, the expert action available at each step (for
replay), the plan, and bookkeeping (invalid actions, cap hits, calls, tokens).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments.alfworld_env import AlfEnv, sample_games, list_games, TASK_TYPES
from experiments.alfworld_agents import run_episode, trace_to_json
from refalign.llm_logged import LoggedLLM


def det_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


_LLM = None
_LLM_KW = None


def _init(kw):
    """Per-process LLM client with its own log file (TextWorld's parser is not
    thread-safe, so episodes run in separate processes)."""
    global _LLM, _LLM_KW
    _LLM_KW = dict(kw)
    _LLM_KW["log_path"] = f"{kw['log_path']}.p{os.getpid()}.jsonl"
    _LLM = LoggedLLM(**_LLM_KW)


def one(args):
    game, split, i, max_steps = args
    llm = _LLM
    env = AlfEnv(expert=True)
    seed = det_seed("alf", split, game["game_id"], i)
    v = llm.view(seed=seed, tag=f"alf|{split}|{game['task_type']}|{game['game_id']}|r{i:02d}")
    t0 = time.time()
    try:
        ep = run_episode(env, v, game["game_file"], max_steps=max_steps, task_type=game["task_type"])
    finally:
        env.close()
    rid = f"alf_{split}_{game['game_id'].replace('/', '__')}_r{i:02d}"
    rec = trace_to_json(ep["trace"], rid, {
        "game_id": game["game_id"], "game_file": game["game_file"], "task_type": game["task_type"], "split": split, "rollout": i,
        "goal": ep["trace"].query, "plan": ep["plan"], "actions": ep["actions"], "expert_actions": ep["experts"],
        "n_actor_steps": ep["n_actor_steps"], "invalid_actions": ep["invalid_actions"], "hit_cap": ep["hit_cap"],
        "seed_base": seed, "seeds_used": v.seeds_used, "n_llm_calls": v.usage.calls,
        "tokens": v.usage.prompt_tokens + v.usage.completion_tokens, "response_models": sorted(v.response_models),
        "wall_s": round(time.time() - t0, 1)})
    return rec


def funnel(store):
    by_game = {}
    for r in store.values():
        by_game.setdefault(r["game_id"], []).append(r)
    n_games = len(by_game)
    n_runs = len(store)
    n_succ = sum(r["outcome"] == "success" for r in store.values())
    n_fail = n_runs - n_succ
    mixed = sum(1 for v in by_game.values() if any(r["outcome"] == "success" for r in v) and any(r["outcome"] == "failure" for r in v))
    all_s = sum(1 for v in by_game.values() if all(r["outcome"] == "success" for r in v))
    all_f = sum(1 for v in by_game.values() if all(r["outcome"] == "failure" for r in v))
    elig = sum(1 for v in by_game.values() if any(r["outcome"] == "success" for r in v) for r in v if r["outcome"] == "failure")
    by_type = {}
    for t in TASK_TYPES:
        rs = [r for r in store.values() if r["task_type"] == t]
        if rs:
            gs = {r["game_id"] for r in rs}
            by_type[t] = {"games": len(gs), "runs": len(rs), "success": sum(r["outcome"] == "success" for r in rs),
                          "games_mixed": sum(1 for g in gs if any(r["outcome"] == "success" for r in rs if r["game_id"] == g) and any(r["outcome"] == "failure" for r in rs if r["game_id"] == g))}
    return {"games": n_games, "runs": n_runs, "successes": n_succ, "failures": n_fail, "games_mixed": mixed,
            "games_all_success": all_s, "games_all_failure": all_f, "reference_eligible_failures": elig,
            "failures_without_reference": n_fail - elig, "hit_cap": sum(r["hit_cap"] for r in store.values()),
            "invalid_actions_total": sum(r["invalid_actions"] for r in store.values()),
            "mean_actor_steps": sum(r["n_actor_steps"] for r in store.values()) / max(1, n_runs), "by_task_type": by_type}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid_seen")
    ap.add_argument("--n-games", type=int, default=40, help="<=0 for all")
    ap.add_argument("--game-seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=10, help="rollouts per game")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    if not args.base_url or args.base_url.lower() in ("none", "openai"):
        args.base_url = None
    games = sample_games(args.split, args.n_games, args.game_seed)
    store = json.load(open(args.out)) if os.path.exists(args.out) else {}
    log = args.log or args.out.replace(".json", "_api_log.jsonl")
    kw = dict(model=args.model, base_url=args.base_url, temperature=args.temperature, max_tokens=args.max_tokens,
              log_path=log, tag=f"alf-collect|{args.split}", revision=args.revision)
    todo = [(g, args.split, i, args.max_steps) for g in games for i in range(args.n)
            if f"alf_{args.split}_{g['game_id'].replace('/', '__')}_r{i:02d}" not in store]
    print(f"{len(games)} games ({Counter(g['task_type'] for g in games)}), {len(todo)} episodes to run", flush=True)
    t0 = time.time()
    tok = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(kw,)) as ex:
        futs = [ex.submit(one, j) for j in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            store[rec["run_id"]] = rec
            tok += rec["tokens"]
            if k % 20 == 0 or k == len(todo):
                f = funnel(store)
                print(f"{k}/{len(todo)} episodes, {time.time() - t0:.0f}s, success {f['successes']}/{f['runs']}, {tok} tokens this session", flush=True)
                json.dump(store, open(args.out, "w"), indent=0)
    json.dump(store, open(args.out, "w"), indent=0)
    # merge the per-process logs into the main log
    import glob
    with open(log, "a") as fh:
        for pf in sorted(glob.glob(log + ".p*.jsonl")):
            fh.write(open(pf).read()); os.remove(pf)
    meta = {"split": args.split, "n_games": len(games), "game_seed": args.game_seed, "rollouts_per_game": args.n,
            "max_steps": args.max_steps, "model": args.model, "revision": args.revision, "base_url": args.base_url,
            "temperature": args.temperature, "max_tokens": args.max_tokens, "games": games,
            "usage_this_session": {"calls": sum(store[r]["n_llm_calls"] for r in store), "tokens": sum(store[r]["tokens"] for r in store)},
            "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "funnel": funnel(store)}
    json.dump(meta, open(args.out.replace(".json", "_meta.json"), "w"), indent=1)
    print(json.dumps(meta["funnel"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
