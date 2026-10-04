"""Open-model judge on the chain head-to-head (API-free, reproducible).

Runs a LOCAL open model (default Qwen2.5-7B-Instruct, from the hf_cache) as the
single-trace All-at-Once judge across the length gradient, and compares to
contrastive localization (no LLM). Run under a torch>=2.1 python, e.g.:

    HF_HOME=$HF_HOME \
    /path/to/workspace/ibeq_venv/bin/python -m experiments.run_local_judge \
        --model Qwen/Qwen2.5-7B-Instruct --device cuda:6
"""
import argparse, json, sys
import numpy as np
from refalign.llm import LocalLLM
from refalign.llm_baselines import all_at_once
from refalign.contrastive import contrastive_localize
from experiments.chain_agents import build_population


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--lengths", type=int, nargs="+", default=[5, 12, 24, 40])
    ap.add_argument("--tasks", type=int, default=6)
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--attn", default=None, help='e.g. "eager" for gemma-2')
    ap.add_argument("--out", default="results/local_judge.json")
    args = ap.parse_args(argv)

    llm = LocalLLM(model=args.model, device=args.device, max_new_tokens=args.max_new,
                   attn_implementation=args.attn)
    out = {"model": args.model, "lengths": {}}
    print(f"{'L':>3} {'contrastive':>12} {'all_at_once(local)':>20} {'n':>4}")
    for L in args.lengths:
        pops, anns = [], {}
        for t in range(args.tasks):
            p, a = build_population(None, t, L, seed=0, noise=True)
            pops.append(p); anns.update(a)
        fails = [(p, f) for p in pops for f in p.failures if f.run_id in anns]
        c = np.mean([contrastive_localize(p, f).step == anns[f.run_id].decisive_step
                     for p, f in fails])
        aao = {f.run_id: all_at_once(f, llm) for _p, f in fails}
        a_step = np.mean([aao[f.run_id].step == anns[f.run_id].decisive_step for _p, f in fails])
        a_agent = np.mean([aao[f.run_id].agent == anns[f.run_id].responsible_agent for _p, f in fails])
        out["lengths"][f"L{L}"] = {"contrastive_step": float(c), "aao_step": float(a_step),
                                   "aao_agent": float(a_agent), "n": len(fails)}
        print(f"{L:>3} {c:>12.2f} {a_step:>20.2f} {len(fails):>4}", flush=True)
    out["usage"] = {"calls": llm.usage.calls, "seconds": llm.usage.seconds,
                    "prompt_tokens": llm.usage.prompt_tokens,
                    "completion_tokens": llm.usage.completion_tokens}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nlocal judge: {llm.usage.calls} calls, {llm.usage.seconds:.0f}s, "
          f"{llm.usage.prompt_tokens+llm.usage.completion_tokens} tokens (0 API cost)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
