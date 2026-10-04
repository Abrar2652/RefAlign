"""Head-to-head: contrastive counterfactual attribution vs a single-trace LLM
judge, across a trace-length gradient, with cost accounting.

Hypothesis: the single-trace judge (All-at-Once) degrades on long dependent
chains, while contrastive localisation (align to the nearest successful sibling,
take the earliest value-divergence) stays accurate — and is *free* (no LLM).

Corpus is built deterministically (calculator-tool Worker + templated Planner/
Verifier) so it is reproducible and free; the LLM is spent only on the attribution
methods being compared.
"""
import argparse
import json
import sys
import numpy as np

from refalign.llm import APILLM
from refalign.contrastive import contrastive_localize, contrastive_llm
from refalign.attribute import AttributionConfig, attribute_corpus
from refalign.baselines import run_baseline
from refalign.llm_baselines import all_at_once
from refalign import evaluate as ev
from experiments.chain_agents import build_population


def _score(preds, anns):
    rids = [r for r in preds if r in anns]
    a = np.array([preds[r].agent == anns[r].responsible_agent for r in rids], float)
    s = np.array([preds[r].step == anns[r].decisive_step for r in rids], float)
    return a, s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[5, 12, 24])
    ap.add_argument("--tasks", type=int, default=6)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--out", default="results/contrastive.json")
    args = ap.parse_args(argv)

    judge = APILLM(provider="openai", model=args.model, temperature=0.0)
    results = {}
    for length in args.lengths:
        pops, anns = [], {}
        for t in range(args.tasks):
            # noisy, non-identical siblings (formatting variation + benign indels)
            pop, a = build_population(None, t, length, seed=0, noise=True)
            pops.append(pop)
            anns.update(a)

        preds = {
            "contrastive (no LLM)": {f.run_id: contrastive_localize(pop, f)
                                     for pop in pops for f in pop.failures},
            "pure-structural": attribute_corpus(pops, AttributionConfig()),
            "last_step": run_baseline("last_step", pops, seed=0),
            "most_active_agent": run_baseline("most_active_agent", pops, seed=0),
        }
        # single-trace LLM judge
        c0 = judge.usage.completion_tokens + judge.usage.prompt_tokens
        preds["llm_all_at_once"] = {f.run_id: all_at_once(f, judge)
                                    for pop in pops for f in pop.failures if f.run_id in anns}
        aao_tok = judge.usage.prompt_tokens + judge.usage.completion_tokens - c0
        # contrastive + LLM adjudication
        c1 = judge.usage.prompt_tokens + judge.usage.completion_tokens
        preds["contrastive+LLM"] = {f.run_id: contrastive_llm(pop, f, judge)
                                    for pop in pops for f in pop.failures if f.run_id in anns}
        cllm_tok = judge.usage.prompt_tokens + judge.usage.completion_tokens - c1

        row = {}
        for name, pr in preds.items():
            a, s = _score(pr, anns)
            _, alo, ahi = ev.bootstrap_ci(a)
            _, slo, shi = ev.bootstrap_ci(s)
            row[name] = {"agent": float(a.mean()), "step": float(s.mean()),
                         "step_ci": [slo, shi], "n": int(len(s))}
        row["_tokens"] = {"llm_all_at_once": aao_tok, "contrastive+LLM": cllm_tok,
                          "contrastive (no LLM)": 0}
        results[f"L{length}"] = row

        print(f"\n=== chain length {length} ({row['llm_all_at_once']['n']} faults) ===")
        print(f"  {'method':22s} {'agent':>6s} {'step':>6s} {'step 95% CI':>16s} {'tokens':>8s}")
        for name in ["contrastive (no LLM)", "contrastive+LLM", "llm_all_at_once",
                     "pure-structural", "most_active_agent", "last_step"]:
            r = row[name]
            tok = row["_tokens"].get(name, "-")
            ci = f"[{r['step_ci'][0]:.2f},{r['step_ci'][1]:.2f}]"
            print(f"  {name:22s} {r['agent']:6.2f} {r['step']:6.2f} {ci:>16s} {str(tok):>8s}")

    json.dump(results, open(args.out, "w"), indent=2, default=float)
    print(f"\ntotal judge cost: {judge.usage.calls} calls, "
          f"{judge.usage.prompt_tokens + judge.usage.completion_tokens} tokens, "
          f"{judge.usage.seconds:.0f}s\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
