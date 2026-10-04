"""Phase 0 of the annotation pipeline: collect REAL multi-run populations.

Runs a real Planner->Worker->Verifier agent system many times per task at a
non-zero temperature so that some runs succeed and some fail *naturally* (no
injected faults). Populations with at least one success and at least two failures
are saved to annotation/traces/<qid>.json for human annotation.

Usage:
    python -m annotation.collect_traces --runs 12 --temperature 0.9 \
        --min-success 1 --min-failure 2 --out annotation/traces
"""
import argparse
import json
import os
import sys

from refalign.llm import APILLM
from experiments.real_agents import run_once

# Harder multi-step word problems -> more natural failures than single ops.
TASKS = [
    ("A store had 240 apples. It sold 35% on Monday and 40 more on Tuesday. "
     "How many apples remain?", 116),
    ("A tank holds 900 L. It is 2/3 full, then 150 L are added. How many liters "
     "are in the tank now?", 750),
    ("A worker earns $18/hour for 40 hours and 1.5x that rate for 6 overtime "
     "hours. What is the total pay in dollars?", 882),
    ("A rectangle is 14 m by 9 m. A 2 m wide path runs around its inside edge. "
     "What is the area of the path in square meters?", 76),   # corrected 2026-10-01: 14*9-(14-4)*(9-4)=76 (was 84)
    ("Three friends split a $150 bill: Ann pays twice as much as Bo, and Cy pays "
     "$10 more than Bo. How much does Ann pay in dollars?", 70),
    ("A car travels 210 km using 15 L of fuel. At the same efficiency, how many "
     "liters are needed for 350 km?", 25),
    ("A book club read 3 books averaging 280 pages; a 4th book has 360 pages. "
     "What is the average pages across all 4 books?", 300),
    ("A phone costs $600. Its price drops 15%, then rises 10% of the new price. "
     "What is the final price in dollars?", 561),
    ("A recipe for 4 servings needs 320 g flour. How many grams are needed for "
     "7 servings?", 560),
    ("A train leaves at 09:20 and arrives at 13:05. The trip is 300 km. What is "
     "the average speed in km/h (rounded down)?", 80),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=12, help="runs per task")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--min-success", type=int, default=1)
    ap.add_argument("--min-failure", type=int, default=2)
    ap.add_argument("--tasks", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default="annotation/traces")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    llm = APILLM(provider="openai", model=args.model, temperature=args.temperature)
    tasks = TASKS[: args.tasks] if args.tasks else TASKS
    kept = 0
    for ti, (task, answer) in enumerate(tasks):
        qid = f"real_t{ti:02d}"
        runs = []
        n_succ = n_fail = 0
        for r in range(args.runs):
            tr, ok, _, _ = run_once(llm, task, float(answer), temperature=args.temperature)
            runs.append({
                "run_id": f"{qid}_r{r:02d}",
                "outcome": "success" if ok else "failure",
                "final_answer": tr.final_answer,
                "steps": [{"index": s.index, "agent": s.agent, "content": s.content}
                          for s in tr.steps],
            })
            n_succ += ok
            n_fail += (not ok)
        if n_succ >= args.min_success and n_fail >= args.min_failure:
            obj = {"query_id": qid, "query": task, "ground_truth": str(answer),
                   "runs": runs}
            with open(os.path.join(args.out, f"{qid}.json"), "w") as fh:
                json.dump(obj, fh, indent=1)
            kept += 1
            print(f"  {qid}: {n_succ} success / {n_fail} failure -> SAVED", flush=True)
        else:
            print(f"  {qid}: {n_succ} success / {n_fail} failure -> skipped "
                  f"(need >={args.min_success} succ, >={args.min_failure} fail)", flush=True)
    print(f"\nkept {kept} annotatable populations in {args.out}")
    print(f"LLM cost: {llm.usage.calls} calls, {llm.usage.seconds:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
