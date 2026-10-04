"""Phase 3: evaluate RefAlign on the annotated populated benchmark.

Loads traces + adjudicated labels, runs RefAlign (no LLM) and baselines,
and scores against the adjudicated decisive-error labels.

Usage:
    python -m annotation.run_annotated --traces annotation/traces_scale \
        --labels annotation/annotations_human/adjudicated
"""
import argparse
import sys

from refalign.data.schema import load_annotated_benchmark
from refalign.contrastive import contrastive_localize
from refalign.baselines import run_baseline
from refalign import evaluate as ev


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="annotation/traces")
    ap.add_argument("--labels", required=True, help="adjudicated annotator dir")
    args = ap.parse_args(argv)

    pops, anns = load_annotated_benchmark(args.traces, args.labels)
    n_fail = sum(len(p.failures) for p in pops)
    print(f"loaded {len(pops)} populations, {n_fail} failing runs, "
          f"{len(anns)} adjudicated decisive-error labels")
    if not anns:
        print("no labeled runs yet — annotate first."); return 1

    named = {"RefAlign (no LLM)": {f.run_id: contrastive_localize(p, f)
                                     for p in pops for f in p.failures}}
    for b in ["most_active_agent", "last_step", "random"]:
        named[b] = run_baseline(b, pops, seed=0)

    rows = ev.leaderboard(named, anns, tolerances=(1,), B=2000)
    print("\n=== RefAlign on annotated real populations ===")
    print(ev.format_leaderboard(rows, tolerances=(1,)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
