"""Phase 2: validate annotations and measure inter-annotator agreement.

Checks every label for schema/consistency errors against the traces, then, when
two annotators are given, reports Cohen's kappa on the responsible agent and exact
+ tolerant agreement on the decisive step.

Usage:
    python -m annotation.validate_annotations --traces annotation/traces \
        --a annotation/annotations_human/annotator_01 --b annotation/annotations_human/annotator_02
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

from annotation.annotate import FAULT_TYPES


def _load_traces(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        obj = json.load(open(f))
        out[obj["query_id"]] = obj
    return out


def _load_labels(d):
    out = {}
    if not d or not os.path.isdir(d):
        return out
    for f in glob.glob(os.path.join(d, "*.json")):
        obj = json.load(open(f))
        out.update(obj["labels"])
    return out


def validate(traces, labels, who):
    runs = {r["run_id"]: (obj, r) for obj in traces.values() for r in obj["runs"]}
    errors = 0
    n_ok = n_flag = 0
    for rid, lab in labels.items():
        if lab.get("flag"):
            n_flag += 1
            continue
        if rid not in runs:
            print(f"  [{who}] ERROR {rid}: run not found in traces"); errors += 1; continue
        obj, run = runs[rid]
        roster = {s["agent"] for s in run["steps"]}
        n = len(run["steps"])
        if lab.get("responsible_agent") not in roster:
            print(f"  [{who}] ERROR {rid}: agent '{lab.get('responsible_agent')}' not in {roster}"); errors += 1
        st = lab.get("decisive_step")
        if not isinstance(st, int) or not (0 <= st < n):
            print(f"  [{who}] ERROR {rid}: step {st} out of range 0..{n-1}"); errors += 1
        if lab.get("fault_type") not in FAULT_TYPES:
            print(f"  [{who}] ERROR {rid}: bad fault_type '{lab.get('fault_type')}'"); errors += 1
        if lab.get("confidence") not in (1, 2, 3, 4, 5):
            print(f"  [{who}] ERROR {rid}: confidence {lab.get('confidence')} not 1..5"); errors += 1
        if run["outcome"] != "failure":
            print(f"  [{who}] ERROR {rid}: labeled a non-failing run"); errors += 1
        n_ok += 1
    print(f"  [{who}] {n_ok} valid, {n_flag} flagged, {errors} errors")
    return errors


def cohen_kappa(a, b):
    keys = sorted(set(a) & set(b))
    if not keys:
        return float("nan"), 0
    xa = [a[k] for k in keys]
    xb = [b[k] for k in keys]
    n = len(keys)
    po = sum(1 for i in range(n) if xa[i] == xb[i]) / n
    ca, cb = Counter(xa), Counter(xb)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in set(xa) | set(xb))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return kappa, n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="annotation/traces")
    ap.add_argument("--a", required=True, help="annotator A dir")
    ap.add_argument("--b", default=None, help="annotator B dir (for agreement)")
    args = ap.parse_args(argv)

    traces = _load_traces(args.traces)
    la = _load_labels(args.a)
    print(f"validating {os.path.basename(args.a)} ...")
    err = validate(traces, la, os.path.basename(args.a))
    if args.b:
        lb = _load_labels(args.b)
        print(f"validating {os.path.basename(args.b)} ...")
        err += validate(traces, lb, os.path.basename(args.b))
        common = [r for r in set(la) & set(lb)
                  if not la[r].get("flag") and not lb[r].get("flag")]
        agent_a = {r: la[r]["responsible_agent"] for r in common}
        agent_b = {r: lb[r]["responsible_agent"] for r in common}
        ka, n = cohen_kappa(agent_a, agent_b)
        step_a = {r: la[r]["decisive_step"] for r in common}
        step_b = {r: lb[r]["decisive_step"] for r in common}
        ks, _ = cohen_kappa(step_a, step_b)
        exact = sum(1 for r in common if step_a[r] == step_b[r]) / max(1, n)
        tol1 = sum(1 for r in common if abs(step_a[r] - step_b[r]) <= 1) / max(1, n)
        print(f"\n=== inter-annotator agreement (n={n} shared runs) ===")
        print(f"  agent: Cohen's kappa = {ka:.3f}")
        print(f"  step : exact = {exact:.3f}, within +-1 = {tol1:.3f}, kappa = {ks:.3f}")
        print("  (target: agent kappa >= 0.6; resolve disagreements in adjudication)")
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
