"""Phase 1: interactive annotation tool.

Presents each failing run alongside a successful run of the same task (the
counterfactual) and records the decisive error with validated input, so an
annotator cannot produce a malformed label. Saves after every run (resumable).

Usage:
    python -m annotation.annotate --annotator annotator_01 \
        --traces annotation/traces --out annotation/annotations
"""
import argparse
import glob
import json
import os
import sys

FAULT_TYPES = [
    "wrong_computation",   # a step produces an incorrect value/result
    "wrong_operation",     # uses the wrong operation / tool / approach
    "missing_step",        # skips a necessary step
    "repeated_step",       # repeats or loops without making progress
    "misplan",             # the plan itself is wrong or incomplete
    "premature_stop",      # stops or answers before finishing
    "bad_verification",    # verifier accepts a wrong answer / rejects a right one
    "other",
]


def _ask(prompt, validate, err):
    while True:
        raw = input(prompt).strip()
        ok, value = validate(raw)
        if ok:
            return value
        print(f"  ! {err}")


def _print_run(run, header):
    print(f"\n{header}  [{run['run_id']}]  final_answer={run.get('final_answer')}")
    for s in run["steps"]:
        print(f"    [{s['index']:>2}] {s['agent']}: {s['content'][:100]}")


def annotate_population(obj, annotator, out_labels, out="annotation/annotations"):
    runs = obj["runs"]
    roster = sorted({s["agent"] for r in runs for s in r["steps"]})
    successes = [r for r in runs if r["outcome"] == "success"]
    failures = [r for r in runs if r["outcome"] == "failure"]
    print("\n" + "=" * 78)
    print(f"TASK {obj['query_id']}: {obj['query']}")
    print(f"GROUND TRUTH: {obj.get('ground_truth','?')}   agents: {roster}")
    if successes:
        _print_run(successes[0], "REFERENCE (a SUCCESSFUL run)")

    for fr in failures:
        rid = fr["run_id"]
        if rid in out_labels:
            continue                              # resume: already done
        _print_run(fr, "FAILING run to annotate")
        n = len(fr["steps"])
        print("\n  Enter 's' to skip/flag this run as ambiguous.")
        first = input("  annotate this run? [Enter=yes / s=skip]: ").strip().lower()
        if first == "s":
            out_labels[rid] = {"flag": "ambiguous", "annotator": annotator}
            _save(out_labels, obj, annotator, out)
            continue

        agent = _ask(
            f"  responsible agent {roster}: ",
            lambda x: (x in roster, x), f"must be one of {roster}")
        step = _ask(
            f"  decisive step index (0..{n-1}): ",
            lambda x: (x.isdigit() and 0 <= int(x) < n, int(x) if x.isdigit() else None),
            f"must be an integer in 0..{n-1}")
        print("  fault types: " + ", ".join(f"{i}={t}" for i, t in enumerate(FAULT_TYPES)))
        ft = _ask(
            "  fault type number: ",
            lambda x: (x.isdigit() and 0 <= int(x) < len(FAULT_TYPES),
                       FAULT_TYPES[int(x)] if x.isdigit() and int(x) < len(FAULT_TYPES) else None),
            f"must be 0..{len(FAULT_TYPES)-1}")
        conf = _ask(
            "  confidence (1=low .. 5=high): ",
            lambda x: (x in list("12345"), int(x) if x in list("12345") else None),
            "must be 1..5")
        reason = input("  one-line reason: ").strip()
        out_labels[rid] = {"responsible_agent": agent, "decisive_step": step,
                           "fault_type": ft, "confidence": conf, "reason": reason,
                           "annotator": annotator, "flag": None}
        _save(out_labels, obj, annotator, out)
        print("  saved.")


def _save(labels, obj, annotator, out="annotation/annotations"):
    d = os.path.join(out, annotator)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{obj['query_id']}.json"), "w") as fh:
        json.dump({"query_id": obj["query_id"], "annotator": annotator, "labels": labels},
                  fh, indent=1)


# --- Informed-consent gate for human-subject annotation runs -----------------
# Human annotation runs must be preceded by informed consent (see
# human_study/INFORMATION_SHEET_AND_CONSENT.md). This records an auditable,
# PII-free consent record keyed only by the annotator's anonymous id. Use
# --skip-consent for non-human runs (automated pilots / re-analysis / tests).
CONSENT_NOTICE = (
    "\n" + "=" * 78 + "\n"
    "INFORMED CONSENT (read human_study/INFORMATION_SHEET_AND_CONSENT.md first)\n"
    "  * Participation is voluntary; you may stop at any time without penalty.\n"
    "  * You are labeling pre-collected, non-sensitive AI reasoning traces.\n"
    "  * Only your chosen ANONYMOUS id is stored with your labels; no personal\n"
    "    data is collected. Aggregated, anonymized labels may be released and\n"
    "    reported in a research publication.\n"
    "  * Contact / ethics approval details are in the information sheet.\n"
    + "=" * 78 + "\n")


def _consent_gate(annotator, consent_log):
    print(CONSENT_NOTICE)
    resp = input('Type "I CONSENT" to confirm informed consent (anything else exits): ').strip()
    if resp != "I CONSENT":
        print("Consent not given — exiting. No annotation data was recorded.")
        return False
    import datetime
    d = os.path.dirname(consent_log)
    if d:
        os.makedirs(d, exist_ok=True)
    rec = {"annotator": annotator, "consent": True,
           "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
           "protocol": "annotation/ANNOTATION_PROTOCOL.md",
           "information_sheet": "annotation/human_study/INFORMATION_SHEET_AND_CONSENT.md"}
    with open(consent_log, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"Consent recorded (anonymous id '{annotator}') to {consent_log}.\n")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", required=True,
                    help="the annotator's ANONYMOUS id (e.g. annotator_01) — never a real name")
    ap.add_argument("--traces", default="annotation/traces")
    ap.add_argument("--out", default="annotation/annotations")
    ap.add_argument("--consent-log", default="annotation/human_study/consent_log.jsonl",
                    help="where the informed-consent record is appended (human runs)")
    ap.add_argument("--skip-consent", action="store_true",
                    help="skip the consent gate (non-human runs: automated pilots / re-analysis / tests)")
    args = ap.parse_args(argv)

    if not args.skip_consent and not _consent_gate(args.annotator, args.consent_log):
        return 2

    files = sorted(glob.glob(os.path.join(args.traces, "*.json")))
    if not files:
        print(f"no trace files in {args.traces} (run collect_traces.py first)")
        return 1
    print(f"{len(files)} populations to annotate as '{args.annotator}'. "
          f"Progress is saved after every run.")
    for f in files:
        obj = json.load(open(f))
        prev_path = os.path.join(args.out, args.annotator, f"{obj['query_id']}.json")
        labels = json.load(open(prev_path))["labels"] if os.path.exists(prev_path) else {}
        annotate_population(obj, args.annotator, labels, args.out)
    print("\nDone. Thank you!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
