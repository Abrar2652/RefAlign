"""Re-parse stored judge outputs with the bracket-tolerant STEP parser.

gpt-4o-2024-08-06 answers ``STEP: [2] | AGENT: Worker`` (mirroring the ``[i]``
rendering of steps in the prompt); the existing strict regex rejects the
brackets, so 1105 of its matched-judge answers were scored unparseable at run
time. This pass re-parses every stored AAO / matched record of every judge
from its stored raw text with :func:`parse_judge_answer` (tolerant), keeps the
strict flag in ``strict_step_parsed`` and marks changed records with
``reparsed_bracket``. Records whose stored raw text holds no STEP token (e.g.
R1 outputs truncated inside <think>) are left as they were.

    python -m experiments.reparse_judge_outputs results/new/matched_judge_paired_*.json
"""
import json
import sys

from refalign.llm_baselines import parse_judge_answer, strict_step_parsed, UNPARSED_AGENT, UNPARSED_STEP
from experiments.run_matched_judge import build_cells


def main(argv=None):
    files = (argv or sys.argv[1:])
    cells = build_cells()
    traces, anns = {}, {}
    for cell, (pops, a) in cells.items():
        anns.update(a)
        for pop in pops:
            for f in pop.failures:
                traces[f.run_id] = f
    for path in files:
        obj = json.load(open(path))
        changed = strict_unp = 0
        for j, C in obj.items():
            for cell, M in C.items():
                for m, runs in M.items():
                    if m == "phylo":
                        continue
                    for rid, r in runs.items():
                        raw = r.get("raw", "")
                        # The run-time flag was computed by the strict regex on the FULL
                        # answer; the stored raw is truncated (300 chars), so derive the
                        # strict flag from it rather than from the truncated text.
                        if r.get("reparsed_bracket"):
                            r["strict_step_parsed"] = False
                        else:
                            r["strict_step_parsed"] = bool(r.get("step_parsed", True))
                        if r["strict_step_parsed"]:
                            continue
                        strict_unp += 1
                        if r.get("reparsed_bracket"):
                            changed += 1
                            continue
                        ag, st, sp, ap_ = parse_judge_answer(raw, traces[rid])
                        if not sp:
                            continue                     # still unparseable (no STEP token)
                        ann = anns[rid]
                        r.update({"agent_pred": ag if ap_ else UNPARSED_AGENT, "step_pred": int(st),
                                  "agent_correct": bool(ap_ and ag == ann.responsible_agent),
                                  "step_correct": bool(st == ann.decisive_step),
                                  "step_parsed": True, "agent_parsed": ap_, "reparsed_bracket": True})
                        changed += 1
        json.dump(obj, open(path, "w"), indent=0, default=float)
        print(f"{path}: strict-unparsed {strict_unp}, re-parsed from bracketed index {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
