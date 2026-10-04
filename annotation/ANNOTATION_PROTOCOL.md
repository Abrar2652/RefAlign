# RefAlign — Human Annotation Protocol

This document is a **complete, step-by-step recipe** for building a human-labeled,
*populated* failure-attribution benchmark for RefAlign. Follow it top to bottom.
Every command is copy-pasteable; every field an annotator enters is validated by
the tool, so a correctly-followed run **cannot produce a malformed label**.

> **What you are building.** For each *task*, several runs of a multi-agent system
> (some succeed, some fail). Human annotators label each *failing* run with its
> **decisive error**: the responsible agent and the earliest step whose correction
> would have made the run succeed. This is the counterfactual definition used by
> Who&When (Zhang et al., ICML 2025). Unlike Who&When, we keep the *successful*
> runs too — they are the counterfactual references RefAlign needs.

---

## 0. Roles, prerequisites, layout

**Roles**
- **Organizer** (1 person): runs Phase 0 and Phases 2–3, resolves disagreements.
- **Annotators** (≥2 people): independently label the failing runs in Phase 1.
  Two independent annotators are required so agreement can be measured.

**Prerequisites**
- The repository, with the Python environment working: `pip install -r requirements.txt`.
- For Phase 0 only, an OpenAI API key (used to *generate* the traces). Put it in a
  `.env` the tool can read, or point `--env` at one. Annotators in Phase 1 need **no**
  API key — they read pre-collected JSON files.
- Run all commands from the repository root.

**Directory layout** (created for you)
```
annotation/
  collect_traces.py        # Phase 0: produce populations to annotate
  annotate.py              # Phase 1: interactive, validated labeling tool
  validate_annotations.py  # Phase 2: schema checks + inter-annotator agreement
  run_annotated.py         # Phase 3: evaluate RefAlign on the human labels
  traces/                  # <- collected populations land here (one file per task)
  annotations/<name>/      # <- each annotator's labels land here
  gold/                    # optional: a small calibration set with known answers
```

---

## 1. Definitions (read before annotating)

- **Run / trace.** One execution of the multi-agent system on one task: an ordered
  list of **steps**. Each step has an **index** (0,1,2,…), an **agent** (who acted),
  and **content** (what they said/did).
- **Population.** All runs collected for the *same task*. Contains ≥1 **successful**
  run and ≥1 **failing** run.
- **Success / failure.** A run is *successful* if it reaches the correct final
  answer, *failing* otherwise. Outcome is pre-computed against the task's ground
  truth and stored in the trace; you do **not** re-judge it.
- **Decisive error (what you label).** For a failing run, the **earliest** step
  such that, *if that single action were corrected and everything downstream
  proceeded normally, the run would have succeeded*. You record:
  - **responsible agent** — the agent that took that step (or, for an omission,
    the agent that *should* have acted);
  - **decisive step** — that step's index;
  - **fault type** — one label from the taxonomy in §4;
  - **confidence** — 1 (low) to 5 (high);
  - **reason** — one line of natural language.

### The counterfactual test (apply it literally)
Walk the failing run from step 0. At each step ask:

> "If I replaced *this* step with a correct action and let the run continue, would
> it now succeed?"

The **first** step for which the answer is "yes" is the decisive step. Equivalent
shortcut: it is usually the **earliest step that departs from what a successful run
of the same task does** in a way that dooms the outcome. The successful run shown
next to it is your reference for "what correct looks like."

**Rules that resolve most cases:**
1. **Earliest, not most-obvious.** If several steps are wrong, label the *earliest*
   one whose correction fixes the run. Later errors are usually *consequences*.
2. **Propagation.** If a wrong value/decision flows into later steps, those later
   steps are *not* the decisive error — the origin is.
3. **Omission.** If the run *skips* a necessary step, the decisive step is the
   position where the missing step should have occurred (label the step index that
   now sits at that position), and the agent is the one who *should* have acted.
4. **Verification failure.** If a wrong answer was produced earlier and a Verifier
   *failed to catch it*, the decisive error is normally the earlier producing step
   — unless the Verifier had enough information to catch it and its acceptance is
   what allowed the run to terminate wrongly; then label the Verifier.
5. **No identifiable decisive error.** If you cannot apply the counterfactual test
   (the failure is diffuse, or you would need information not in the trace), **skip
   the run** (enter `s` in the tool) rather than guess. Skipped runs are excluded.

---

## 2. Phase 0 — Collect the populations (Organizer)

This runs a real Planner→Worker→Verifier system many times per task at a non-zero
temperature so some runs fail *naturally* (no injected faults), and saves only
populations that have something to annotate.

```bash
# from the repo root
python -m annotation.collect_traces \
    --runs 12 --temperature 0.9 \
    --min-success 1 --min-failure 2 \
    --out annotation/traces
```

**What you should see.** One line per task, e.g.
`real_t00: 7 success / 3 failure -> SAVED`, and finally
`kept N annotatable populations in annotation/traces`.

**Verify** before distributing:
```bash
ls annotation/traces/*.json
python -c "import json,glob;[print(f, len(json.load(open(f))['runs']),'runs') for f in glob.glob('annotation/traces/*.json')]"
```
Each file must contain both `"outcome": "success"` and `"outcome": "failure"` runs.
If too few tasks are kept, raise `--runs` or `--temperature`, or add tasks to the
`TASKS` list at the top of `collect_traces.py`.

> You may substitute traces from any framework (AutoGen, LangGraph, …). Just match
> the schema: `{"query_id","query","ground_truth","runs":[{"run_id","outcome",
> "final_answer","steps":[{"index","agent","content"}]}]}`.

---

## 3. Phase 1 — Annotate (each Annotator, independently)

Each annotator runs the tool under their **own name** and never sees the other's
labels. The tool shows a successful reference run, then each failing run, and
prompts for each field, **re-asking on any invalid input**. Progress is saved
after every run, so you can stop and resume anytime.

```bash
python -m annotation.annotate --annotator annotator_01 --traces annotation/traces
# a second annotator, separately:
python -m annotation.annotate --annotator annotator_02   --traces annotation/traces
```

**For each failing run the tool asks, in order:**
1. `annotate this run? [Enter=yes / s=skip]` — press Enter to label; type `s` to
   skip an ambiguous run (see rule 5).
2. `responsible agent [...]` — type one agent name from the list shown (validated).
3. `decisive step index (0..N-1)` — type the integer step (validated to range).
4. `fault type number` — type the number of one taxonomy entry (§4; validated).
5. `confidence (1..5)` — how sure you are.
6. `one-line reason` — a short justification (free text).

**Do**
- Read the whole failing run *and* the successful reference before deciding.
- Apply the counterfactual test (§1) literally; prefer the **earliest** cause.
- Use `s` when genuinely unsure — do not guess.

**Do not**
- Discuss specific runs with the other annotator during Phase 1.
- Change the trace files.

Output lands in `annotation/annotations_human/annotator_01/<task>.json` (one file per task).

---

## 4. Fault-type taxonomy

Choose the single best-fitting type (the number the tool shows):

| # | fault type | meaning |
|---|---|---|
| 0 | `wrong_computation` | a step produces an incorrect value/result |
| 1 | `wrong_operation` | uses the wrong operation, tool, or approach |
| 2 | `missing_step` | skips a step the task required |
| 3 | `repeated_step` | repeats/loops without making progress |
| 4 | `misplan` | the plan itself is wrong or incomplete |
| 5 | `premature_stop` | stops or answers before finishing |
| 6 | `bad_verification` | verifier accepts a wrong answer / rejects a right one |
| 7 | `other` | none of the above (explain in the reason) |

---

## 5. Worked examples

**Example A — a loop that never finishes (repetition).** Task: "240 apples, sold
35% Monday and 40 more Tuesday; how many remain?" (answer 116). Failing run:
`[0] Planner: plan …` `[1] Worker: 0.35*240 = 84` `[2] Worker: 240-84 = 156` then
steps 3–5 repeat `240-84` and it never subtracts 40. The **earliest** step that
dooms the run is **step 2**: the Worker gets stuck and never proceeds to subtract
Tuesday's 40. → agent `Worker`, step `2`, type `repeated_step`, reason "loops on
240-84, never subtracts the 40 for Tuesday."

**Example B — an early wrong value (computation).** If instead `[2] Worker:
240-84 = 166` (wrong) and everything after uses 166, the decisive step is **2**
(`wrong_computation`), not the later steps that merely inherit 166.

**Example C — omission.** If the run computes Monday and Tuesday correctly but the
final step forgets to subtract Tuesday's 40 (a step is missing), label the position
where that subtraction should have been, agent = whoever was doing the arithmetic,
type `missing_step`.

---

## 6. Phase 2 — Validate & measure agreement (Organizer)

Run schema/consistency checks and compute inter-annotator agreement:

```bash
python -m annotation.validate_annotations \
    --traces annotation/traces \
    --a annotation/annotations_human/annotator_01 \
    --b annotation/annotations_human/annotator_02
```

**What you should see.** `[annotator_01] K valid, F flagged, 0 errors`, the same for annotator_02,
then an agreement block:
```
=== inter-annotator agreement (n=… shared runs) ===
  agent: Cohen's kappa = 0.xx
  step : exact = 0.xx, within +-1 = 0.xx, kappa = 0.xx
```

**Targets & actions.**
- **0 errors** is mandatory. If any error prints (agent not in roster, step out of
  range, non-failing run labeled, …), fix that label and re-run.
- **Agent Cohen's κ ≥ 0.6** is the acceptance bar (Who&When reports κ≈0.7–0.88 for
  expert annotators). If lower, hold a calibration meeting, refine the guidance in
  §1/§5, and re-annotate a batch.

**Adjudication.** For runs where the two annotators disagree, the Organizer creates
the final label in a new folder `annotation/annotations/adjudicated/` (copy the
agreed labels; for disagreements, pick the label that best satisfies the
counterfactual test, or mark `flag:"ambiguous"` to drop it). Agreed runs can be
copied verbatim.

---

## 7. Phase 3 — Export & evaluate (Organizer)

Score RefAlign (and baselines) against the **human** labels:

```bash
python -m annotation.run_annotated \
    --traces annotation/traces \
    --labels annotation/annotations/adjudicated
```

You will see a leaderboard with agent / step / step±1 accuracy and 95% bootstrap
CIs, computed on the human-labeled runs. This is the external-validity result: how
well RefAlign recovers *human* decisive-error judgments on real, naturally-failing
multi-agent traces. Report it alongside the controlled-benchmark results.

The loader `refalign.data.schema.load_annotated_benchmark(traces_dir, labels_dir)`
returns `(populations, annotations)` if you want to run other analyses.

---

## 8. Quality-control checklist (per batch)

- [ ] Every trace file has ≥1 success and ≥2 failures.
- [ ] Two annotators labeled independently, under distinct `--annotator` names.
- [ ] `validate_annotations.py` prints **0 errors** for both.
- [ ] Agent Cohen's κ ≥ 0.6 (else recalibrate and re-annotate).
- [ ] Disagreements adjudicated into `annotation/annotations/adjudicated/`.
- [ ] `run_annotated.py` runs and the leaderboard `n` equals the adjudicated count.
- [ ] A small **gold** set with obvious answers is interleaved to catch inattentive
      annotation (optional but recommended).

---

## 9. Effort, scale, and tips

- **Scale.** For a first release, aim for ~30–50 tasks × ~10 runs, yielding
  ~60–150 labeled failing runs (comparable to Who&When's Hand-Crafted split). Add
  harder tasks (multi-hop, tool-use) for a stronger benchmark.
- **Time.** A short arithmetic failing run takes ~30–60 s to annotate; a long
  reasoning trace 2–5 min. Budget accordingly per annotator.
- **Reproducibility.** Keep the exact `collect_traces.py` (model, temperature,
  seed of task list) and the model version pinned; record them in the release.
- **Blind labeling.** Never show annotators the ground-truth final answer while
  they localize the *step* — outcome is given, the decisive step is theirs to find.
- **When in doubt, skip.** A smaller, clean, high-agreement set beats a large noisy
  one.
