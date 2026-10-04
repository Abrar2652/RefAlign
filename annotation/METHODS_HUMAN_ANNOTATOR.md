# Paper-ready Methods + Ethics — human annotation of decisive errors

Drop-in text for the paper. All numbers are computed from the released label files
and reproduce exactly (deterministic adjudication; traces pinned to `gpt-4o-mini`).
Regenerate with the commands in §Repro.

> **Provenance (per the authors).** The labels in `annotation/annotations_human/`
> were produced by **five human graduate-student volunteers**. (An earlier
> agent-generated placeholder panel was discarded; it is not part of this study.)

---

## Methods — external validity on real traces (human annotation)

**Goal.** Test whether RefAlign's step localization transfers from the controlled
benchmark to *real, naturally-failing* multi-agent traces, judged by human
annotators under the decisive-error protocol.

**Data.** A Planner→Worker→Verifier system (`gpt-4o-mini`, temperature 1.0) was run
on arithmetic word problems; we kept only populations that fail *naturally* (no
injected faults) and retain a successful counterfactual reference: **4 populations,
32 failing runs, 24 successful reference runs**. Of 10 candidate tasks, 6 were
dropped because the model either never fails (nothing to label) or never succeeds
(no counterfactual) — natural-failure collection only yields annotatable
populations in a difficulty sweet spot.

**Annotators.** **Five graduate-student volunteers** (from biomedical engineering,
computer science, and quantitative computational biology) annotated all 32 failing
runs **independently and blind** — no annotator saw another's labels or any key —
each under an anonymous id (`annotator_01`…`annotator_05`). Participation was
**voluntary and uncompensated**, with verbal informed consent (see Ethics).

**Task & instructions.** Following our annotation protocol (supplementary
`ANNOTATION_PROTOCOL.md`), for each failing run an annotator is shown a successful
reference run and marks the **decisive error** — the earliest step whose correction
would flip the run to success (the counterfactual definition of Zhang et al.,
2025) — recording the responsible agent, decisive step index, a fault type from an
8-category taxonomy, a confidence, and a one-line rationale. A schema-validating
tool makes malformed labels impossible; annotators were told to prefer the earliest
cause, treat propagated errors as consequences, and skip rather than guess.

**Agreement.** Over the 32 runs (one label was skipped as ambiguous): **responsible
agent Fleiss κ = 1.00** (unanimous 32/32); **decisive step Fleiss κ = 0.92** (exact
0.94, unanimous 30/32, **within ±1 = 1.00**); **fault type Fleiss κ ≈ 0.0**
(near chance; unanimous 0/32). We report this honestly: agent and step attribution
are highly reliable, but the **fault-type taxonomy is not reliable across
annotators** — a genuine boundary ambiguity among `missing_step` /
`premature_stop` / `wrong_operation` for "answered a partial or unevaluated
expression." Fault type is **not used** in any reported result.

**Adjudication.** Per-run majority vote across the five annotators (step tie →
earliest) yields **32 gold decisive-error labels**. Step disagreement is confined
to two loop runs (stall-origin vs first-repeat convention).

**Result.** Scored against the human gold (95% bootstrap CIs):

| method | n | agent acc | step acc | step ±1 |
|---|---|---|---|---|
| **RefAlign (no LLM)** | 32 | 0.531 [0.375, 0.719] | **0.469** [0.312, 0.656] | **0.906** [0.812, 1.000] |
| most-active-agent | 32 | 1.000 [1.000, 1.000] | 0.062 [0.000, 0.156] | 0.906 [0.812, 1.000] |
| last-step | 32 | 0.000 | 0.000 | 0.938 [0.844, 1.000] |
| random | 32 | 0.375 [0.219, 0.532] | 0.250 [0.125, 0.406] | 0.500 [0.343, 0.688] |

**Honest reading (two real limitations, stated).** (1) *Agent accuracy is
non-discriminative on this set*: the Worker is the culprit in all 32 runs, so the
trivial `most-active-agent` baseline scores 1.00, and RefAlign's 0.531 (it
mis-blames the Verifier on the 11 `real_t07` runs, whose successful siblings use a
structurally different decomposition — the known solution-diversity limitation
manifesting on real traces) sits below that ceiling. A discriminative agent metric
needs multi-culprit populations, which single-Worker arithmetic traces do not
provide. (2) *Where RefAlign helps is step localization*: **0.469 exact / 0.906
within ±1** vs random's 0.250 / 0.500 — on runs where the counterfactual aligns, it
recovers the human decisive step, or lands within one step 91% of the time. This is
the same step-localization strength the controlled benchmark shows, now
transferring to human judgments on real traces.

---

## Ethics Statement

*(Paste-in text — factual. See the author note below before submitting.)*

Annotation was performed by five graduate-student volunteers who gave **verbal
informed consent** and participated **voluntarily and without compensation**; they
were free to withdraw at any time. The task is **minimal risk** — reading short,
non-sensitive machine-generated arithmetic reasoning traces — and collected **no
personal data** about the annotators; the released labels contain only anonymous
ids (`annotator_01`…`annotator_05`), so participants are not identifiable from any
published output. The trace data being annotated is machine-generated and contains
no personal information. We release the anonymized labels, adjudicated gold, and
annotation tooling for reproduction.

> **⚠ Author note — do not skip before submitting.** This was an informal volunteer
> study **without prior IRB review**. Annotation of non-personal, machine-generated
> data by consenting volunteer colleagues is commonly treated as *not human-subjects
> research* — but this is institution-dependent, and reporting inter-annotator
> agreement is data about the annotators' behavior. **Before paper, obtain a
> formal "exempt" or "not-human-subjects-research" determination from your IRB/ethics
> office** (usually fast for minimal-risk annotation) and, if granted, add its
> reference here. **Do not state an IRB number you do not have.** Confirm each
> annotator consented specifically to the labels being *published*.

---

## the venue reproducibility checklist — annotation items

- **Human subjects?** Yes — 5 volunteer graduate-student annotators; verbal informed
  consent; voluntary; uncompensated; minimal-risk; anonymized. IRB determination:
  *to be obtained (see author note).*
- **Annotation protocol described?** Yes — `ANNOTATION_PROTOCOL.md`.
- **Number of annotators / independence?** 5, independent and blind.
- **Inter-annotator agreement reported?** Yes — agent κ = 1.00, step κ = 0.92
  (±1 = 1.00), fault κ ≈ 0.0 (reported honestly; fault type not used).
- **Adjudication method?** Per-run majority vote → 32 gold labels.
- **Anonymized before release?** Yes — anonymous ids only; id↔identity mapping (if
  any) kept off-repo.
- **Labels/tooling released?** Yes — `annotations_human/`, `traces_scale/`, tools.

## Repro

```bash
python -m annotation.validate_annotations --traces annotation/traces_scale \
    --a annotation/annotations_human/annotator_01 --b annotation/annotations_human/annotator_02
python -m annotation.run_annotated --traces annotation/traces_scale \
    --labels annotation/annotations_human/adjudicated
```

See [`SCALED_BENCHMARK.md`](SCALED_BENCHMARK.md) for the full per-task breakdown and
observed failure modes.
