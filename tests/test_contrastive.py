from refalign.types import Step, Trace, RunPopulation, Annotation
from refalign.contrastive import contrastive_localize
from refalign.attribute import AttributionConfig, attribute_corpus


def _run(rid, values, success):
    """A run whose Worker calc steps carry the given numeric values."""
    steps = [Step(0, "Planner", "plan", "plan the ops")]
    for i, v in enumerate(values):
        steps.append(Step(len(steps), "Worker", "calc", f"CALC: step{i} = {v}"))
    steps.append(Step(len(steps), "Worker", "answer", f"ANSWER: {values[-1]}"))
    steps.append(Step(len(steps), "Verifier", "check", "VERDICT: CORRECT"))
    return Trace(rid, "q0", steps, success, query="chain")


def test_contrastive_localizes_content_fault():
    # three healthy runs with the same values; one failing run diverges at calc #3
    healthy_vals = [10, 13, 39, 45]
    pop_runs = [_run(f"h{i}", healthy_vals, True) for i in range(3)]
    fail_vals = [10, 13, 42, 48]          # first divergence at index 2 -> trace step 3
    failing = _run("f0", fail_vals, False)
    pop = RunPopulation("q0", pop_runs + [failing], "chain")

    pred = contrastive_localize(pop, failing)
    assert pred.step == 3                  # Planner(0) + calc0(1) + calc1(2) + calc2(3)
    assert pred.agent == "Worker"
    assert pred.meta["kind"] == "value"


def test_contrastive_beats_structural_on_content_fault():
    # pure-structural (action-only) cannot see a value-only divergence
    healthy_vals = [5, 8, 24, 20]
    runs = [_run(f"h{i}", healthy_vals, True) for i in range(3)]
    failing = _run("f0", [5, 8, 30, 26], False)   # diverges at calc index 2 (step 3)
    pop = RunPopulation("q0", runs + [failing], "chain")
    anns = {"f0": Annotation("q0", "f0", "Worker", 3)}

    c = contrastive_localize(pop, failing)
    s = attribute_corpus([pop], AttributionConfig())["f0"]
    assert c.step == anns["f0"].decisive_step      # contrastive nails it
    assert s.step != anns["f0"].decisive_step       # action-only misses it


def test_contrastive_falls_back_without_successes():
    failing = _run("f0", [1, 2, 3], False)
    pop = RunPopulation("q0", [failing], "chain")   # no healthy sibling
    pred = contrastive_localize(pop, failing)
    assert pred.meta.get("fallback") == "no_sibling"


def test_contrastive_earliest_divergence_wins():
    # two divergences; the earlier one must be chosen
    runs = [_run(f"h{i}", [4, 9, 18, 20, 25], True) for i in range(3)]
    failing = _run("f0", [4, 99, 18, 20, 77], False)  # diverges at index 1 and 4
    pop = RunPopulation("q0", runs + [failing], "chain")
    pred = contrastive_localize(pop, failing)
    assert pred.step == 2                            # Planner(0)+calc0(1)+calc1(2)
