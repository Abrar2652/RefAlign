"""Tests for the annotation pipeline: the benchmark loader and the validators
produce consistent, usable objects (so a followed protocol yields a runnable
benchmark)."""
import json
import os

from refalign.data.schema import load_annotated_benchmark
from refalign.contrastive import contrastive_localize
from refalign import evaluate as ev
from annotation.validate_annotations import cohen_kappa


def _write_bench(tmpdir):
    traces = os.path.join(tmpdir, "traces")
    labels = os.path.join(tmpdir, "labels")
    os.makedirs(traces); os.makedirs(labels)
    trace = {
        "query_id": "q0", "query": "add then subtract", "ground_truth": "20",
        "runs": [
            {"run_id": "q0_c0", "outcome": "success", "final_answer": "20",
             "steps": [{"index": 0, "agent": "Planner", "content": "plan"},
                       {"index": 1, "agent": "Worker", "content": "CALC 10+15=25"},
                       {"index": 2, "agent": "Worker", "content": "CALC 25-5=20"},
                       {"index": 3, "agent": "Verifier", "content": "VERDICT: CORRECT"}]},
            {"run_id": "q0_f0", "outcome": "failure", "final_answer": "28",
             "steps": [{"index": 0, "agent": "Planner", "content": "plan"},
                       {"index": 1, "agent": "Worker", "content": "CALC 10+15=25"},
                       {"index": 2, "agent": "Worker", "content": "CALC 25-5=28"},
                       {"index": 3, "agent": "Verifier", "content": "VERDICT: WRONG"}]},
        ],
    }
    json.dump(trace, open(os.path.join(traces, "q0.json"), "w"))
    lab = {"query_id": "q0", "annotator": "gold",
           "labels": {"q0_f0": {"responsible_agent": "Worker", "decisive_step": 2,
                                "fault_type": "wrong_computation", "confidence": 5,
                                "reason": "25-5 should be 20", "flag": None}}}
    json.dump(lab, open(os.path.join(labels, "q0.json"), "w"))
    return traces, labels


def test_loader_builds_populations_and_annotations(tmp_path):
    traces, labels = _write_bench(str(tmp_path))
    pops, anns = load_annotated_benchmark(traces, labels)
    assert len(pops) == 1
    pop = pops[0]
    assert len(pop.successes) == 1 and len(pop.failures) == 1
    assert "q0_f0" in anns
    assert anns["q0_f0"].responsible_agent == "Worker"
    assert anns["q0_f0"].decisive_step == 2


def test_contrastive_runs_on_annotated_benchmark(tmp_path):
    traces, labels = _write_bench(str(tmp_path))
    pops, anns = load_annotated_benchmark(traces, labels)
    preds = {f.run_id: contrastive_localize(pops[0], f) for f in pops[0].failures}
    sc = ev.score(preds, anns)
    # the wrong-computation at step 2 is exactly a value divergence from the sibling
    assert preds["q0_f0"].step == 2
    assert sc.summary()["agent_acc"] == 1.0


def test_cohen_kappa_perfect_and_chance():
    a = {"x": "Worker", "y": "Planner", "z": "Worker"}
    assert cohen_kappa(a, a)[0] == 1.0            # perfect agreement
    b = {"x": "Planner", "y": "Worker", "z": "Planner"}
    k, n = cohen_kappa(a, b)
    assert n == 3 and k <= 0.0                     # total disagreement
