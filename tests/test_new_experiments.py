"""Tests for the the paper follow-up experiments: matched-information judge
prompt construction, answer parsing (incl. unparseable -> incorrect and counted),
the paired-outcome file schema, the statistics helpers, and reference forcing."""
import json
from unittest import mock

import numpy as np
import pytest

from refalign.types import Step, Trace, Annotation, RunPopulation
from refalign import contrastive as C
from refalign.contrastive import contrastive_localize
from refalign.llm_baselines import (matched_judge, matched_judge_prompt, parse_judge_answer,
                                      UNPARSED_AGENT, UNPARSED_STEP, _MATCHED_SYS)
from refalign import stats_new as S
from refalign import evaluate as ev


def _run(rid, values, success):
    steps = [Step(0, "Planner", "plan", "plan the ops")]
    for i, v in enumerate(values):
        steps.append(Step(len(steps), "Worker", "calc", f"CALC: step{i} = {v}"))
    steps.append(Step(len(steps), "Worker", "answer", f"ANSWER: {values[-1]}"))
    steps.append(Step(len(steps), "Verifier", "check", "VERDICT: CORRECT"))
    return Trace(rid, "q0", steps, success, query="chain of ops")


class FakeJudge:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def chat(self, system, user):
        self.calls.append((system, user))
        return self.reply


@pytest.fixture
def pop():
    healthy = [_run(f"h{i}", [10, 13, 39, 45], True) for i in range(3)]
    failing = _run("f0", [10, 13, 42, 48], False)      # diverges at trace step 3
    return RunPopulation("q0", healthy + [failing], "chain of ops"), failing, healthy


# --- prompt construction -----------------------------------------------------
def test_matched_prompt_contains_both_runs_and_exact_answer_format(pop):
    p, failing, healthy = pop
    prompt = matched_judge_prompt(failing, healthy[0])
    assert "SUCCESSFUL run:" in prompt and "FAILED run:" in prompt
    assert prompt.index("SUCCESSFUL run:") < prompt.index("FAILED run:")
    assert "Task: chain of ops" in prompt
    for s in failing.steps:                       # every failing step rendered, zero-based index
        assert f"[{s.index}] {s.agent}:" in prompt
    assert "[0] Planner:" in prompt
    assert "STEP: <failing step index> | AGENT: <agent name>" in prompt
    assert "zero-based" in _MATCHED_SYS
    assert "CALC: step2 = 42" in prompt and "CALC: step2 = 39" in prompt   # both runs' contents


def test_matched_judge_sends_system_and_user(pop):
    p, failing, healthy = pop
    j = FakeJudge("STEP: 3 | AGENT: Worker")
    matched_judge(failing, healthy[0], j)
    assert len(j.calls) == 1
    assert j.calls[0][0] == _MATCHED_SYS
    assert "FAILED run:" in j.calls[0][1]


# --- answer parsing ----------------------------------------------------------
def test_parse_well_formed_both_orders(pop):
    p, failing, _ = pop
    assert parse_judge_answer("STEP: 3 | AGENT: Worker", failing) == ("Worker", 3, True, True)
    assert parse_judge_answer("AGENT: Worker | STEP: 3", failing) == ("Worker", 3, True, True)
    assert parse_judge_answer("The answer is\nSTEP=3 | AGENT=Worker.", failing)[:2] == ("Worker", 3)


def test_matched_judge_unparseable_is_scored_incorrect_and_counted(pop):
    p, failing, healthy = pop
    ann = Annotation("q0", "f0", "Worker", 3)
    preds = [matched_judge(failing, healthy[0], FakeJudge(txt))
             for txt in ["I cannot tell.", "", "Step three by the worker", "STEP: three"]]
    for pr in preds:
        assert pr.step == UNPARSED_STEP and pr.agent == UNPARSED_AGENT
        assert pr.meta["step_parsed"] is False
        assert (pr.agent == ann.responsible_agent) is False       # scored incorrect ...
        assert (pr.step == ann.decisive_step) is False
    assert sum(1 for pr in preds if not pr.meta["step_parsed"]) == 4   # ... and countable


def test_matched_judge_step_only_marks_agent_unparsed(pop):
    p, failing, healthy = pop
    pr = matched_judge(failing, healthy[0], FakeJudge("STEP: 3"))
    assert pr.step == 3 and pr.agent == UNPARSED_AGENT
    assert pr.meta["step_parsed"] and not pr.meta["agent_parsed"]


def test_out_of_range_step_clamped_like_existing_parser(pop):
    p, failing, healthy = pop
    pr = matched_judge(failing, healthy[0], FakeJudge("STEP: 99 | AGENT: Worker"))
    assert pr.step == len(failing) - 1 and pr.agent == "Worker"


def test_unknown_agent_falls_back_to_roster_like_existing_parser(pop):
    p, failing, healthy = pop
    pr = matched_judge(failing, healthy[0], FakeJudge("STEP: 3 | AGENT: Wrkr"))
    assert pr.step == 3 and pr.agent == "Worker"          # fuzzy roster match (existing behaviour)


# --- paired-outcome schema ----------------------------------------------------
def _rec(agent, step, ann, tokens=10):
    return {"agent_pred": agent, "step_pred": step, "agent_correct": agent == ann.responsible_agent,
            "step_correct": step == ann.decisive_step, "tokens": tokens}


def test_paired_schema_roundtrip(tmp_path):
    ann = Annotation("q0", "f0", "Worker", 3)
    obj = {"judge": {"L5": {"phylo": {"f0": _rec("Worker", 3, ann, 0)},
                            "matched-phylo": {"f0": _rec("Worker", 2, ann)},
                            "aao": {"f0": _rec(UNPARSED_AGENT, UNPARSED_STEP, ann)}}}}
    assert S.validate_paired(obj) == 3
    f = tmp_path / "paired.json"
    f.write_text(json.dumps(obj))
    back = json.loads(f.read_text())
    assert S.validate_paired(back) == 3
    assert back["judge"]["L5"]["aao"]["f0"]["step_correct"] is False


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("tokens"),
    lambda r: r.__setitem__("step_pred", 2.0),
    lambda r: r.__setitem__("step_correct", 1),
    lambda r: r.__setitem__("tokens", -1),
])
def test_paired_schema_rejects_bad_records(mutate):
    ann = Annotation("q0", "f0", "Worker", 3)
    rec = _rec("Worker", 3, ann)
    mutate(rec)
    with pytest.raises(ValueError):
        S.validate_paired({"judge": {"L5": {"phylo": {"f0": rec}}}})


def test_paired_schema_rejects_empty():
    with pytest.raises(ValueError):
        S.validate_paired({})
    with pytest.raises(ValueError):
        S.validate_paired({"judge": {"L5": {"phylo": {}}}})


def test_run_judge_produces_valid_paired_file_and_summary():
    """End-to-end on a tiny cell with a fake LLM (no network)."""
    import experiments.run_matched_judge as M
    from experiments.chain_agents import build_population

    class FakeView:
        def __init__(self, reply):
            self.reply, self.last_raw, self.last_tokens = reply, "", (7, 3)

        def chat(self, system, user):
            self.last_raw = self.reply
            return self.reply

    class FakeLLM:
        def __init__(self):
            from refalign.llm import Usage
            self.usage = Usage()

        def view(self, tag=""):
            # answer step 1 for every run: sometimes right, mostly wrong; one unparseable
            return FakeView("garbage" if tag.endswith("_f00") else "STEP: 1 | AGENT: Worker")

    p, a = build_population(None, 0, 5, seed=0, noise=True)
    cells = {"L5": ([p], a)}
    paired = {}
    M.run_judge("fake", FakeLLM(), cells, seeds=[0, 1], workers=2, paired=paired)
    assert S.validate_paired(paired) > 0
    methods = paired["fake"]["L5"]
    assert set(methods) == {"phylo", "aao", "matched-phylo", "matched-random-s0", "matched-random-s1"}
    assert all(len(v) == len(p.failures) for v in methods.values())
    assert all(r["tokens"] == 0 for r in methods["phylo"].values())
    assert all(r["tokens"] == 10 for r in methods["aao"].values())
    unp = [r for r in methods["matched-phylo"].values() if r["step_parsed"] is False]
    assert len(unp) == 1 and unp[0]["step_pred"] == UNPARSED_STEP and unp[0]["step_correct"] is False
    summary, tests = M.summarize(paired["fake"], "fake")
    assert summary["L5"]["matched-phylo"]["n_unparsed_step"] == 1
    assert summary["L5"]["phylo"]["step"]["acc"] == 1.0
    assert "matched-random" in summary["L5"] and len(summary["L5"]["matched-random"]["seeds"]) == 2
    t = tests["L5"]["phylo_vs_matched-phylo"]["step"]
    assert t["n"] == len(p.failures) and 0.0 <= t["p_value"] <= 1.0 and "p_holm" in t


# --- statistics helpers -------------------------------------------------------
def test_mcnemar_exact_matches_existing_helper_and_scipy():
    rng = np.random.RandomState(0)
    x = rng.rand(40) < 0.7
    y = rng.rand(40) < 0.4
    m = S.mcnemar_exact(x, y)
    assert abs(m["p_value"] - ev.mcnemar(x.astype(int), y.astype(int))["p_value"]) < 1e-12
    assert m["b"] == int(np.sum(x & ~y)) and m["c"] == int(np.sum(~x & y))
    assert S.mcnemar_exact([1, 1, 1], [1, 1, 1])["p_value"] == 1.0


def test_wilson_interval_properties():
    lo, hi = S.wilson(24, 24)
    assert lo > 0.85 and hi == 1.0
    lo, hi = S.wilson(0, 24)
    assert lo == 0.0 and hi < 0.15
    r = S.acc_record([True, False, True, True])
    assert r["c"] == 3 and r["n"] == 4 and r["wilson_lo"] < r["acc"] < r["wilson_hi"]


def test_fisher_and_diff_ci():
    f = S.fisher_2x2(10, 14, 2, 14)
    assert 0.0 <= f["p_value"] <= 1.0 and f["table"] == [[10, 4], [2, 12]]
    d = S.diff_ci(10, 14, 2, 14)
    assert d["ci_lo"] < d["diff"] < d["ci_hi"] and abs(d["diff"] - (2 / 14 - 10 / 14)) < 1e-12


# --- reference forcing (Experiment 2 mechanism) --------------------------------
def test_forced_reference_is_used_by_contrastive_localize(pop):
    p, failing, healthy = pop
    forced = healthy[2]
    with mock.patch.object(C, "_nearest_sibling", lambda failing, successes, alpha, seqs, cfg=None: forced):
        pred = contrastive_localize(p, failing)
    assert pred.meta["sibling"] == forced.run_id and pred.step == 3


def test_phylo_reference_matches_selected_sibling(pop):
    import experiments.run_matched_judge as M
    p, failing, _ = pop
    ref, pred, how = M.phylo_reference(p, failing)
    assert ref.run_id == pred.meta["sibling"] and how == "meta.sibling"


def test_bracketed_step_index_is_parsed_and_strict_flag_kept(pop):
    from refalign.llm_baselines import strict_step_parsed
    p, failing, healthy = pop
    pr = matched_judge(failing, healthy[0], FakeJudge("STEP: [3] | AGENT: Worker"))
    assert pr.step == 3 and pr.agent == "Worker" and pr.meta["step_parsed"] is True
    assert pr.meta["strict_step_parsed"] is False and strict_step_parsed("STEP: 3") is True
    assert parse_judge_answer("STEP: [99] | AGENT: Worker", failing)[1] == len(failing) - 1
