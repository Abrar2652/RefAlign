"""Tests for the revision code: structured comparator, positional
baselines (leave-one-family-out), ALFWorld action parsing and trace typing, the
t03 rescoring, and the replay job builder. No environment or network needed."""
import json
from collections import Counter

import numpy as np
import pytest

from refalign.types import Step, Trace, RunPopulation, Annotation
from refalign.structured import parse_fields, skey, structured_localize, nearest_reference, StructuredConfig
from refalign.contrastive import contrastive_localize


def _run(rid, cmds, success):
    steps = [Step(0, "Planner", "plan", "find the mug and put it on the desk")]
    for c in cmds:
        verb = c.split()[0]
        steps.append(Step(len(steps), "Actor", verb, f"ACTION: {c} -> OBS: You {verb} it."))
    return Trace(rid, "g", steps, success, query="put a mug on desk")


def test_parse_fields_strips_instances_and_reads_status():
    f = parse_fields("ACTION: take mug 2 from shelf 3 -> OBS: You pick up the mug 2.")
    assert f == {"verb": "take", "obj": "mug", "recep": "shelf", "status": "effect"}
    f = parse_fields("ACTION: go to drawer 4 -> OBS: Nothing happens.")
    assert f["verb"] == "go" and f["recep"] == "drawer" and f["status"] == "noeffect"
    assert parse_fields("plan text") is None
    assert skey(Step(1, "Actor", "take", "ACTION: take mug 1 from shelf 1 -> OBS: ok"), StructuredConfig()) == "take|mug|shelf|effect"


def test_structured_localize_finds_skipped_required_action_and_abstains_when_identical():
    ok = [_run("s1", ["go to shelf 1", "take mug 1 from shelf 1", "go to desk 1", "move mug 1 to desk 1"], True),
          _run("s2", ["go to shelf 2", "go to shelf 1", "take mug 1 from shelf 1", "go to desk 1", "move mug 1 to desk 1"], True)]
    bad = _run("f1", ["go to shelf 1", "take mug 1 from shelf 1", "go to desk 1"], False)      # never places
    pop = RunPopulation("g", ok + [bad], "put a mug on desk")
    p = structured_localize(pop, bad)
    assert p.meta["kind"] == "deletion" and p.meta["abstain"] is False and p.step == len(bad) - 1
    same = _run("f2", ["go to shelf 1", "take mug 1 from shelf 1", "go to desk 1", "move mug 1 to desk 1"], False)
    pop2 = RunPopulation("g", ok + [same], "put a mug on desk")
    p2 = structured_localize(pop2, same)
    assert p2.meta["abstain"] is True and p2.step == len(same) - 1
    assert nearest_reference(bad, pop).run_id in ("s1", "s2")


def test_numeric_refalign_has_no_value_signal_on_tool_traces():
    ok = [_run("s1", ["go to shelf 1", "take mug 1 from shelf 1", "go to desk 1", "move mug 1 to desk 1"], True)]
    bad = _run("f1", ["go to shelf 3", "take mug 1 from shelf 3", "go to desk 1", "move mug 1 to desk 1"], False)
    pop = RunPopulation("g", ok + [bad], "put")
    p = contrastive_localize(pop, bad)
    assert p.meta.get("fallback") == "no_sibling"          # same action sequence, no numeric action -> fallback
    ps = structured_localize(pop, bad)                      # instance numbers stripped: same keys -> abstain too
    assert ps.meta["abstain"] is True


def test_positional_baselines_lofo_fit_and_clamp():
    import experiments.rev_positional_baselines as PB
    tr = [(_run(f"r{i}", ["go to a 1", "take b 1 from a 1"], False), Annotation("g", f"r{i}", "Actor", s)) for i, s in enumerate([2, 2, 1])]
    params = PB.fit(tr)
    assert params["mode_step"] == 2 and params["mode_agent"] == "Actor" and params["mode_joint"] == ("Actor", 2)
    short = _run("x", ["go to a 1"], False)                # 2 steps: fitted step 2 is invalid -> clamped to 1
    assert PB.predict("mode_step", params, short)[1] == 1
    assert PB.predict("penultimate_step", params, short)[1] == 0
    assert 0 <= PB.predict("norm_pos", params, short)[1] <= 1


def test_alfworld_action_parsing_and_verbs():
    from experiments.alfworld_agents import parse_action, verb_of, exemplars
    adm = ["go to shelf 1", "take mug 1 from shelf 1", "look"]
    assert parse_action("ACTION: go to shelf 1", adm) == ("go to shelf 1", True)
    assert parse_action("ACTION: Go To Shelf 1", adm) == ("go to shelf 1", True)
    assert parse_action("I will take the mug from shelf 1", adm)[1] is False
    assert parse_action("xyz", adm) == ("look", False)
    assert verb_of("move mug 1 to desk 1") == "move" and verb_of("") == "other"
    assert exemplars("unknown_type") == ""


def test_rescore_t03_recomputes_outcomes():
    import experiments.rev_rescore_t03 as RS
    store = {"int_baseline_t03_r00": {"outcome": "failure", "final_answer": "76.0", "steps": []},
             "int_targeted_t03_r01": {"outcome": "failure", "final_answer": "84", "steps": []},
             "int_placebo_t00_r02": {"outcome": "success", "final_answer": "116", "steps": []}}
    tasks = [("t0", 116.0), ("t1", 1), ("t2", 1), ("t3 path", 76.0)]
    changed = RS.rescore(store, tasks)
    assert [c[0] for c in changed] == ["int_baseline_t03_r00"] and store["int_baseline_t03_r00"]["outcome"] == "success"
    assert store["int_targeted_t03_r01"]["outcome"] == "failure" and store["int_placebo_t00_r02"]["outcome"] == "success"


def test_replay_job_builder_marks_unreplayable_steps(tmp_path):
    import experiments.rev_alfworld_jobs as JB
    att = {"r1": {"refalign_s": {"step": 2}, "penultimate": {"step": 40}, "task_type": "x"}}
    store = {"r1": {"actions": ["a"] * 5}}
    ap, tp, op = tmp_path / "a.json", tmp_path / "t.json", tmp_path / "o.json"
    ap.write_text(json.dumps(att)); tp.write_text(json.dumps(store))
    JB.main(["--attribution", str(ap), "--traces", str(tp), "--out", str(op), "--profile-n", "1", "--kmax", "3", "--methods", "refalign_s", "penultimate"])
    out = json.loads(op.read_text())
    steps = {j["step"] for j in out["jobs"]}
    assert steps == {1, 2, 3} and out["not_replayable"] == [["r1", "penultimate", 40]]
    j2 = next(j for j in out["jobs"] if j["step"] == 2)
    assert set(j2["modes"]) == {"expert", "repair", "noedit"} and "profile" in j2["selected_by"]
