"""Loader for the Who&When failure-attribution benchmark.

Who&When (Zhang et al., ICML 2025; HF ``Kevin355/Who_and_When``) ships one JSON
object per failure instance. Field names vary a little across the Algorithm-
Generated and Hand-Crafted subsets, so this loader is deliberately tolerant: it
maps whatever "history"/"conversation" list it finds into :class:`Trace` steps,
and whatever annotation block it finds into an :class:`Annotation`.

If the real dataset is not present, the synthetic generator in ``synth.py`` gives
a fully runnable, ground-truthed stand-in. This loader is exercised only when a
path is provided.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from ..types import Step, Trace, Annotation, RunPopulation

# candidate keys, in priority order, for each field we need
_HISTORY_KEYS = ("history", "conversation", "messages", "log", "trace", "chat_history")
_QUERY_KEYS = ("query", "question", "task", "problem")
_AGENT_KEYS = ("name", "agent", "role", "speaker", "sender")
_CONTENT_KEYS = ("content", "message", "text", "value")
_ANN_KEYS = ("annotation", "ground_truth", "label", "gt")
_RESP_AGENT_KEYS = ("mistake_agent", "responsible_agent", "agent", "who")
_RESP_STEP_KEYS = ("mistake_step", "decisive_step", "step", "when")
_REASON_KEYS = ("mistake_reason", "reason", "explanation")


def _first(d: dict, keys) -> Optional[object]:
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return None


def _action_of(agent: str, content: str) -> str:
    """Heuristic canonical action label from a message (kept short & stable).

    Calculation/answer steps are recognised first so that numeric content is
    aligned and value-compared by the contrastive attributor.
    """
    c = (content or "").lower()
    if "answer" in c or "final" in c:
        return "answer"
    if any(k in c for k in ("calc", "compute", "add ", "subtract", "multiply",
                            "divide")) or "=" in c:
        return "calc"
    for kw in ("search", "browse", "read", "code", "run", "execute", "test",
               "verify", "check", "plan", "error", "tool"):
        if kw in c:
            return kw
    # fall back to the agent's role token
    return (agent or "act").split()[0].lower()[:8] or "act"


def _parse_instance(obj: dict, idx: int) -> Tuple[Optional[Trace], Optional[Annotation]]:
    history = _first(obj, _HISTORY_KEYS)
    if not isinstance(history, list) or not history:
        return None, None

    query = str(_first(obj, _QUERY_KEYS) or "")
    qid = str(obj.get("query_id", obj.get("id", f"who_{idx:04d}")))
    rid = str(obj.get("run_id", qid))

    steps: List[Step] = []
    for i, msg in enumerate(history):
        if isinstance(msg, str):
            agent, content = "system", msg
        elif isinstance(msg, dict):
            agent = str(_first(msg, _AGENT_KEYS) or "agent")
            content = str(_first(msg, _CONTENT_KEYS) or "")
        else:
            continue
        steps.append(Step(index=len(steps), agent=agent,
                          action=_action_of(agent, content), content=content))
    if not steps:
        return None, None

    ann_block = _first(obj, _ANN_KEYS)
    ann_src = ann_block if isinstance(ann_block, dict) else obj
    resp_agent = _first(ann_src, _RESP_AGENT_KEYS)
    resp_step = _first(ann_src, _RESP_STEP_KEYS)

    success = bool(obj.get("success", False))  # Who&When instances are failures
    trace = Trace(run_id=rid, query_id=qid, steps=steps, success=success, query=query)

    annotation: Optional[Annotation] = None
    if resp_agent is not None and resp_step is not None:
        try:
            step_int = int(resp_step)
        except (TypeError, ValueError):
            step_int = 0
        step_int = max(0, min(step_int, len(steps) - 1))
        annotation = Annotation(
            query_id=qid, run_id=rid, responsible_agent=str(resp_agent),
            decisive_step=step_int, reason=str(_first(ann_src, _REASON_KEYS) or ""),
        )
    return trace, annotation


def load_annotated_benchmark(traces_dir: str, labels_dir: str
                             ) -> Tuple[List[RunPopulation], Dict[str, Annotation]]:
    """Load the human-annotated populated benchmark produced by the annotation
    pipeline (annotation/traces/*.json + annotation/annotations/<name>/*.json).

    Returns (populations, annotations) ready for RefAlign and the evaluation
    harness. Flagged (ambiguous) labels are dropped.
    """
    import glob

    # labels: merge every *.json under labels_dir (one annotator's adjudicated set)
    labels: Dict[str, dict] = {}
    for f in glob.glob(os.path.join(labels_dir, "*.json")):
        obj = json.load(open(f))
        for rid, lab in obj.get("labels", {}).items():
            if not lab.get("flag"):
                labels[rid] = lab

    populations: List[RunPopulation] = []
    annotations: Dict[str, Annotation] = {}
    for tf in sorted(glob.glob(os.path.join(traces_dir, "*.json"))):
        obj = json.load(open(tf))
        qid = obj["query_id"]
        runs: List[Trace] = []
        for r in obj["runs"]:
            steps = [Step(index=s["index"], agent=str(s["agent"]),
                          action=_action_of(str(s["agent"]), str(s.get("content", ""))),
                          content=str(s.get("content", ""))) for s in r["steps"]]
            if not steps:
                continue
            trace = Trace(run_id=r["run_id"], query_id=qid, steps=steps,
                          success=(r.get("outcome") == "success"),
                          query=obj.get("query", ""),
                          final_answer=str(r.get("final_answer", "")))
            runs.append(trace)
            lab = labels.get(r["run_id"])
            if lab and r.get("outcome") == "failure":
                st = int(lab["decisive_step"])
                st = max(0, min(st, len(steps) - 1))
                annotations[r["run_id"]] = Annotation(
                    query_id=qid, run_id=r["run_id"],
                    responsible_agent=str(lab["responsible_agent"]), decisive_step=st,
                    reason=str(lab.get("reason", "")),
                    meta={"fault_type": lab.get("fault_type"),
                          "confidence": lab.get("confidence")})
        if runs:
            populations.append(RunPopulation(query_id=qid, runs=runs,
                                             query=obj.get("query", "")))
    return populations, annotations


def load_who_and_when_parquet(path: str, subset: Optional[int] = None
                              ) -> Tuple[List[Trace], Dict[str, Annotation]]:
    """Load the official Who&When parquet (Kevin355/Who_and_When).

    Columns: is_correct/is_corrected, question, question_ID, ground_truth,
    history (array of {content, name, role}), mistake_agent, mistake_step,
    mistake_reason. Each row is ONE failing run for a unique question — there are
    no multiple-run populations, which is why RefAlign's phylogeny cannot be
    built on raw Who&When (the single-trace LLM baselines can, though).
    """
    import pandas as pd

    df = pd.read_parquet(path)
    if subset:
        df = df.iloc[:subset]
    traces: List[Trace] = []
    annotations: Dict[str, Annotation] = {}
    correct_col = "is_correct" if "is_correct" in df.columns else "is_corrected"

    for i, row in df.reset_index(drop=True).iterrows():
        qid = str(row.get("question_ID", f"ww_{i:04d}"))
        rid = qid
        history = row["history"]
        steps: List[Step] = []
        for j, msg in enumerate(history):
            agent = str(msg.get("name") or msg.get("role") or "agent")
            content = str(msg.get("content") or "")
            steps.append(Step(index=len(steps), agent=agent,
                              action=_action_of(agent, content), content=content))
        if not steps:
            continue
        success = bool(row.get(correct_col, False))
        trace = Trace(run_id=rid, query_id=qid, steps=steps, success=success,
                      query=str(row.get("question", "")),
                      final_answer=str(row.get("ground_truth", row.get("groundtruth", ""))))
        traces.append(trace)

        try:
            step_i = int(float(row["mistake_step"]))
        except (TypeError, ValueError, KeyError):
            step_i = 0
        step_i = max(0, min(step_i, len(steps) - 1))
        annotations[rid] = Annotation(
            query_id=qid, run_id=rid, responsible_agent=str(row.get("mistake_agent", "")),
            decisive_step=step_i, reason=str(row.get("mistake_reason", "")))
    return traces, annotations


def load_who_and_when(path: str
                      ) -> Tuple[List[RunPopulation], Dict[str, Annotation]]:
    """Load Who&When JSON instances from ``path`` (a file or a directory of files).

    Returns (populations, annotations). Each instance becomes its own single-run
    population (Who&When gives one failing log per instance); callers that want a
    multi-run population per query should group by ``query_id`` afterwards.
    """
    files: List[str] = []
    if os.path.isdir(path):
        for root, _dirs, names in os.walk(path):
            files.extend(os.path.join(root, n) for n in names if n.endswith(".json"))
    elif os.path.isfile(path):
        files = [path]
    else:
        raise FileNotFoundError(path)

    populations: List[RunPopulation] = []
    annotations: Dict[str, Annotation] = {}
    by_query: Dict[str, List[Trace]] = {}

    for fi, fpath in enumerate(sorted(files)):
        with open(fpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        instances = data if isinstance(data, list) else [data]
        for ii, obj in enumerate(instances):
            if not isinstance(obj, dict):
                continue
            trace, ann = _parse_instance(obj, fi * 1000 + ii)
            if trace is None:
                continue
            by_query.setdefault(trace.query_id, []).append(trace)
            if ann is not None:
                annotations[ann.run_id] = ann

    for qid, traces in by_query.items():
        populations.append(RunPopulation(query_id=qid, runs=traces,
                                         query=traces[0].query))
    return populations, annotations
