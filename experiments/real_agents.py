"""A small *real* multi-agent system with typed actions, run many times per task
to produce genuine LLM-trace populations for RefAlign — plus AEGIS-style
controlled fault injection for ground-truth decisive-error labels.

System: Planner -> Worker (a ReAct loop over typed tools SEARCH/CALC/ANSWER) ->
Verifier. Each turn is a real API call; each becomes a typed Step so the encoder
sees clean (agent, action) symbols (the OTel-style tool-span abstraction the
research grounding recommends). Success = final answer matches ground truth.

Fault injection (AEGIS): for a failing run we corrupt one worker turn at a known
step (force a wrong CALC or a premature ANSWER), which propagates to a wrong
final answer — giving a verifiable (agent, step) decisive error.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from refalign.types import Step, Trace, Annotation, RunPopulation

# small arithmetic word-problem tasks with checkable numeric answers
TASKS: List[Tuple[str, float]] = [
    ("A shop sells pens at $3 each. Ann buys 7 pens and pays with a $50 bill. How much change does she get?", 29),
    ("A train travels 60 km in 45 minutes. What is its speed in km/h?", 80),
    ("A rectangle is 8 m long and 5 m wide. What is its perimeter in meters?", 26),
    ("Tom has 3 boxes with 12 apples each and eats 5. How many apples remain?", 31),
    ("A 20% discount is applied to a $75 jacket. What is the final price in dollars?", 60),
    ("A car uses 6 liters per 100 km. How many liters for a 250 km trip?", 15),
    ("If 5 workers build a wall in 8 days, how many worker-days is that?", 40),
    ("A recipe needs 250 g flour per cake. How many kg for 6 cakes?", 1.5),
    ("A tank holds 1200 L and drains at 40 L/min. How many minutes to empty?", 30),
    ("Sara saves $15 per week. How much has she saved after 12 weeks?", 180),
    ("A book has 320 pages; Jo reads 40 pages a day. How many days to finish?", 8),
    ("An item costs $12; buying 25 of them, what is the total cost in dollars?", 300),
]

_SAFE = re.compile(r"^[0-9+\-*/(). %]+$")


def _calc(expr: str) -> Optional[float]:
    expr = expr.strip().replace("%", "/100")
    if not expr or not _SAFE.match(expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


PLANNER_SYS = "You are Planner. Give a one-sentence plan to solve the arithmetic task. Be brief."
WORKER_SYS = (
    "You are Worker, solving an arithmetic task with tools. Each turn output exactly one action:\n"
    "  SEARCH: <fact you need>\n  CALC: <arithmetic expression, digits and + - * / ( ) only>\n"
    "  ANSWER: <final number>\n"
    "Use CALC for every arithmetic step. When confident, output ANSWER. One action per turn.")
VERIFIER_SYS = ("You are Verifier. Given the worker's answer, reply exactly:\n"
                "VERDICT: CORRECT or VERDICT: WRONG\nFINAL: <number>")


def _num(s: str) -> Optional[float]:
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return float(m.group()) if m else None


def _last_num(s: str) -> Optional[float]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return float(nums[-1]) if nums else None


def run_once(llm, task: str, answer: float, temperature: float = 0.7,
             max_worker_turns: int = 5, inject: Optional[Tuple[int, str]] = None
             ) -> Tuple[Trace, bool, Optional[int], Optional[str]]:
    """Run the system once. ``inject=(turn, kind)`` corrupts worker turn ``turn``
    (kind in {'premature','wrong_tool','bad_calc'}). Returns (trace, success, fault_step, agent)."""
    steps: List[Step] = []
    fault_step = fault_agent = None

    plan = _chat(llm, PLANNER_SYS, f"Task: {task}", temperature)
    steps.append(Step(len(steps), "Planner", "plan", plan))

    history = f"Task: {task}\nPlan: {plan}\n"
    final_answer = None
    for wt in range(max_worker_turns):
        raw = _chat(llm, WORKER_SYS, history + "\nYour next action:", temperature)
        act, arg = _parse_action(raw)
        # AEGIS injection at a known worker turn
        if inject is not None and wt == inject[0]:
            fault_step = len(steps)
            fault_agent = "Worker"
            if inject[1] == "premature":          # structural: answers too early
                act, arg = "ANSWER", str(answer + 7)
            elif inject[1] == "wrong_tool":       # structural: searches where a calc is due
                act, arg = "SEARCH", "unrelated background info"
            else:  # bad_calc                     # content: right action, wrong arithmetic
                act, arg = "CALC", f"{int(answer)}+9"
        if act == "CALC":
            val = _calc(arg)
            history += f"\nWorker CALC {arg} = {val}"
            steps.append(Step(len(steps), "Worker", "calc", f"CALC {arg} = {val}"))
        elif act == "SEARCH":
            history += f"\nWorker SEARCH {arg}"
            steps.append(Step(len(steps), "Worker", "search", f"SEARCH {arg}"))
        else:  # ANSWER
            final_answer = _last_num(arg)
            steps.append(Step(len(steps), "Worker", "answer", f"ANSWER {arg}"))
            break

    verdict = _chat(llm, VERIFIER_SYS,
                    f"Task: {task}\nWorker's answer: {final_answer}", temperature)
    steps.append(Step(len(steps), "Verifier", "check", verdict))
    # success is judged on the Worker's committed answer (the Verifier turn is
    # kept for trace realism but its self-reported FINAL is not trusted).
    result = final_answer
    success = result is not None and abs(result - answer) < 1e-6

    trace = Trace(run_id="", query_id="", steps=steps, success=success,
                  query=task, final_answer=str(result))
    return trace, success, fault_step, fault_agent


def _chat(llm, sys, user, temperature):
    return llm.chat(sys, user)


def _parse_action(raw: str) -> Tuple[str, str]:
    for key in ("ANSWER", "CALC", "SEARCH"):
        m = re.search(rf"{key}\s*[:=]\s*(.+)", raw, re.I)
        if m:
            return key, m.group(1).splitlines()[0].strip()
    return "ANSWER", raw.strip()[:40]


def build_population(llm, task_idx: int, n_runs: int = 8, n_fault: int = 4,
                     temperature: float = 0.8
                     ) -> Tuple[RunPopulation, Dict[str, Annotation]]:
    """Build one real-trace population: clean runs + AEGIS-injected failing runs."""
    task, answer = TASKS[task_idx]
    qid = f"task{task_idx:02d}"
    runs: List[Trace] = []
    anns: Dict[str, Annotation] = {}
    n_clean = n_runs - n_fault

    for r in range(n_clean):
        tr, success, _, _ = run_once(llm, task, answer, temperature)
        tr.run_id = f"{qid}_c{r:02d}"
        tr.query_id = qid
        runs.append(tr)

    kinds = ["premature", "wrong_tool", "bad_calc"]   # 2 structural + 1 content
    for r in range(n_fault):
        turn = 1 + (r % 2)          # inject at worker turn 1 or 2
        kind = kinds[r % len(kinds)]
        tr, success, fstep, fagent = run_once(llm, task, answer, temperature,
                                              inject=(turn, kind))
        rid = f"{qid}_f{r:02d}"
        tr.run_id = rid
        tr.query_id = qid
        tr.success = False           # injection is designed to fail
        runs.append(tr)
        if fstep is not None:
            anns[rid] = Annotation(qid, rid, fagent, fstep,
                                   reason=f"AEGIS {kind}", meta={"fault_type": kind})
    return RunPopulation(qid, runs, task), anns
