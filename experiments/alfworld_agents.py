"""Planner -> Actor agent system for ALFWorld (TextWorld) household tasks, with
typed traces, controlled replay, and an environment adapter interface.

Trace convention (one agent per step, zero-based):
  step 0        Planner / plan   : a short plan from the goal and first observation
  steps 1..K    Actor   / <verb> : "ACTION: <command> -> OBS: <observation>" where
                                  <verb> is the command's head word (go, take, put,
                                  open, close, toggle, heat, cool, clean, use,
                                  examine, look, inventory)
The environment's own goal check (info['won']) decides success. The environment
is deterministic given the action sequence, so a run can be *replayed*: the
saved prefix of actions is re-executed, an action at step k is optionally
replaced (by the environment's expert action = a known valid correction, or by a
repair model's proposal), and the policy continues with fresh sampling seeds.
"""
from __future__ import annotations

import difflib
import json
import os
import re
from typing import Dict, List, Optional, Tuple

from refalign.types import Step, Trace

PLANNER_SYS = ("You are Planner for a household robot in a text environment. Given the task and the "
               "first observation, give a one- or two-sentence plan: where to look for the object, "
               "what to do with it, and where to put it. Be brief.")
ACTOR_SYS = ("You are Actor, a household robot in a text environment. Each turn choose exactly one "
             "command from the list of admissible commands and output it as:\n  ACTION: <command>\n"
             "Output nothing else. Explore receptacles to find the object, then complete every part "
             "of the task (for example, heating, cooling or cleaning the object before placing it). "
             "Do not repeat an action whose observation showed it had no effect.")
VERBS = ("go", "take", "put", "move", "open", "close", "toggle", "heat", "cool", "clean", "use", "examine", "look", "inventory")
# Two act-only exemplars per task type from the public ReAct prompt file
# (Yao et al., 2023; prompts/alfworld_3prompts.json), the standard few-shot
# protocol on this benchmark. Loaded once; keyed by ALFWorld task type.
_EX_PATH = os.path.join(os.path.dirname(__file__), "alfworld_react_prompts.json")
_EX_PREFIX = {"pick_and_place_simple": "put", "look_at_obj_in_light": "examine", "pick_clean_then_place_in_recep": "clean",
              "pick_heat_then_place_in_recep": "heat", "pick_cool_then_place_in_recep": "cool", "pick_two_obj_and_place": "puttwo"}
_EX = json.load(open(_EX_PATH)) if os.path.exists(_EX_PATH) else {}


def exemplars(task_type: Optional[str], k: int = 2) -> str:
    pre = _EX_PREFIX.get(task_type or "")
    if not pre or not _EX:
        return ""
    exs = [_EX[f"act_{pre}_{i}"].strip() for i in range(k) if f"act_{pre}_{i}" in _EX]
    return "\n\nHere are example runs of similar tasks (commands follow '>'):\n\n" + "\n\n".join(exs) if exs else ""


OBS_CHARS = 300        # history entries in the prompt and trace content
CUR_OBS_CHARS = 1500   # the current observation shown to the Actor is not truncated in practice


def verb_of(cmd: str) -> str:
    w = (cmd or "").strip().split()
    v = w[0].lower() if w else "noop"
    return v if v in VERBS else "other"


def parse_action(raw: str, admissible: List[str]) -> Tuple[str, bool]:
    """Return (command, was_valid). Non-admissible outputs are mapped to the closest
    admissible command (difflib ratio >= 0.6) or to 'look'."""
    m = re.search(r"ACTION\s*[:=]\s*(.+)", raw, re.I)
    cand = (m.group(1) if m else raw).strip().splitlines()[0].strip().strip("`'\". ")
    cand = re.sub(r"^\d+[.)]\s*", "", cand)
    if cand in admissible:
        return cand, True
    low = {a.lower(): a for a in admissible}
    if cand.lower() in low:
        return low[cand.lower()], True
    close = difflib.get_close_matches(cand, admissible, n=1, cutoff=0.6)
    if close:
        return close[0], False
    return ("look" if "look" in admissible else admissible[0]), False


def actor_prompt(goal: str, plan: str, history: List[Tuple[str, str]], obs: str, admissible: List[str]) -> str:
    hist = "\n".join(f"  {i + 1}. {a} -> {o[:OBS_CHARS]}" for i, (a, o) in enumerate(history)) or "  (none yet)"
    return (f"Task: {goal}\nPlan: {plan}\n\nHistory of your actions and observations:\n{hist}\n\n"
            f"Current observation: {obs[:CUR_OBS_CHARS]}\n\nAdmissible commands:\n" +
            "\n".join(f"  - {a}" for a in admissible) + "\n\nYour next action:")


def run_episode(env, llm, game: str, max_steps: int = 30, prefix_actions: Optional[List[str]] = None,
                override_at: Optional[int] = None, override_action: Optional[str] = None,
                plan: Optional[str] = None, record_expert: bool = True, task_type: Optional[str] = None) -> Dict:
    """Run (or replay) one episode. ``env`` is an adapter with reset(game)->(obs, info)
    and step(cmd)->(obs, done, info); info has 'admissible_commands', 'won' and
    (optionally) 'expert_plan'. ``prefix_actions`` are re-executed verbatim (Actor
    steps 1..len(prefix)); if ``override_at`` (an Actor step index, 1-based within
    the Actor sequence) is given, that action is replaced by ``override_action``
    (or by the expert action when ``override_action`` is None) and the policy
    takes over afterwards."""
    obs0, info = env.reset(game)
    goal = env.goal
    actor_sys = ACTOR_SYS + exemplars(task_type)
    steps: List[Step] = []
    calls = []
    if plan is None:
        plan = llm.chat(PLANNER_SYS, f"Task: {goal}\nFirst observation: {obs0[:600]}")
        calls.append("plan")
    steps.append(Step(0, "Planner", "plan", plan))
    history: List[Tuple[str, str]] = []
    obs = obs0
    won = False
    invalid = 0
    actions: List[str] = []
    experts: List[Optional[str]] = []
    sources: List[str] = []
    for k in range(1, max_steps + 1):
        adm = list(info.get("admissible_commands", []))
        expert = (info.get("expert_plan") or [None])[0] if record_expert else None
        source = "policy"
        if override_at is not None and k == override_at:
            cmd = override_action if override_action is not None else expert
            if cmd is None:
                raw = llm.chat(actor_sys, actor_prompt(goal, plan, history, obs, adm)); calls.append("act")
                cmd, _ = parse_action(raw, adm)
                source = "policy(no-expert-available)"
            else:
                cmd = cmd if cmd in adm else parse_action(cmd, adm)[0]
                source = "override"
        elif prefix_actions is not None and k <= len(prefix_actions):
            cmd = prefix_actions[k - 1]
            if cmd not in adm:
                cmd = parse_action(cmd, adm)[0]
            source = "replayed-prefix"
        else:
            raw = llm.chat(actor_sys, actor_prompt(goal, plan, history, obs, adm)); calls.append("act")
            cmd, valid = parse_action(raw, adm)
            invalid += (not valid)
        obs_new, done, info = env.step(cmd)
        steps.append(Step(len(steps), "Actor", verb_of(cmd), f"ACTION: {cmd} -> OBS: {obs_new[:OBS_CHARS]}"))
        history.append((cmd, obs_new))
        actions.append(cmd); experts.append(expert); sources.append(source)
        obs = obs_new
        if info.get("won"):
            won = True
            break
        if done:
            break
    trace = Trace("", "", steps, won, query=goal, final_answer="won" if won else "failed")
    return {"trace": trace, "won": won, "actions": actions, "experts": experts, "sources": sources, "plan": plan,
            "n_actor_steps": len(actions), "invalid_actions": invalid, "n_llm_calls": len(calls),
            "hit_cap": (not won) and len(actions) >= max_steps}


def trace_to_json(tr: Trace, rid: str, extra: Dict) -> Dict:
    return {"run_id": rid, "outcome": "success" if tr.success else "failure", "final_answer": tr.final_answer,
            "steps": [{"index": s.index, "agent": s.agent, "action": s.action, "content": s.content} for s in tr.steps],
            **extra}
