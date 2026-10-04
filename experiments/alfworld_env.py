"""Single-game ALFWorld (TextWorld) environment adapter with the hand-coded expert.

Run under the ALFWorld venv (/path/to/workspace/alf_venv) with
ALFWORLD_DATA pointing at the downloaded data. The adapter exposes
    env.reset(game_file) -> (observation, info)
    env.step(command)    -> (observation, done, info)
with info = {'admissible_commands', 'won', 'expert_plan' (list, next expert
action first), 'gamefile'} and env.goal = the task sentence. The simulator is
deterministic given the command sequence (no domain randomization), which the
replay experiments rely on.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Tuple

import numpy as np

ALFWORLD_DATA = os.environ.setdefault("ALFWORLD_DATA", "/path/to/workspace/alfworld_data")
TASK_TYPES = ["pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
              "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"]
SPLITS = {"valid_unseen": "json_2.1.1/valid_unseen", "valid_seen": "json_2.1.1/valid_seen", "train": "json_2.1.1/train"}


def list_games(split: str) -> List[Dict]:
    root = os.path.join(ALFWORLD_DATA, SPLITS[split])
    out = []
    for gf in sorted(glob.glob(os.path.join(root, "*", "*", "game.tw-pddl"))):
        task_dir = os.path.basename(os.path.dirname(os.path.dirname(gf)))
        ttype = next((t for t in TASK_TYPES if task_dir.startswith(t)), "unknown")
        out.append({"game_file": gf, "game_id": os.path.relpath(gf, root).replace("/game.tw-pddl", ""),
                    "task_type": ttype, "task_dir": task_dir})
    return out


def sample_games(split: str, n: int, seed: int, stratify: bool = True) -> List[Dict]:
    games = list_games(split)
    if n <= 0 or n >= len(games):
        return games
    rng = np.random.RandomState(seed)
    if not stratify:
        idx = sorted(rng.choice(len(games), n, replace=False).tolist())
        return [games[i] for i in idx]
    by = {}
    for g in games:
        by.setdefault(g["task_type"], []).append(g)
    quota = {t: int(round(n * len(v) / len(games))) for t, v in by.items()}
    while sum(quota.values()) < n:
        quota[max(by, key=lambda t: len(by[t]))] += 1
    while sum(quota.values()) > n:
        quota[max(quota, key=quota.get)] -= 1
    out = []
    for t, v in sorted(by.items()):
        k = min(quota[t], len(v))
        out += [v[i] for i in sorted(rng.choice(len(v), k, replace=False).tolist())]
    return out


class AlfEnv:
    """One TextWorld game at a time, with the hand-coded expert attached."""

    def __init__(self, expert: bool = True):
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos, AlfredExpert, AlfredExpertType
        self._tw = textworld
        self._wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
        extras = ["gamefile"]
        if expert:
            self._wrappers.append(AlfredExpert(expert_type=AlfredExpertType.HANDCODED))
            extras.append("expert_plan")
        self._infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=extras)
        self._env = None
        self.goal = ""
        self.game_file = None

    def reset(self, game_file: str) -> Tuple[str, Dict]:
        # Reuse one underlying environment per process: TextWorld's PDDL backend copies
        # the Fast Downward shared library into a new temporary directory on every
        # Env() construction and frees it only at interpreter exit, so creating an
        # environment per episode leaks disk space. load() switches the game instead.
        if self._env is None:
            self._env = self._tw.start(game_file, request_infos=self._infos, wrappers=self._wrappers)
        else:
            self._env.load(game_file)
        state = self._env.reset()
        self.game_file = game_file
        obs = state.feedback
        m = re.search(r"Your task is to:\s*(.+)", obs)
        self.goal = m.group(1).strip() if m else obs[-200:]
        return obs, self._info(state)

    def step(self, command: str) -> Tuple[str, bool, Dict]:
        state, reward, done = self._env.step(command)
        return state.feedback, bool(done), self._info(state)

    def _info(self, state) -> Dict:
        return {"admissible_commands": list(state.get("admissible_commands", [])),
                "won": bool(state.get("won", False)),
                "expert_plan": list(state.get("extra.expert_plan", []) or []),
                "gamefile": state.get("extra.gamefile")}

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
