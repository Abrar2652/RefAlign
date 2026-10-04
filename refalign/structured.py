"""RefAlign-S: the same reference selection and backbone alignment as
:func:`refalign.contrastive.contrastive_localize`, with a *structured*
comparator on tool arguments and tool outcomes instead of the numeric
value extractor. This is a task-specific comparator for traces whose steps are
"ACTION: <verb> <object> [from|in/on|with <receptacle>] -> OBS: <text>" (the
ALFWorld Planner->Actor system). It is reported as a separate variant, developed
on the ALFWorld development split only, and the original numeric implementation
is left untouched as the baseline.

Comparator: two aligned backbone steps diverge when their (verb, object type,
receptacle type) differ after stripping instance numbers, or when exactly one of
them had no effect ("Nothing happens."). Reference selection ('content' mode)
adds a unit penalty per aligned backbone step whose structured key differs.
Abstention: when the walk finds no divergence the method returns the last-step
fallback *and* marks the prediction ``abstain=True`` so coverage and selective
accuracy can be reported separately.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .contrastive import ContrastiveConfig, _encode_actions, _required_actions, _fallback
from .encode import Alphabet, GAP
from .msa import needleman_wunsch
from .types import Prediction, RunPopulation, Trace

_ACT = re.compile(r"ACTION:\s*([A-Za-z]+)\s*(.*?)(?:\s*->\s*OBS:\s*(.*))?$", re.S)
_INST = re.compile(r"\s+\d+\b")


@dataclass
class StructuredConfig:
    sibling: str = "content"       # 'content' (verb+args) | 'edit' (action only)
    thresh: float = 0.9            # backbone count-consistency threshold (unchanged from RefAlign)
    use_status: bool = True        # treat "Nothing happens." vs effect as a divergence
    strip_instances: bool = True   # "shelf 2" -> "shelf"


def parse_fields(content: str, strip_instances: bool = True) -> Optional[Dict[str, str]]:
    m = _ACT.match((content or "").strip())
    if not m:
        return None
    verb, rest, obs = m.group(1).lower(), (m.group(2) or "").strip(), (m.group(3) or "").strip()
    norm = (lambda s: _INST.sub("", s.strip()).strip()) if strip_instances else (lambda s: s.strip())
    obj = recep = ""
    if verb == "go":
        recep = norm(rest[3:] if rest.startswith("to ") else rest)
    else:
        mm = re.match(r"(.+?)\s+(from|in/on|in|on|with)\s+(.+)$", rest)
        if mm:
            obj, recep = norm(mm.group(1)), norm(mm.group(3))
        else:
            obj = norm(rest)
    status = "noeffect" if obs.lower().startswith("nothing happens") else "effect"
    return {"verb": verb, "obj": obj, "recep": recep, "status": status}


def skey(step, cfg: StructuredConfig) -> str:
    f = parse_fields(step.content, cfg.strip_instances)
    if f is None:
        return step.action
    k = f"{f['verb']}|{f['obj']}|{f['recep']}"
    return k + (f"|{f['status']}" if cfg.use_status else "")


def _dist(failing: Trace, s: Trace, alpha, seqs, cfg: StructuredConfig) -> float:
    cost, af, asb = needleman_wunsch(seqs[failing.run_id], seqs[s.run_id], alpha)
    if cfg.sibling != "content":
        return cost
    fp = sp = 0
    for x, y in zip(af.tolist(), asb.tolist()):
        if x != GAP and y != GAP:
            if skey(failing.steps[fp], cfg) != skey(s.steps[sp], cfg):
                cost += 1.0
            fp += 1; sp += 1
        elif x != GAP:
            fp += 1
        else:
            sp += 1
    return cost


def nearest_reference(failing: Trace, pop: RunPopulation, cfg: StructuredConfig = StructuredConfig()) -> Trace:
    alpha, seqs = _encode_actions(pop)
    best, best_cost = None, None
    for s in pop.successes:
        d = _dist(failing, s, alpha, seqs, cfg)
        if best_cost is None or d < best_cost:
            best, best_cost = s, d
    return best


def _backbone(tr: Trace, val_alpha: Alphabet, required: set, cfg: StructuredConfig):
    idx, ids, steps = [], [], []
    for s in tr.steps:
        if s.action not in required:
            continue
        idx.append(s.index); steps.append(s)
        ids.append(val_alpha.encode(s.action, skey(s, cfg)))
    return idx, np.asarray(ids, dtype=np.int64), steps


def candidates(failing: Trace, ref: Trace, required: set, cfg: StructuredConfig) -> List[Tuple[int, str, str, Optional[str]]]:
    val_alpha = Alphabet(mode="action")
    fidx, fids, fsteps = _backbone(failing, val_alpha, required, cfg)
    sidx, sids, ssteps = _backbone(ref, val_alpha, required, cfg)
    if len(fids) == 0 or len(sids) == 0:
        return []
    _, af, asb = needleman_wunsch(fids, sids, val_alpha)
    out = []
    fp = sp = 0
    for x, y in zip(af.tolist(), asb.tolist()):
        if x != GAP and y != GAP:
            fs, ss = fsteps[fp], ssteps[sp]
            if fs.action != ss.action:
                out.append((fidx[fp], "action", f"did '{fs.action}' where the reference did '{ss.action}'", None))
            elif skey(fs, cfg) != skey(ss, cfg):
                out.append((fidx[fp], "argument", f"{skey(fs, cfg)} vs reference {skey(ss, cfg)}", None))
            fp += 1; sp += 1
        elif x != GAP:
            out.append((fidx[fp], "insertion", f"extra '{fsteps[fp].action}' step", None)); fp += 1
        else:
            here = fidx[fp] if fp < len(fidx) else (len(failing) - 1)
            out.append((here, "deletion", f"skipped required '{ssteps[sp].action}'", ssteps[sp].agent)); sp += 1
    return out


def structured_localize(pop: RunPopulation, failing: Trace, cfg: StructuredConfig = StructuredConfig(),
                        reference: Optional[Trace] = None) -> Prediction:
    if not pop.successes:
        p = _fallback(failing); p.meta["abstain"] = True; p.meta["method"] = "refalign_s"; return p
    ref = reference or nearest_reference(failing, pop, cfg)
    required = _required_actions(pop, cfg.thresh)
    cands = candidates(failing, ref, required, cfg)
    if not cands:
        p = _fallback(failing)
        p.meta.update({"abstain": True, "method": "refalign_s", "sibling": ref.run_id, "required": sorted(required)})
        return p
    step, kind, note, resp = cands[0]
    step = min(step, len(failing) - 1)
    agent = resp or failing.agent_at(step)
    return Prediction(failing.run_id, agent, step, 1.0,
                      meta={"method": "refalign_s", "kind": kind, "sibling": ref.run_id, "note": note,
                            "abstain": False, "required": sorted(required), "n_candidates": len(cands)})
