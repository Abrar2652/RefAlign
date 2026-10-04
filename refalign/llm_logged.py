"""OpenAI-compatible chat client with per-call provenance logging.

Every call appends one JSON line to ``results/new/api_log.jsonl`` carrying the
response's *reported* model identifier (the dated snapshot when the provider
returns one), the UTC request timestamp, temperature, max_tokens, the request
seed, prompt/completion tokens, finish reason and latency. The same code path
talks to the OpenAI API and to a local vLLM server (``base_url``), so judge and
rollout results are comparable across providers.

Billing / quota refusals are treated as fatal: the call is logged with the error
and the exception is re-raised, so an exhausted account can never degrade a
result silently.

``LoggedLLM.view(seed=...)`` returns a lightweight per-rollout handle with its
own sequential seed stream and token tally, which keeps sampled rollouts
reproducible even when many rollouts run concurrently.
"""
from __future__ import annotations

import datetime
import json
import os
import threading
import time
from typing import Optional, Tuple

from .llm import Usage, load_env_keys

_FATAL_MARKERS = ("insufficient_quota", "credit_balance", "billing", "invalid_api_key",
                  "authentication")


class LoggedLLM:
    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 256,
                 log_path: str = "results/new/api_log.jsonl", tag: str = "",
                 seed: Optional[int] = None, revision: Optional[str] = None,
                 env_path: str = "/path/to/workspace/credentials.env",
                 max_retries: int = 6, timeout: float = 900.0):
        from openai import OpenAI
        if api_key is None:
            api_key = "EMPTY" if base_url else load_env_keys(env_path).get("OPENAI_API_KEY")
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.log_path = log_path
        self.tag = tag
        self.seed = seed
        self.revision = revision
        self.max_retries = max_retries
        self.usage = Usage()
        self.last_raw = ""
        self.n_errors = 0
        self._lock = threading.Lock()
        self._counter = 0
        self._merge_system = False   # set when the served chat template rejects a system role
        d = os.path.dirname(log_path)
        if d:
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    def view(self, seed: Optional[int] = None, tag: Optional[str] = None) -> "LLMView":
        return LLMView(self, seed, tag if tag is not None else self.tag)

    def _log(self, rec: dict) -> None:
        with self._lock:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")

    def chat_full(self, system: str, user: str, seed: Optional[int] = None,
                  tag: Optional[str] = None, temperature: Optional[float] = None,
                  max_tokens: Optional[int] = None) -> Tuple[str, int, int, str]:
        """Return (text, prompt_tokens, completion_tokens, response_model)."""
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if seed is None and self.seed is not None:
            with self._lock:
                seed = self.seed + self._counter
                self._counter += 1
        def _messages():
            if self._merge_system:   # e.g. gemma-2: system text is prepended to the user turn
                return [{"role": "user", "content": system + "\n\n" + user}]
            return [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kwargs = dict(model=self.model, temperature=temperature, max_tokens=max_tokens,
                      messages=_messages())
        if seed is not None:
            kwargs["seed"] = int(seed)
        base = {"ts_utc": None, "tag": tag if tag is not None else self.tag,
                "request_model": self.model, "revision": self.revision,
                "base_url": self.base_url, "temperature": temperature,
                "max_tokens": max_tokens, "seed": seed}
        last_err = None
        attempt = rl = 0
        while attempt < self.max_retries:
            t0 = time.time()
            kwargs["messages"] = _messages()     # honours a merge flag set by any thread
            base["ts_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            try:
                r = self._client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001 - we log and decide below
                last_err = e
                low = str(e).lower()
                fatal = any(m in low for m in _FATAL_MARKERS)
                is_rl = (not fatal) and ("rate limit" in low or "rate_limit" in low or "429" in low)
                msg = f"{type(e).__name__}: {str(e)[:300]}"
                self._log({**base, "error": msg, "attempt": attempt, "fatal": fatal, "rate_limited": is_rl})
                with self._lock:
                    self.n_errors += 1
                if fatal:
                    raise
                if "system role" in low and not self._merge_system:
                    self._merge_system = True          # retry at once with the merged prompt
                    continue
                if is_rl:                              # provider throttling: wait, do not burn attempts
                    rl += 1
                    if rl > 60:
                        raise
                    time.sleep(min(30.0, 1.5 ** min(rl, 9)))
                    continue
                attempt += 1
                time.sleep(min(60.0, 2.0 ** attempt))
                continue
            dt = time.time() - t0
            txt = r.choices[0].message.content or ""
            u = getattr(r, "usage", None)
            pt = int(getattr(u, "prompt_tokens", 0) or 0)
            ct = int(getattr(u, "completion_tokens", 0) or 0)
            rec = {**base, "response_model": getattr(r, "model", None),
                   "system_fingerprint": getattr(r, "system_fingerprint", None),
                   "finish_reason": getattr(r.choices[0], "finish_reason", None),
                   "prompt_tokens": pt, "completion_tokens": ct,
                   "latency_s": round(dt, 3), "response_chars": len(txt),
                   "system_merged_into_user": self._merge_system}
            self._log(rec)
            with self._lock:
                self.usage.add(pt, ct, dt)
                self.last_raw = txt
            return txt, pt, ct, str(getattr(r, "model", self.model))
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")

    def chat(self, system: str, user: str) -> str:
        return self.chat_full(system, user)[0]


class LLMView:
    """Per-rollout handle: sequential seed stream + own token tally; shares the
    parent's client, log and global tally."""

    def __init__(self, parent: LoggedLLM, seed: Optional[int], tag: str):
        self.parent = parent
        self.base_seed = seed
        self.tag = tag
        self.k = 0
        self.usage = Usage()
        self.last_raw = ""
        self.last_tokens = (0, 0)
        self.response_models = set()
        self.seeds_used = []

    @property
    def model(self) -> str:
        return self.parent.model

    def chat(self, system: str, user: str) -> str:
        seed = None if self.base_seed is None else int(self.base_seed + self.k)
        self.k += 1
        t0 = time.time()
        txt, pt, ct, rm = self.parent.chat_full(system, user, seed=seed, tag=self.tag)
        self.usage.add(pt, ct, time.time() - t0)
        self.last_raw = txt
        self.last_tokens = (pt, ct)
        self.response_models.add(rm)
        if seed is not None:
            self.seeds_used.append(seed)
        return txt
