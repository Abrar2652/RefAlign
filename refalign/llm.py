"""Unified LLM interface for the Who&When LLM baselines.

Two backends, same ``.chat(system, user) -> str`` API:

  * :class:`LocalLLM`  — a local HuggingFace causal LM (e.g. Qwen2.5-7B-Instruct)
    on a chosen GPU; free and fully reproducible (temperature 0).
  * :class:`APILLM`    — OpenAI / Anthropic via their SDKs, keys read from a
    ``.env`` (a credentials file).

A tiny call-count / token tally is kept for the cost accounting the evaluation
protocol requires.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict


def load_env_keys(path: str = "/path/to/workspace/credentials.env") -> Dict[str, str]:
    """Read KEY=VALUE lines from a .env into a dict (no shell export)."""
    out: Dict[str, str] = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def add(self, pt: int, ct: int, dt: float) -> None:
        self.calls += 1
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.seconds += dt


class LocalLLM:
    """Local HF causal LM (lazy-loaded)."""

    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda:6",
                 max_new_tokens: int = 256,
                 hf_home: str = "$HF_HOME",
                 attn_implementation: str = None):
        self.model_name = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation  # e.g. "eager" for gemma-2
        os.environ.setdefault("HF_HOME", hf_home)
        self._tok = None
        self._model = None
        self.usage = Usage()

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        kw = dict(torch_dtype=torch.bfloat16, device_map=self.device)
        if self.attn_implementation:
            kw["attn_implementation"] = self.attn_implementation
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kw)
        self._model.eval()

    def chat(self, system: str, user: str) -> str:
        self._ensure()
        import torch
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        try:
            text = self._tok.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
        except Exception:   # some templates (e.g. gemma-2) reject a system role
            text = self._tok.apply_chat_template(
                [{"role": "user", "content": system + "\n\n" + user}],
                tokenize=False, add_generation_prompt=True)
        inp = self._tok(text, return_tensors="pt").to(self.device)
        t0 = time.time()
        with torch.no_grad():
            out = self._model.generate(**inp, max_new_tokens=self.max_new_tokens,
                                       do_sample=False, temperature=None, top_p=None,
                                       pad_token_id=self._tok.eos_token_id)
        dt = time.time() - t0
        gen = out[0][inp["input_ids"].shape[1]:]
        resp = self._tok.decode(gen, skip_special_tokens=True)
        self.usage.add(int(inp["input_ids"].shape[1]), int(gen.shape[0]), dt)
        return resp


class APILLM:
    """OpenAI or Anthropic chat model."""

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini",
                 env_path: str = "/path/to/workspace/credentials.env",
                 max_tokens: int = 256, temperature: float = 0.0):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.usage = Usage()
        keys = load_env_keys(env_path)
        self._client = None
        if provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=keys.get("OPENAI_API_KEY"))
        elif provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=keys.get("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"unknown provider {provider!r}")

    def chat(self, system: str, user: str) -> str:
        t0 = time.time()
        if self.provider == "openai":
            r = self._client.chat.completions.create(
                model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            txt = r.choices[0].message.content or ""
            u = r.usage
            self.usage.add(u.prompt_tokens, u.completion_tokens, time.time() - t0)
        else:
            r = self._client.messages.create(
                model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
                system=system, messages=[{"role": "user", "content": user}])
            txt = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            self.usage.add(r.usage.input_tokens, r.usage.output_tokens, time.time() - t0)
        return txt
