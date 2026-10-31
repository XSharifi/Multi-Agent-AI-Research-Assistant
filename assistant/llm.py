"""Thin LLM wrapper so every agent talks to one interface.

Providers:
  anthropic : Claude models via the Anthropic API
  openai    : any OpenAI-compatible endpoint (OpenAI, Ollama, LM Studio, vLLM)
"""
import json
import logging
import os
import re
import time

log = logging.getLogger("assistant.llm")


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "anthropic")).lower()
        if self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
            self.model = model or os.getenv("LLM_MODEL", "claude-sonnet-5-5")
        elif self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(
                base_url=os.getenv("OPENAI_BASE_URL") or None,
                api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
            )
            self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        else:
            raise ValueError(f"Unknown LLM_PROVIDER '{self.provider}' (use 'anthropic' or 'openai')")

    def chat(self, system: str, messages: list[dict], max_tokens: int = 2000) -> str:
        """Send a conversation and return the reply text. Retries transient errors."""
        last_error = None
        for attempt in range(3):
            try:
                return self._call(system, messages, max_tokens)
            except Exception as e:  # network errors, rate limits, ...
                last_error = e
                log.warning("LLM call failed (attempt %d/3): %s", attempt + 1, e)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after 3 attempts: {last_error}")

    def _call(self, system, messages, max_tokens):
        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system, messages=messages
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.2,
            messages=[{"role": "system", "content": system}] + messages,
        )
        return resp.choices[0].message.content or ""


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply (handles ```json fences and chatter)."""
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        candidates.append(fence.group(1))
    for cand in candidates:
        start = cand.find("{")
        if start == -1:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(cand)):
            c = cand[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cand[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("No valid JSON object found in LLM reply")
