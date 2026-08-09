"""Anthropic Messages API adapter."""

from __future__ import annotations

import os
import time

from tradingbench.agent.clients.base import Completion


class AnthropicClient:
    def __init__(self, model_name: str, api_key: str | None = None, **_):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def complete(
        self,
        system: str,
        user: str,
        *,
        seed: int | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: dict | None = None,
    ) -> Completion:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("Install anthropic package: pip install anthropic") from e

        client = anthropic.Anthropic(api_key=self.api_key)
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency = (time.perf_counter() - t0) * 1000
        text = ""
        for block in resp.content:
            if hasattr(block, "text"):
                text += block.text
        return Completion(
            text=text,
            model=self.model_name,
            input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
            latency_ms=latency,
            raw={"id": getattr(resp, "id", None)},
        )
