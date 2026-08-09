"""OpenAI chat completions adapter."""

from __future__ import annotations

import os
import time

from tradingbench.agent.clients.base import Completion


class OpenAIClient:
    def __init__(self, model_name: str, api_key: str | None = None, **_):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

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
            raise RuntimeError("OPENAI_API_KEY not set")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("Install openai package: pip install openai") from e

        client = OpenAI(api_key=self.api_key)
        kwargs: dict = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if response_format:
            kwargs["response_format"] = response_format

        t0 = time.perf_counter()
        resp = client.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - t0) * 1000
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return Completion(
            text=text,
            model=self.model_name,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency,
            raw={"id": getattr(resp, "id", None)},
        )
