"""Thin model client adapters. One method: complete(...)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)


class ModelClient(Protocol):
    def complete(
        self,
        system: str,
        user: str,
        *,
        seed: int | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: dict | None = None,
    ) -> Completion: ...


def get_client(provider: str, model_name: str, **kwargs) -> ModelClient:
    provider = provider.lower()
    if provider in ("baseline", "mock", "local"):
        from .mock import MockClient
        return MockClient(model_name=model_name, **kwargs)
    if provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(model_name=model_name, **kwargs)
    if provider in ("xai", "grok"):
        from .xai import XAIClient
        return XAIClient(model_name=model_name, **kwargs)
    if provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model_name=model_name, **kwargs)
    raise ValueError(f"Unknown provider: {provider}")
