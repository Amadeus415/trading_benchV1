"""Mock client that routes to baselines — zero API cost for MVP demos."""

from __future__ import annotations

import json
import time

from tradingbench.agent.baselines import get_baseline, seed_rng
from tradingbench.agent.clients.base import Completion


class MockClient:
    """Uses a baseline policy. Observation is re-parsed from the user prompt JSON."""

    def __init__(self, model_name: str = "buy_and_hold", baseline: str | None = None, **_):
        # model_name can be baseline id
        self.model_name = model_name
        self.baseline_name = baseline or model_name
        if self.baseline_name not in (
            "buy_and_hold", "sixty_forty", "random_agent", "momentum_lite"
        ):
            self.baseline_name = "momentum_lite"
        self._fn = get_baseline(self.baseline_name)
        self._step_counter = 0
        self._seed = 1

    def set_episode_context(self, seed: int, step: int) -> None:
        self._seed = seed
        self._step_counter = step

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
        t0 = time.perf_counter()
        obs = _extract_obs(user)
        rng = seed_rng(seed if seed is not None else self._seed, self._step_counter)
        if self.baseline_name == "random_agent":
            decision = self._fn(obs, rng)
        else:
            decision = self._fn(obs, rng)
        text = json.dumps(decision)
        latency = (time.perf_counter() - t0) * 1000
        # rough token estimate
        in_tok = max(1, len(system + user) // 4)
        out_tok = max(1, len(text) // 4)
        return Completion(
            text=text,
            model=f"mock:{self.baseline_name}",
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency,
        )


def _extract_obs(user: str) -> dict:
    marker = "Full observation JSON:"
    if marker in user:
        blob = user.split(marker, 1)[1].strip()
        # up to next blank-line section end — find JSON object
        start = blob.find("{")
        if start < 0:
            return _empty_obs()
        # brace match
        depth = 0
        for i, ch in enumerate(blob[start:]):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(blob[start : start + i + 1])
                    except json.JSONDecodeError:
                        break
    return _empty_obs()


def _empty_obs() -> dict:
    return {
        "as_of": "2025-01-06",
        "episode": {"step": 0, "total_steps": 12, "cadence": "weekly"},
        "portfolio": {"nav": 1000.0, "cash": 1000.0, "positions": []},
        "market": [],
        "news": [],
        "prior_decisions": [],
        "last_step_violations": [],
        "rules": {"max_position_weight": 0.25, "min_order_usd": 10.0},
    }
