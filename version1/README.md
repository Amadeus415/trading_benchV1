# Trading Bench — version1 (MVP)

A **measurement instrument** for sequential decision-making under uncertainty. Each agent gets **$1,000 fake money**, trades a fixed universe of equities + crypto on a weekly cadence, and is scored on return *and* behaviour — with honest error bars.

This folder is the v1 implementation of [VISION.md](VISION.md) against [MVP_SPEC.md](MVP_SPEC.md).

## What you get

| Layer | Status |
|---|---|
| Frozen snapshot + point-in-time store (no lookahead) | ✅ |
| Deterministic simulator (fills, fees, ledger, corporate actions) | ✅ |
| Baselines: buy-and-hold, 60/40, random, momentum-lite | ✅ |
| Agent observation → prompt → parse → orders loop | ✅ |
| Thin LLM clients (xAI / OpenAI / Anthropic) + mock | ✅ |
| Episode artifacts + static HTML reports | ✅ |
| Scorecard + bootstrap CIs + paired MDE | ✅ |
| Contamination stubs (blind mode, probe, leak scan) | ✅ partial |
| Prediction markets / live tools / interactive dashboard | ❌ deferred (see spec) |

## Quick start

```bash
cd version1
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. Build a reproducible synthetic market snapshot
tradingbench build-snapshot --out snapshots

# 2. Run one episode (offline, no API key)
SNAPSHOT=$(ls -d snapshots/*/ | head -1)
tradingbench run-episode \
  --snapshot "$SNAPSHOT" \
  --model buy_and_hold \
  --start 2025-01-06 \
  --steps 12

# 3. Open the report
open runs/*/report.html   # or xdg-open

# 4. Smoke campaign (several baselines × windows)
#    Edit configs/campaign_smoke.yaml snapshot_id if needed — "auto" resolves the first snapshot.
tradingbench run-campaign --config configs/campaign_smoke.yaml --max-episodes 6
```

### Wire a real model

```bash
export OPENAI_API_KEY=...   # or XAI_API_KEY / ANTHROPIC_API_KEY

tradingbench run-episode \
  --snapshot "$SNAPSHOT" \
  --provider openai \
  --model my-gpt \
  --model-name gpt-4o-mini \
  --start 2025-01-06
```

Or uncomment models in `configs/campaign_v1.yaml`.

## Layout

```
version1/
  MVP_SPEC.md              # full specification (copied)
  VISION.md                # project vision (copied)
  configs/                 # campaign YAML
  snapshots/               # immutable frozen data
  runs/                    # per-episode artifacts + HTML
  tradingbench/
    data/                  # snapshot builder + PointInTimeStore
    sim/                   # ledger, validate, execution, engine
    agent/                 # observation, prompt, parser, baselines, clients
    eval/                  # metrics, aggregate, contamination
    report/                # static HTML
    runner.py              # episode + campaign orchestration
    cli.py
  tests/
```

**Dependency rule** (enforced by convention; import-linter recommended for CI):

```
data  →  sim  →  eval  →  report
   ↘   agent   ↗
```

`agent/` reaches market data **only** through `agent/observation.py`.

## Design choices that matter

1. **Unadjusted prices** — splits/dividends applied by the simulator on ex-date (avoids adj_close lookahead).
2. **Separate PIT stores** — agent sees `as_of=t`; fills use a different store with `as_of=t+1`.
3. **Notional orders** — models specify dollars, not shares.
4. **Empty orders allowed** — holding is not penalised; no churn incentive.
5. **One repair pass** then `MALFORMED` + zero orders — malformed rate is a scorecard metric.
6. **No CLI harnesses as runtime** — thin raw API adapters only (Comparable principle).

## Tests

```bash
pytest -q
```

Acceptance anchors:

- Golden ledger (3 trades + split + dividend) to the cent
- PIT guard over random `as_of` dates
- Buy-and-hold episode completes; replay is deterministic

## Campaign scaling

| | Smoke | Spec target |
|---|---|---|
| Windows | 1–4 | 8 |
| Seeds | 1–3 | 5 |
| Modes | standard | standard + named_control + blind + synthetic |
| Episodes / model | ~12 | ≥40 |

Report MDE on the leaderboard. If model spread &lt; MDE, say *indistinguishable* — do not invent a ranking.

## Fake money only

All trading is simulated. Models never receive authority over real capital or a brokerage account.
