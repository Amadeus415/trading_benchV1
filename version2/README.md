# Trading Bench — version2 (MVP)

A **measurement instrument** for sequential decision-making under uncertainty. Each agent gets **$1,000 fake money**, trades a fixed universe of equities + crypto on a weekly cadence, and is scored on **selection alpha** first, total return second — with honest error bars.

This folder implements [MVP_SPEC.md](MVP_SPEC.md) (v1.0) against [VISION.md](VISION.md). It is the v2 codebase; `version1/` remains the earlier scaffold.

## What you get

| Layer | Status |
|---|---|
| Frozen snapshot + point-in-time store (no lookahead) | ✅ |
| Deterministic simulator (fills, fees, ledger, corporate actions) | ✅ |
| Full baseline suite (§8.2): BH, EW rebal, mom_3m, MR, MA, 60/40, random | ✅ |
| Agent observation → prompt → parse → orders loop | ✅ |
| Thin LLM clients (xAI / OpenAI / Anthropic) + mock | ✅ |
| 2×2 mask factorial + alias-map middleware + memory_only | ✅ |
| Selection-alpha attribution (Barra-style CS WLS) | ✅ |
| Bootstrap CIs, paired MDE, Wilcoxon signed-rank | ✅ |
| Attacker probe certificate (Wilson CIs) | ✅ |
| Forward track scaffolding (Phase 0) | ✅ |
| Episode artifacts + static HTML reports | ✅ |
| Prediction markets / live tools / interactive dashboard | ❌ deferred |

## Design that matters

1. **Unadjusted prices** — splits/dividends applied by the simulator on ex-date.
2. **Separate PIT stores** — agent sees `as_of=t`; fills use a different store with `as_of=t+1`.
3. **Notional orders** — models specify dollars, not shares.
4. **Locked harness** — thin raw API adapters only; no Grok/Codex CLI as runtime.
5. **Selection alpha primary** — total return is secondary (KTD-Fin finding).
6. **2×2 masks** — `bright` / `stock_blind` / `date_blind` / `blinded`; news withheld on all but bright.
7. **Alias map at the boundary** — ready for v1.1 tools without retrofit leaks.
8. **Honest power** — MDE on the leaderboard; indistinguishable when spread < MDE.

## Quick start

```bash
cd version2
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

# 3. Smoke campaign (full baseline suite × bright+blinded)
tradingbench run-campaign --config configs/campaign_smoke.yaml

# 4. Open comparison report
open runs/_campaign_smoke/compare_report.html

# 5. Start the forward track (Phase 0 — start day one)
tradingbench forward-init --out forward/live_v1 --models buy_and_hold,momentum_3m
tradingbench forward-step --portfolio forward/live_v1 --snapshot "$SNAPSHOT"
```

### Mask / mode CLI

```bash
# Fully blinded, memory-only null channel
tradingbench run-episode \
  --snapshot "$SNAPSHOT" \
  --model momentum_3m \
  --mask blinded \
  --decision-mode memory_only \
  --start 2025-01-06

# Synthetic price path control
tradingbench run-episode \
  --snapshot "$SNAPSHOT" \
  --model buy_and_hold \
  --price-path synthetic \
  --seed 3
```

### Wire a real model

```bash
export OPENAI_API_KEY=...   # or XAI_API_KEY / ANTHROPIC_API_KEY

tradingbench run-episode \
  --snapshot "$SNAPSHOT" \
  --provider openai \
  --model my-gpt \
  --model-name gpt-4o-mini \
  --mask bright \
  --start 2025-01-06
```

## Layout

```
version2/
  MVP_SPEC.md
  VISION.md
  configs/                 # campaign_smoke.yaml, campaign_v2.yaml
  snapshots/               # immutable frozen data
  runs/                    # per-episode artifacts + HTML
  forward/                 # live track (created by forward-init)
  tradingbench/
    data/                  # snapshot, PIT store, corporate_actions
    sim/                   # ledger, validate, execution, engine
    agent/                 # observation, prompt, parser, baselines, clients
    eval/                  # metrics, aggregate, attribution, contamination/
    report/                # static HTML
    forward.py             # Phase 0 live paper portfolio
    runner.py
    cli.py
  tests/
```

**Dependency rule:**

```
data  →  sim  →  eval  →  report
   ↘   agent   ↗
```

`agent/` reaches market data **only** through `agent/observation.py`.

## Baseline suite

| Baseline | Definition |
|---|---|
| `buy_and_hold` | Equal weight at t=0, never rebalanced |
| `equal_weight_rebal` | Equal weight, rebalanced each step |
| `momentum_3m` | Top half by trailing 3m return — **the bar that matters** |
| `mean_reversion_3m` | Bottom half by trailing 3m return |
| `ma_crossover` | Hold when 1m and 3m returns both positive, else cash |
| `random_agent` | Seeded random trades within rules |
| `sixty_forty` | 60% equity / 40% cash |

If your best model cannot beat `momentum_3m` out-of-sample, **say so plainly**. That is a publishable result.

## Contamination grid

| Mask | Ticker | Date | Isolates |
|---|---|---|---|
| `bright` | real | real | Upper bound on contamination |
| `stock_blind` | `ASSET_####` | real | Calendar priors only |
| `date_blind` | real | `day_+N` | Ticker priors only |
| `blinded` | alias | relative | Contamination floor |

Plus `decision_mode: memory_only` (null-channel) and `price_path: synthetic` (block bootstrap).

## Tests

```bash
pytest -q
```

Acceptance anchors:

- Golden ledger (3 trades + split + dividend) to the cent
- PIT guard over random `as_of` dates
- Alias map stability + factorial behaviour
- Full baseline suite produces legal decisions
- Attribution Common + Style + Selection = portfolio
- Wilcoxon + bootstrap aggregate structure
- Buy-and-hold episode completes; replay is deterministic

## Campaign scaling

| | Smoke | Spec target |
|---|---|---|
| Windows | 1 | 8 |
| Seeds | 1 | 5 |
| Masks | bright + blinded | full 2×2 |
| Episodes / model / cell | few | ≥40 |

**Anchor-deep / others-shallow:** full mask × mode grid on one anchor model; other models get `blinded × standard` only.

## Fake money only

All trading is simulated. Models never receive authority over real capital or a brokerage account.
