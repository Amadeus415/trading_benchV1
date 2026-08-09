# Trading Bench — MVP Specification (v1.0)

**Status:** draft for implementation
**Companion doc:** `VISION.md`
**Scope:** the smallest system that can produce a *trustworthy, statistically meaningful* comparison of LLM trading agents.

---

## 0. Framing

Trading Bench is a **measurement instrument** for sequential decision-making under uncertainty. Trading is the substrate, not the point.

That framing sets the engineering priorities. An instrument is only as good as its error bars, so the two design problems that dominate v1 are:

| Problem | Consequence if ignored | v1 mitigation |
|---|---|---|
| **Sampling noise** | One 12-month path is one sample. Path variance swamps model skill. A leaderboard from single runs measures luck. | Short episodes, many of them. Paired comparisons. Bootstrap CIs. |
| **Training contamination** | Models have priors about 2025–2026 prices. "Skill" may be recall. | Not eliminated — *quantified*. Blind A/B, memory probes, synthetic controls. |

Everything below serves those two. Features that do not are deferred.

---

## 1. Goals and non-goals

### v1.0 delivers

- A deterministic paper-trading simulator with auditable accounting
- A point-in-time data layer with an enforced no-lookahead guarantee
- An agent loop that turns one observation into one validated set of orders
- An episode runner producing ≥40 paths per model across multiple windows and seeds
- A contamination harness (blind A/B, memory probes, synthetic control)
- Baselines: buy-and-hold, 60/40, random-agent
- A scorecard with distributions and confidence intervals, not point estimates
- A static HTML report per run and a comparison report across models

### Explicit non-goals for v1.0

| Deferred | Why | Target |
|---|---|---|
| Prediction markets | No clean point-in-time historical data; resolution mechanics and thin liquidity are high-complexity/low-signal | v2 |
| Agent tool use / web research | Multiplies cost, variance, and leak surface. v1.0 is a fixed observation → decision. | v1.1 (bounded tool budget) |
| Intraday data | Weekly cadence makes it irrelevant | v2 |
| Shorting, leverage, options, margin | Accounting complexity, unbounded loss paths | v2 |
| Interactive dashboard | Static HTML is sufficient to inspect a run | v1.2 |
| Multi-agent decomposition (research/risk/execution) | Cannot interpret it before the single-agent baseline exists | v2 |
| Live/forward-test track | Runs in parallel but shares only the sim + ledger; not on the v1.0 critical path | v1.1 |

---

## 2. Harness policy (the leakage rule)

> **Live tools only in live mode.**

**Backtest mode**
- Process runs in a container with **no network egress**. Enforced at the container/network layer, not by prompt instruction.
- The only information source is the frozen snapshot, rendered into the observation.
- Model calls go through a single egress-allowlisted proxy to the inference endpoint. Nothing else resolves.

**Forward/live mode (v1.1)**
- Real-time data and live search are legitimate. Separate config, separate result namespace. Live results are never pooled with backtest results.

### On CLIs

Do **not** use Grok CLI or Codex CLI as the runtime harness.

- They inject their own system prompts, tool sets, and retry behaviour, which you cannot hold constant across models. This violates the *Comparable* design principle in `VISION.md`.
- Grok's live X/web search is precisely the leak vector this project exists to control, and it cannot be reliably disabled.

Instead: implement `agent/clients/` as thin adapters over the raw model APIs (xAI, OpenAI, Anthropic) exposing one method:

```python
def complete(system: str, user: str, *, seed: int | None, temperature: float,
             max_tokens: int, response_format: dict) -> Completion
```

Use the CLIs as *development* tools for building this repo. Reserve Grok's live search for the forward-test track, where live data is allowed by design.

---

## 3. Repo layout

```
tradingbench/
  data/
    build_snapshot.py      # vendor pulls -> frozen parquet + manifest
    store.py               # PointInTimeStore, FutureDataError guard
    corporate_actions.py   # splits/dividends applied by the simulator
  sim/
    ledger.py              # cash, positions, mark-to-market, P&L
    execution.py           # fill model, fees, slippage
    validate.py            # order legality; emits Violation records
    engine.py              # step loop: orders -> fills -> ledger -> snapshot
  agent/
    observation.py         # THE ONLY path from data to a prompt
    prompt.py              # versioned templates
    parser.py              # response -> Decision, with repair pass
    clients/               # xai.py, openai.py, anthropic.py
    baselines.py           # buy_and_hold, sixty_forty, random_agent
  eval/
    metrics.py             # per-episode scorecard
    aggregate.py           # paired stats, bootstrap CIs, power
    contamination/
      blind.py             # anonymisation transform
      probe.py             # memory probe battery + judge
      synthetic.py         # block-bootstrap price paths
      leak_detect.py       # post-as_of entity mentions in reasoning
  report/
    run_report.py          # single-episode HTML
    compare_report.py      # cross-model HTML
  runner.py                # episode + campaign orchestration
  cli.py

configs/                   # YAML campaign definitions
snapshots/{snapshot_id}/   # immutable frozen data
runs/{run_id}/             # per-episode artifacts
tests/
```

### Dependency rule

```
data  →  sim  →  eval  →  report
   ↘   agent   ↗
```

`agent/` may import `data/` **only** through `agent/observation.py`. Enforce with an import-linter rule in CI. This is the single most important structural constraint in the project: if any other module can reach the data store, the no-lookahead guarantee becomes a promise instead of a property.

---

## 4. Data layer

### 4.1 Snapshot

A snapshot is **immutable**. `snapshot_id = sha256(sorted file hashes)[:12]`. Every run records it.

```
snapshots/{snapshot_id}/
  prices.parquet
  corporate_actions.parquet
  news.parquet
  universe.parquet
  manifest.json
```

**`prices.parquet`** — *unadjusted* OHLCV.

| column | type | notes |
|---|---|---|
| `date` | date | exchange local trading date |
| `symbol` | str | |
| `open,high,low,close` | float64 | **raw, unadjusted** |
| `volume` | float64 | |
| `currency` | str | `USD` in v1 |

> **Why unadjusted:** `adj_close` bakes in dividends and splits that occurred *after* the bar. Using it is a subtle lookahead leak. The simulator applies corporate actions on their ex-date instead.

**`corporate_actions.parquet`**

| column | type | notes |
|---|---|---|
| `ex_date` | date | |
| `symbol` | str | |
| `action_type` | enum | `split` \| `cash_dividend` |
| `ratio` | float64 | split ratio, null for dividends |
| `amount` | float64 | dividend per share, null for splits |

**`universe.parquet`** — must include names that were later delisted.

| column | type | notes |
|---|---|---|
| `symbol` | str | |
| `asset_class` | enum | `equity` \| `crypto` |
| `sector` | str | GICS-ish, coarse |
| `listed_from` | date | |
| `delisted_at` | date \| null | |
| `eligible_from` | date | first date the name met the liquidity screen |

> **Survivorship bias:** the universe for an episode is resolved **as of the episode start date** using `eligible_from`/`delisted_at` — never "the top 25 names as of today". Selecting today's large caps and backtesting them through 2025 is the most common silent failure in this genre.

**`news.parquet`**

| column | type | notes |
|---|---|---|
| `published_at` | timestamp (UTC) | **publication**, not crawl, time |
| `symbol` | str \| null | null = macro |
| `headline` | str | |
| `summary` | str | ≤ 400 chars |
| `source` | str | |
| `url_sha256` | str | dedup key; raw URLs not stored |

**`manifest.json`**

```json
{
  "snapshot_id": "a3f9c21b8e04",
  "created_at": "2026-08-08T14:02:11Z",
  "sources": [{"name": "...", "endpoint": "...", "pulled_at": "..."}],
  "files": [{"path": "prices.parquet", "rows": 412885, "sha256": "..."}],
  "coverage": {"start": "2023-01-01", "end": "2026-06-30", "symbols": 30}
}
```

### 4.2 The point-in-time guard

```python
class FutureDataError(RuntimeError): ...

class PointInTimeStore:
    def __init__(self, snapshot: Snapshot, as_of: date): ...

    def prices(self, symbols, lookback_days) -> pd.DataFrame: ...
    def news(self, symbols, lookback_days, limit) -> pd.DataFrame: ...
    def universe(self) -> pd.DataFrame: ...
```

Rules, all enforced in code:

1. `as_of` is set by the runner from episode state. **No caller may pass it**, and no model output may influence it.
2. Every returned frame passes `_guard(df, col)` which raises `FutureDataError` if `df[col].max() > as_of`. News uses `published_at <= as_of` (strict, timestamp-level).
3. `universe()` returns rows where `listed_from <= as_of` and (`delisted_at` is null or `> as_of`) and `eligible_from <= as_of`.
4. The simulator holds a **separate** store instance with `as_of = t+1` for fills. Agent and simulator stores are never the same object.

**Test:** a property-based test that, for 500 random `as_of` dates, calls every public method and asserts no returned row post-dates `as_of`. Plus a mutation test: deliberately break each filter and assert the suite fails.

---

## 5. Simulator

### 5.1 Timeline

```
day t (close)   observation built with as_of = t
                model returns Decision
day t+1 (open)  orders validated and filled
day t+1..t+7    daily mark-to-market
day t+7 (close) next decision
```

Weekly cadence. 12 steps per episode (~one quarter). Crypto decisions align to the same equity-market calendar so all assets fill at a single timestamp.

### 5.2 Execution

| Rule | Value |
|---|---|
| Fill price | `open_{t+1} × (1 + side × slippage_bps/1e4)`, side = +1 buy / −1 sell |
| Slippage | 10 bps equity, 25 bps crypto |
| Commission | 5 bps of notional, no fixed component |
| Fractional shares | allowed (required at $1,000 NAV) |
| Gap guard | if `|open_{t+1}/close_t − 1| > 20%`, fill at `open` and flag `LARGE_GAP` |
| Halts / missing bar | order rejected, `NO_MARKET` violation |

Orders are **notional-denominated**, not share-denominated. This removes a class of arithmetic errors that would otherwise be scored as poor judgment.

### 5.3 Validation rules

Each order is checked in order; first failure rejects the order (not the whole decision):

| Code | Condition |
|---|---|
| `UNKNOWN_SYMBOL` | not in universe as of `t` |
| `NO_MARKET` | no bar on `t+1` |
| `INSUFFICIENT_CASH` | notional + fees > cash after prior orders in the batch |
| `INSUFFICIENT_POSITION` | sell notional > current market value of holding |
| `POSITION_CAP` | resulting weight > 25% of NAV |
| `DUST` | notional < $10 |
| `NO_SHORTING` | sell without a long position |
| `MALFORMED` | schema-invalid after repair pass |

Every rejection appends to `violations.jsonl` and is surfaced in the *next* observation. A model that repeats the same violation is being measured on rule-following, which is a scored dimension.

### 5.4 Ledger

```python
@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    avg_cost: float          # cost basis per unit, split-adjusted on ex-date

@dataclass(frozen=True)
class LedgerState:
    ts: date
    cash: float
    positions: dict[str, Position]
    nav: float
    realized_pnl: float
    fees_paid: float
```

Corporate actions on ex-date: `split` multiplies `qty` and divides `avg_cost` by `ratio`; `cash_dividend` credits `qty × amount` to cash.

**Golden test:** a hand-computed 3-trade / 1-split / 1-dividend scenario with expected cash, qty, avg_cost, and NAV asserted to the cent. This test is the foundation of every downstream claim. Write it before the agent loop exists.

---

## 6. Contracts

### 6.1 Observation (system → model)

Rendered as compact JSON plus a small markdown table. Target ≤ 3,500 tokens so cheap models stay coherent and campaigns stay affordable.

```json
{
  "as_of": "2025-03-14",
  "episode": {"step": 3, "total_steps": 12, "cadence": "weekly"},
  "portfolio": {
    "nav": 1043.21,
    "cash": 212.44,
    "positions": [
      {"symbol": "NVDA", "qty": 0.842, "avg_cost": 118.40,
       "last": 121.02, "market_value": 101.90, "weight": 0.098,
       "unrealized_pnl_pct": 0.0221, "held_steps": 2}
    ]
  },
  "market": [
    {"symbol": "NVDA", "asset_class": "equity", "sector": "Semis",
     "last": 121.02, "ret_1w": -0.031, "ret_1m": 0.042, "ret_3m": 0.115,
     "vol_20d": 0.38, "drawdown_from_52w_high": -0.12}
  ],
  "news": [
    {"published_at": "2025-03-12T13:30:00Z", "symbol": "NVDA",
     "headline": "...", "summary": "..."}
  ],
  "prior_decisions": [
    {"step": 2, "thesis_summary": "...", "orders": ["buy NVDA 100"]}
  ],
  "last_step_violations": [
    {"code": "POSITION_CAP", "symbol": "TSLA", "detail": "would reach 31% of NAV"}
  ],
  "rules": {
    "max_position_weight": 0.25, "min_order_usd": 10.0,
    "shorting": false, "leverage": false,
    "fees_bps": 5, "slippage_bps": {"equity": 10, "crypto": 25},
    "fill": "next session open"
  }
}
```

Notes:
- Derived stats, not raw OHLCV dumps. Cheap models degrade badly on long numeric tables, and it is a large fraction of token cost.
- `news` capped at 40 items, ranked by recency then portfolio relevance. **The cap and ranking are identical for every model.**
- `prior_decisions` is the last 2 steps only. Full history lives in the audit log, not the context. Memory beyond this is a v2 research question, not a v1 confound.

### 6.2 Decision (model → system)

```json
{
  "portfolio_view": "string, <= 600 chars",
  "orders": [
    {"symbol": "NVDA", "side": "buy", "notional_usd": 150.0,
     "thesis": "string, <= 300 chars",
     "confidence": 0.62,
     "horizon_steps": 4}
  ],
  "changed_view_because": "string | null",
  "risk_note": "string, <= 300 chars"
}
```

- `confidence` ∈ [0,1] = stated probability the position is up at `horizon_steps`. This is what calibration is scored against, so it must be defined in the prompt in exactly these words.
- `changed_view_because` must be non-null if any position opened in a prior step is closed. Enforced by the parser; failure is a `MALFORMED` violation.
- Empty `orders` is valid and is *not* penalised. Holding must be a first-class action or you have built a churn incentive.

**Parser:** strict JSON schema validation → on failure, one repair attempt (return the schema error to the model, no new market data) → on second failure, record `MALFORMED`, execute zero orders, continue. Never silently coerce.

---

## 7. Run artifacts

```
runs/{run_id}/
  config.json            # resolved campaign config
  manifest.json          # snapshot_id, model, prompt_version, seed, git_sha, mode
  steps/{nn}/
    observation.json
    prompt.txt
    response_raw.txt
    decision.json
    orders_validated.json
    fills.json
    ledger_after.json
  ledger_daily.parquet
  violations.jsonl
  metrics.json
  probe.json             # contamination probe, if run
  report.html
```

`run_id = {campaign}_{model}_{window}_{seed}_{mode}`. Deterministic and greppable.

**Reproducibility contract:** given `manifest.json` and a cached model response set, replaying a run must produce a byte-identical `ledger_daily.parquet`. There is a CI test for this.

---

## 8. Scorecard

Computed per episode; aggregated across episodes.

**Return & risk** — total return, weekly Sharpe, Sortino, max drawdown, realised vol, best/worst step.

**Behaviour** — turnover (traded notional / mean NAV), fees paid, trade count, mean positions held, HHI concentration, mean holding period in steps, cash drag (mean cash weight).

**Compliance** — violation count by code, invalid order rate, malformed response rate.

**Reasoning quality**
- *Thesis coverage:* share of opened positions with a non-empty, non-boilerplate thesis (LLM judge, 0–3).
- *Churn:* share of positions closed within 1 step of opening — a proxy for thesis instability.
- *Consistency:* judge score for whether `changed_view_because` cites information actually present in that step's observation. This catches confabulation directly.
- *Calibration:* Brier score of `confidence` against the realised sign of the position's return at `horizon_steps`. Report the reliability curve, not just the scalar.

**Cost** — USD, input/output tokens, wall-clock per episode.

**Contamination** — probe score, blind−named delta, real−synthetic delta, leak flag count.

### 8.1 Aggregation and honesty about power

- Never report a bare mean across episodes. Report median, IQR, and a bootstrap 95% CI (10,000 resamples).
- Compare models **paired** on `(window, seed)` — same market path, different model. Paired deltas remove path variance and are the only comparison with usable power at N=40.
- `eval/aggregate.py` emits a **minimum detectable effect** for the achieved N. Print it on the leaderboard. If MDE exceeds the observed spread between models, the report must say the models are indistinguishable rather than ranking them.
- Every model is compared against buy-and-hold, 60/40, and the random agent on the identical path set. A model that does not beat the random agent by more than the CI width has demonstrated nothing.

---

## 9. Contamination harness

### 9.1 Blind vs. named A/B

There are **four run modes**, and only two of them form the A/B pair:

| Mode | Symbols | News | Prices | Role |
|---|---|---|---|---|
| `standard` | real | yes | real | The headline run. All scorecard numbers come from here. |
| `named_control` | real | **no** | real | A/B arm 1 |
| `blind` | pseudonymous | **no** | rebased to 100 | A/B arm 2 |
| `synthetic` | real | yes | block-bootstrapped | Skill control (§9.3) |

Critical detail: **the A/B pair is `named_control` vs `blind`, not `standard` vs `blind`.** News is withheld from *both* arms. If you compare `standard` against `blind`, the delta measures "news helps" confounded with "identifiability helps" and tells you nothing. Isolate one variable.

`named_control` and `blind` run on the same window, same seed, same underlying path.

`blind_premium = mean_paired(named_control_return − blind_return)`. A large positive premium on historical windows and ~zero on post-cutoff windows is the signature of recall, not skill.

### 9.2 Memory probe battery

Before each episode, off the record:

> As of {as_of}, describe what you expect for {symbol} over the following 90 days. If you recall actual events after {as_of}, state them explicitly.

Score 0–3 against ground truth from the snapshot (LLM judge, rubric in `contamination/probe.py`):
`0` no recall · `1` vague directional · `2` specific direction with rough magnitude · `3` specific dated events.

Store per model per window. **Publish it as a leaderboard column.** A model with mean probe 2.4 and a strong return is reporting a different quantity than a model with probe 0.3.

### 9.3 Synthetic control

Stationary block bootstrap of returns, block length 10 trading days, resampling the **date index jointly across all symbols** so cross-sectional correlation and volatility clustering survive. Names and news are kept — only the future is counterfactual.

`skill_signal = mean(real_alpha) − mean(synthetic_alpha)`. Persistent alpha on real paths that vanishes on statistically matched synthetic paths is memorization.

### 9.4 Leak detection

Scan `portfolio_view`, `thesis`, `changed_view_because` for named entities and dated events post-dating `as_of`. Two-stage: cheap regex/NER pass → LLM judge on hits. Emits `LOOKAHEAD_MENTION` flags. These do not invalidate a run but appear on the scorecard.

### 9.5 Window design

| Track | Purpose |
|---|---|
| **Post-cutoff windows** | Headline number. Cleanest signal, smallest sample. |
| **Historical windows** | Diagnostic. Larger sample, contaminated — always reported next to the probe score. |
| **Synthetic windows** | Control arm. |

Never pool these into one average.

---

## 10. Configuration

```yaml
# configs/campaign_v1.yaml
campaign: v1_baseline
snapshot_id: a3f9c21b8e04

universe:
  equity: [AAPL, MSFT, NVDA, ...]      # 25 names, liquidity-screened as-of
  crypto: [BTC, ETH, SOL, LINK, AVAX]

episode:
  starting_cash: 1000.0
  cadence: weekly
  steps: 12
  max_position_weight: 0.25

windows:                                # 8 start dates
  - {id: w1, start: 2025-01-06, track: historical}
  - {id: w6, start: 2026-04-06, track: post_cutoff}

seeds: [1, 2, 3, 4, 5]                  # 8 x 5 = 40 episodes per model per mode

models:
  - {id: luna,  provider: xai,    name: "...", temperature: 0.7, max_tokens: 2000}
  - {id: terra, provider: openai, name: "...", temperature: 0.7, max_tokens: 2000}

baselines: [buy_and_hold, sixty_forty, random_agent]
modes: [standard, named_control, blind, synthetic]   # see §9.1
prompt_version: p1
```

Prompt templates are versioned and hashed into `manifest.json`. Changing a prompt starts a new campaign — it never silently invalidates prior results.

---

## 11. Build order

Each phase has an acceptance test. Do not start the next phase until it passes.

### Phase 1 — Data + simulator, zero LLM
Build `data/`, `sim/`, `agent/baselines.py`.

**Accept when:** the golden ledger test passes to the cent; buy-and-hold through the engine matches an independent pandas calculation within 1 bp; the property-based lookahead test passes over 500 random dates; the mutation test confirms the guard actually fails when broken.

*This phase is the entire engineering risk. If the ledger or the `as_of` guard is wrong, nothing downstream means anything.*

### Phase 2 — Agent loop, one model, one episode
Build `agent/`, `runner.py` single-episode path.

**Accept when:** a 12-step episode completes end to end; every step has a complete artifact set; malformed responses are handled without crashing; replay from cached responses is byte-identical.

### Phase 3 — Campaign scale + contamination
Build `eval/`, parallel episode execution, all three modes.

**Accept when:** 40 episodes × 2 models × 4 modes complete (320 episodes — budget for this: at ~4k tokens/step × 12 steps, this is the campaign's dominant cost); `aggregate.py` produces paired CIs and an MDE; probe scores are populated; the random-agent baseline is included in every comparison.

### Phase 4 — Reporting
Build `report/`.

**Accept when:** someone who has not read the code can open `compare_report.html`, see the leaderboard *with* CIs and probe scores, click into a single episode, and read the model's reasoning next to the resulting equity curve.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| N=40 still underpowered | Publish the MDE. Add windows before adding models. Underpowered-but-honest beats a confident false ranking. |
| Judge-based metrics (thesis, consistency) are noisy | Fixed rubric, fixed judge model, judge version in manifest. Spot-check 20 samples by hand and report judge–human agreement. |
| News corpus has survivorship/coverage bias | Document coverage in the manifest. Both A/B arms are price-only, so the headline contamination result is unaffected. |
| Post-cutoff window shrinks as models update | Treat window freshness as a maintained asset. Re-snapshot quarterly. |
| Cheap models produce unparseable output | One repair pass, then `MALFORMED` + zero orders. Malformed rate is itself a reported metric. |
| Scope creep toward the autonomous OS | v1.0 ships with the non-goals table intact. The vision doc's long-term section is explicitly out of scope. |

---

## 13. Open decisions

1. **Crypto in v1.0?** Cheap to add (OHLCV only) but adds a vol regime that may dominate a $1,000 portfolio's variance. Recommend: include, cap crypto at 40% of NAV in aggregate.
2. **Judge model.** Should be a model *not* under evaluation, to avoid self-preference. Pin the version.
3. **Temperature.** 0.7 across all models for behavioural realism, with seed variation supplying the sampling spread — or 0.0 with variation coming only from windows? Recommend 0.7; deterministic agents understate real deployment variance.
4. **Episode length.** 12 weeks maximises sample count; 26 weeks tests thesis persistence better. Recommend 12 for v1.0 and revisit once power is measured.
