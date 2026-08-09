"""Build a frozen synthetic snapshot for offline, reproducible MVP runs.

Real vendor pulls can replace this later; the schema and manifest contract
remain identical so the rest of the system does not care about the source.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Small, explicit universe for the MVP (equities + crypto).
EQUITY_UNIVERSE = [
    ("AAPL", "Technology"),
    ("MSFT", "Technology"),
    ("NVDA", "Semis"),
    ("GOOGL", "Technology"),
    ("AMZN", "Consumer"),
    ("META", "Technology"),
    ("TSLA", "Consumer"),
    ("JPM", "Financials"),
    ("V", "Financials"),
    ("UNH", "Healthcare"),
    ("XOM", "Energy"),
    ("JNJ", "Healthcare"),
    ("WMT", "Consumer"),
    ("PG", "Consumer"),
    ("MA", "Financials"),
    ("HD", "Consumer"),
    ("CVX", "Energy"),
    ("MRK", "Healthcare"),
    ("ABBV", "Healthcare"),
    ("KO", "Consumer"),
    ("PEP", "Consumer"),
    ("COST", "Consumer"),
    ("AVGO", "Semis"),
    ("LLY", "Healthcare"),
    ("BAC", "Financials"),
]

CRYPTO_UNIVERSE = [
    ("BTC", "Crypto"),
    ("ETH", "Crypto"),
    ("SOL", "Crypto"),
    ("LINK", "Crypto"),
    ("AVAX", "Crypto"),
]

# Deterministic seed prices (approximate order-of-magnitude).
SEED_PRICES = {
    "AAPL": 180.0, "MSFT": 380.0, "NVDA": 120.0, "GOOGL": 140.0, "AMZN": 175.0,
    "META": 480.0, "TSLA": 250.0, "JPM": 190.0, "V": 270.0, "UNH": 520.0,
    "XOM": 105.0, "JNJ": 155.0, "WMT": 160.0, "PG": 155.0, "MA": 440.0,
    "HD": 340.0, "CVX": 150.0, "MRK": 120.0, "ABBV": 170.0, "KO": 60.0,
    "PEP": 170.0, "COST": 720.0, "AVGO": 1300.0, "LLY": 750.0, "BAC": 35.0,
    "BTC": 42000.0, "ETH": 2200.0, "SOL": 95.0, "LINK": 14.0, "AVAX": 35.0,
}


def _trading_days(start: date, end: date) -> list[date]:
    """Weekdays only (simplified equity calendar; crypto uses same for alignment)."""
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_synthetic_prices(
    start: date = date(2023, 1, 3),
    end: date = date(2026, 6, 30),
    seed: int = 42,
) -> pd.DataFrame:
    """Generate unadjusted OHLCV with correlated drift + noise."""
    rng = np.random.default_rng(seed)
    days = _trading_days(start, end)
    rows = []

    # Per-asset annualized drift / vol (roughly plausible)
    params = {}
    for sym, _ in EQUITY_UNIVERSE:
        params[sym] = (0.08 + rng.normal(0, 0.04), 0.18 + abs(rng.normal(0, 0.05)))
    for sym, _ in CRYPTO_UNIVERSE:
        params[sym] = (0.15 + rng.normal(0, 0.1), 0.55 + abs(rng.normal(0, 0.1)))

    # Shared market factor
    market_rets = rng.normal(0.0003, 0.01, size=len(days))

    for sym, (mu, vol) in params.items():
        price = SEED_PRICES[sym]
        daily_mu = mu / 252
        daily_vol = vol / np.sqrt(252)
        beta = 0.7 if sym not in dict(CRYPTO_UNIVERSE) else 0.4
        idio = 0.6 if sym not in dict(CRYPTO_UNIVERSE) else 0.9

        for i, d in enumerate(days):
            r = beta * market_rets[i] + idio * rng.normal(daily_mu, daily_vol)
            # Occasional jumps
            if rng.random() < 0.002:
                r += rng.choice([-1, 1]) * rng.uniform(0.05, 0.12)
            open_p = price * (1 + rng.normal(0, daily_vol * 0.3))
            close_p = max(price * (1 + r), 0.01)
            high_p = max(open_p, close_p) * (1 + abs(rng.normal(0, daily_vol * 0.2)))
            low_p = min(open_p, close_p) * (1 - abs(rng.normal(0, daily_vol * 0.2)))
            vol_shares = float(rng.lognormal(12, 0.5)) if sym not in dict(CRYPTO_UNIVERSE) else float(rng.lognormal(14, 0.8))
            rows.append({
                "date": d,
                "symbol": sym,
                "open": round(open_p, 4),
                "high": round(high_p, 4),
                "low": round(low_p, 4),
                "close": round(close_p, 4),
                "volume": vol_shares,
                "currency": "USD",
            })
            price = close_p

    return pd.DataFrame(rows)


def generate_corporate_actions(prices: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """A few deterministic splits and dividends for golden-path testing."""
    rng = np.random.default_rng(seed + 7)
    rows = []
    # Fixed actions for reproducibility / tests
    rows.append({
        "ex_date": date(2024, 6, 10),
        "symbol": "NVDA",
        "action_type": "split",
        "ratio": 10.0,
        "amount": None,
    })
    rows.append({
        "ex_date": date(2025, 2, 14),
        "symbol": "AAPL",
        "action_type": "cash_dividend",
        "ratio": None,
        "amount": 0.25,
    })
    rows.append({
        "ex_date": date(2025, 5, 15),
        "symbol": "MSFT",
        "action_type": "cash_dividend",
        "ratio": None,
        "amount": 0.75,
    })
    # A few random small dividends
    equities = [s for s, _ in EQUITY_UNIVERSE]
    for _ in range(8):
        sym = rng.choice(equities)
        year = int(rng.choice([2024, 2025]))
        month = int(rng.integers(1, 12))
        day = int(rng.integers(1, 28))
        rows.append({
            "ex_date": date(year, month, day),
            "symbol": sym,
            "action_type": "cash_dividend",
            "ratio": None,
            "amount": round(float(rng.uniform(0.1, 1.5)), 4),
        })
    return pd.DataFrame(rows)


def generate_universe() -> pd.DataFrame:
    rows = []
    for sym, sector in EQUITY_UNIVERSE:
        rows.append({
            "symbol": sym,
            "asset_class": "equity",
            "sector": sector,
            "listed_from": date(2015, 1, 1),
            "delisted_at": None,
            "eligible_from": date(2023, 1, 1),
        })
    for sym, sector in CRYPTO_UNIVERSE:
        rows.append({
            "symbol": sym,
            "asset_class": "crypto",
            "sector": sector,
            "listed_from": date(2018, 1, 1),
            "delisted_at": None,
            "eligible_from": date(2023, 1, 1),
        })
    # Survivorship example: a delisted name that should not appear after delist date
    rows.append({
        "symbol": "FAKECO",
        "asset_class": "equity",
        "sector": "Industrials",
        "listed_from": date(2018, 1, 1),
        "delisted_at": date(2024, 3, 15),
        "eligible_from": date(2020, 1, 1),
    })
    return pd.DataFrame(rows)


def generate_news(prices: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 99)
    symbols = prices["symbol"].unique().tolist()
    headlines = [
        "{sym} reports quarterly results",
        "Analysts raise price target on {sym}",
        "{sym} announces product update",
        "Regulatory scrutiny around {sym}",
        "Macro: Fed holds rates steady",
        "Macro: CPI print in line with expectations",
        "{sym} expands into new market",
        "Supply chain update for {sym}",
        "Institutional flow note on {sym}",
        "Sector rotation favors {sector}",
    ]
    rows = []
    dates = sorted(prices["date"].unique())
    # ~2 headlines per week across the universe
    for d in dates[::3]:
        n = int(rng.integers(1, 4))
        for _ in range(n):
            is_macro = rng.random() < 0.2
            if is_macro:
                sym = None
                sector = "Macro"
                tmpl = rng.choice([h for h in headlines if "Macro" in h or "Fed" in h or "CPI" in h] or headlines)
                headline = tmpl.format(sym="markets", sector=sector)
            else:
                sym = rng.choice(symbols)
                sector = "Tech"
                tmpl = rng.choice(headlines)
                headline = tmpl.format(sym=sym, sector=sector)
            hour = int(rng.integers(8, 20))
            minute = int(rng.integers(0, 60))
            published = datetime(d.year, d.month, d.day, hour, minute, tzinfo=timezone.utc)
            summary = f"{headline}. Brief synthetic summary for evaluation harness."
            url_key = hashlib.sha256(f"{published.isoformat()}|{headline}".encode()).hexdigest()
            rows.append({
                "published_at": published,
                "symbol": sym,
                "headline": headline[:200],
                "summary": summary[:400],
                "source": "synthetic",
                "url_sha256": url_key,
            })
    return pd.DataFrame(rows)


def build_snapshot(
    out_dir: str | Path,
    start: date = date(2023, 1, 3),
    end: date = date(2026, 6, 30),
    seed: int = 42,
) -> Path:
    """Write parquet files + manifest; return snapshot directory path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    prices = generate_synthetic_prices(start=start, end=end, seed=seed)
    actions = generate_corporate_actions(prices, seed=seed)
    universe = generate_universe()
    news = generate_news(prices, seed=seed)

    prices.to_parquet(out / "prices.parquet", index=False)
    actions.to_parquet(out / "corporate_actions.parquet", index=False)
    universe.to_parquet(out / "universe.parquet", index=False)
    news.to_parquet(out / "news.parquet", index=False)

    files = []
    digests = []
    for name in ["prices.parquet", "corporate_actions.parquet", "universe.parquet", "news.parquet"]:
        p = out / name
        digest = _file_sha256(p)
        digests.append(f"{name}:{digest}")
        if name == "prices.parquet":
            rows = len(prices)
        elif name == "corporate_actions.parquet":
            rows = len(actions)
        elif name == "universe.parquet":
            rows = len(universe)
        else:
            rows = len(news)
        files.append({"path": name, "rows": rows, "sha256": digest})

    snapshot_id = hashlib.sha256("\n".join(sorted(digests)).encode()).hexdigest()[:12]

    manifest = {
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": [
            {
                "name": "synthetic_generator",
                "endpoint": "local",
                "pulled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "seed": seed,
            }
        ],
        "files": files,
        "coverage": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "symbols": int(prices["symbol"].nunique()),
        },
    }

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Rename folder to snapshot_id if out_dir is a parent
    final = out
    if out.name != snapshot_id:
        final = out.parent / snapshot_id
        if final.resolve() != out.resolve():
            if final.exists():
                # overwrite contents in place if re-running
                for f in out.iterdir():
                    target = final / f.name
                    if target.exists():
                        target.unlink()
                    f.rename(target)
                try:
                    out.rmdir()
                except OSError:
                    pass
            else:
                out.rename(final)
            # rewrite paths already written — files moved
            with open(final / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

    # Ensure snapshot_id is in the directory name path returned
    if final.name != snapshot_id:
        # write into named subdir
        named = final / snapshot_id if False else final
        return named

    # Also write a symlink/copy pointer convenience: snapshot_id file
    (final / "SNAPSHOT_ID").write_text(snapshot_id + "\n")
    return final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build synthetic Trading Bench snapshot")
    parser.add_argument("--out", default="snapshots", help="Output parent directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dest = Path(args.out) / "_building"
    path = build_snapshot(dest, seed=args.seed)
    print(f"Wrote snapshot to {path}")
    print(f"snapshot_id={(path / 'SNAPSHOT_ID').read_text().strip()}")
