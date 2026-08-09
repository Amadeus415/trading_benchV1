"""Cross-model comparison HTML report."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_compare_report(
    out_path: str | Path,
    *,
    campaign: str,
    aggregate: dict[str, Any],
    episode_metrics: list[dict],
) -> Path:
    out_path = Path(out_path)
    models = aggregate.get("models", {})
    paired = aggregate.get("paired", {})

    rows = []
    for name, m in sorted(models.items(), key=lambda kv: kv[1].get("total_return", {}).get("mean", -999), reverse=True):
        tr = m["total_return"]
        mde = m.get("mde_return", float("nan"))
        rows.append(
            f"<tr>"
            f"<td><strong>{html.escape(name)}</strong></td>"
            f"<td>{m['n']}</td>"
            f"<td>{tr['mean']:+.2%}</td>"
            f"<td>{tr['median']:+.2%}</td>"
            f"<td>[{tr['lo']:+.2%}, {tr['hi']:+.2%}]</td>"
            f"<td>{m.get('max_drawdown',{}).get('median',0):.2%}</td>"
            f"<td>{m.get('violation_count_mean',0):.1f}</td>"
            f"<td>{mde:.2%}</td>"
            f"</tr>"
        )

    paired_rows = []
    for name, p in paired.items():
        d = p["delta_return"]
        flag = "indistinguishable" if p.get("indistinguishable") else "signal?"
        paired_rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{p['n_pairs']}</td>"
            f"<td>{d['mean']:+.2%}</td>"
            f"<td>[{d['lo']:+.2%}, {d['hi']:+.2%}]</td>"
            f"<td>{p['mde']:.2%}</td><td>{flag}</td></tr>"
        )

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Trading Bench — {html.escape(campaign)} comparison</title>
<style>
  :root {{ --bg:#0f1419; --card:#1a2332; --text:#e7ecf3; --muted:#8b9bb4; }}
  body {{ font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif; background:var(--bg); color:var(--text); margin:0; padding:2rem; }}
  h1,h2 {{ font-weight:600; }}
  .meta {{ color:var(--muted); margin-bottom:1.5rem; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card); border-radius:10px; overflow:hidden; }}
  th,td {{ text-align:left; padding:0.55rem 0.75rem; border-bottom:1px solid #2a3548; font-variant-numeric:tabular-nums; }}
  th {{ color:var(--muted); font-weight:500; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.04em; }}
  .note {{ background:var(--card); border-left:3px solid #3d9cf0; padding:0.8rem 1rem; margin:1.2rem 0; color:#c5d0e0; }}
  footer {{ margin-top:2rem; color:var(--muted); font-size:0.8rem; }}
</style>
</head>
<body>
  <h1>Campaign comparison</h1>
  <div class="meta">campaign: {html.escape(campaign)} · episodes: {aggregate.get('n_episodes', 0)}</div>

  <div class="note">
    Never trust a bare mean. Numbers below include bootstrap 95% CIs and a minimum detectable effect (MDE).
    If the spread between models is smaller than the MDE, treat them as indistinguishable.
  </div>

  <h2>Leaderboard (total return)</h2>
  <table>
    <thead>
      <tr>
        <th>Model</th><th>N</th><th>Mean</th><th>Median</th><th>95% CI</th>
        <th>Med. MaxDD</th><th>Violations</th><th>MDE</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows) or '<tr><td colspan=8>No data</td></tr>'}
    </tbody>
  </table>

  <h2>Paired comparisons (same window × seed)</h2>
  <table>
    <thead>
      <tr><th>Pair</th><th>N pairs</th><th>Δ mean</th><th>95% CI</th><th>MDE</th><th>Status</th></tr>
    </thead>
    <tbody>
      {''.join(paired_rows) or '<tr><td colspan=6>No paired data</td></tr>'}
    </tbody>
  </table>

  <h2>Raw aggregate JSON</h2>
  <pre style="background:#1a2332;padding:1rem;border-radius:8px;overflow:auto;font-size:0.75rem;">{html.escape(json.dumps(aggregate, indent=2, default=str))}</pre>

  <footer>Trading Bench v1 · comparable · reproducible · auditable</footer>
</body>
</html>
"""
    out_path.write_text(body)
    return out_path
