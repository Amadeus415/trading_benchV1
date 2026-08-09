"""Single-episode static HTML report."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_run_report(
    out_path: str | Path,
    *,
    run_id: str,
    manifest: dict,
    metrics: dict,
    ledger_daily: pd.DataFrame,
    steps: list[dict[str, Any]],
    violations: list[dict],
) -> Path:
    out_path = Path(out_path)
    nav_points = []
    if ledger_daily is not None and not ledger_daily.empty:
        for _, row in ledger_daily.iterrows():
            nav_points.append((str(row["date"]), float(row["nav"])))

    max_nav = max((p[1] for p in nav_points), default=1000) or 1000
    min_nav = min((p[1] for p in nav_points), default=1000) or 1000
    # SVG sparkline
    w, h = 720, 180
    spark = _sparkline(nav_points, w, h, min_nav * 0.98, max_nav * 1.02)

    step_html = []
    for s in steps:
        dec = s.get("decision") or {}
        view = html.escape(str(dec.get("portfolio_view") or "")[:500])
        orders = dec.get("orders") or []
        order_lines = ", ".join(
            f"{o.get('side')} {o.get('symbol')} ${o.get('notional_usd')}" for o in orders
        ) or "(hold)"
        step_html.append(
            f"<div class='step'><h3>Step {s.get('step')} — {html.escape(str(s.get('decision_date','')))}</h3>"
            f"<p class='view'>{view}</p>"
            f"<p><strong>Orders:</strong> {html.escape(order_lines)}</p>"
            f"<p><strong>NAV after:</strong> ${(s.get('ledger_after') or {}).get('nav', 0):.2f}</p>"
            f"</div>"
        )

    viol_rows = "".join(
        f"<tr><td>{html.escape(str(v.get('code')))}</td>"
        f"<td>{html.escape(str(v.get('symbol')))}</td>"
        f"<td>{html.escape(str(v.get('detail')))}</td></tr>"
        for v in violations
    ) or "<tr><td colspan=3>None</td></tr>"

    m = metrics
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Trading Bench — {html.escape(run_id)}</title>
<style>
  :root {{ --bg:#0f1419; --card:#1a2332; --text:#e7ecf3; --muted:#8b9bb4; --accent:#3d9cf0; --good:#3ecf8e; --bad:#f07178; }}
  body {{ font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif; background:var(--bg); color:var(--text); margin:0; padding:2rem; }}
  h1,h2,h3 {{ font-weight:600; letter-spacing:-0.02em; }}
  .meta {{ color:var(--muted); font-size:0.9rem; margin-bottom:1.5rem; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:1rem; margin:1.5rem 0; }}
  .card {{ background:var(--card); border-radius:10px; padding:1rem 1.1rem; border:1px solid #2a3548; }}
  .card .label {{ color:var(--muted); font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; }}
  .card .value {{ font-size:1.4rem; font-variant-numeric:tabular-nums; margin-top:0.25rem; }}
  .good {{ color:var(--good); }} .bad {{ color:var(--bad); }}
  svg {{ width:100%; max-width:{w}px; height:auto; background:var(--card); border-radius:10px; border:1px solid #2a3548; }}
  .step {{ background:var(--card); border-radius:10px; padding:1rem 1.2rem; margin:0.75rem 0; border:1px solid #2a3548; }}
  .view {{ color:#c5d0e0; font-size:0.95rem; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.9rem; }}
  th,td {{ text-align:left; padding:0.45rem 0.6rem; border-bottom:1px solid #2a3548; }}
  th {{ color:var(--muted); font-weight:500; }}
  footer {{ margin-top:2rem; color:var(--muted); font-size:0.8rem; }}
</style>
</head>
<body>
  <h1>Episode report</h1>
  <div class="meta">
    <div><strong>run_id:</strong> {html.escape(run_id)}</div>
    <div><strong>model:</strong> {html.escape(str(manifest.get('model')))} ·
         <strong>window:</strong> {html.escape(str(manifest.get('window')))} ·
         <strong>seed:</strong> {html.escape(str(manifest.get('seed')))} ·
         <strong>mode:</strong> {html.escape(str(manifest.get('mode')))}</div>
    <div><strong>snapshot:</strong> {html.escape(str(manifest.get('snapshot_id')))} ·
         <strong>prompt:</strong> {html.escape(str(manifest.get('prompt_version')))}</div>
  </div>

  <div class="grid">
    <div class="card"><div class="label">Total return</div>
      <div class="value {'good' if m.get('total_return',0)>=0 else 'bad'}">{m.get('total_return',0):+.2%}</div></div>
    <div class="card"><div class="label">Final NAV</div>
      <div class="value">${m.get('final_nav',0):.2f}</div></div>
    <div class="card"><div class="label">Max drawdown</div>
      <div class="value bad">{m.get('max_drawdown',0):.2%}</div></div>
    <div class="card"><div class="label">Sharpe (weekly)</div>
      <div class="value">{m.get('weekly_sharpe',0):.2f}</div></div>
    <div class="card"><div class="label">Trades</div>
      <div class="value">{m.get('trade_count',0)}</div></div>
    <div class="card"><div class="label">Violations</div>
      <div class="value">{m.get('violation_count',0)}</div></div>
    <div class="card"><div class="label">Fees</div>
      <div class="value">${m.get('fees_paid',0):.2f}</div></div>
    <div class="card"><div class="label">Turnover</div>
      <div class="value">{m.get('turnover',0):.2f}×</div></div>
  </div>

  <h2>Equity curve</h2>
  {spark}

  <h2>Decisions</h2>
  {''.join(step_html) or '<p>No steps recorded.</p>'}

  <h2>Violations</h2>
  <table>
    <thead><tr><th>Code</th><th>Symbol</th><th>Detail</th></tr></thead>
    <tbody>{viol_rows}</tbody>
  </table>

  <h2>Metrics JSON</h2>
  <pre style="background:#1a2332;padding:1rem;border-radius:8px;overflow:auto;font-size:0.8rem;">{html.escape(json.dumps(m, indent=2))}</pre>

  <footer>Trading Bench v1 · fake money only · deterministic paper accounting</footer>
</body>
</html>
"""
    out_path.write_text(body)
    return out_path


def _sparkline(points: list[tuple[str, float]], w: int, h: int, y0: float, y1: float) -> str:
    if not points:
        return "<p>No NAV series.</p>"
    pad = 12
    n = len(points)
    xs = [pad + i * (w - 2 * pad) / max(1, n - 1) for i in range(n)]
    ys = []
    span = (y1 - y0) or 1.0
    for _, v in points:
        ys.append(h - pad - (v - y0) / span * (h - 2 * pad))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    last = points[-1][1]
    color = "#3ecf8e" if last >= points[0][1] else "#f07178"
    return (
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{poly}"/>'
        f'<text x="{pad}" y="{h - 4}" fill="#8b9bb4" font-size="11">'
        f'{points[0][0]} → {points[-1][0]} · ${points[0][1]:.0f} → ${last:.0f}</text>'
        f"</svg>"
    )
