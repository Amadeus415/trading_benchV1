"""Cross-model comparison HTML report.

Primary leaderboard column is selection_alpha (when present); total return
is secondary. Includes CIs, MDE, Wilcoxon, and contamination premiums.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (x != x)):  # NaN
        return "—"
    return f"{x:+.{digits}%}"


def _fmt_num(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x:+.{digits}f}"


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
    contamination = aggregate.get("contamination", {})
    primary = aggregate.get("primary_metric", "total_return")

    def sort_key(kv):
        name, m = kv
        if primary in m and isinstance(m[primary], dict):
            return m[primary].get("mean", -999)
        return m.get("total_return", {}).get("mean", -999)

    rows = []
    for name, m in sorted(models.items(), key=sort_key, reverse=True):
        tr = m.get("total_return") or {}
        sa = m.get("selection_alpha") or {}
        mde = m.get("mde", m.get("mde_return", float("nan")))
        mde_s = _fmt_pct(mde) if isinstance(mde, (int, float)) else "—"
        rows.append(
            f"<tr>"
            f"<td><strong>{html.escape(name)}</strong></td>"
            f"<td>{m['n']}</td>"
            f"<td>{_fmt_num(sa.get('mean'))}</td>"
            f"<td>{_fmt_num(sa.get('median'))}</td>"
            f"<td>[{_fmt_num(sa.get('lo'))}, {_fmt_num(sa.get('hi'))}]</td>"
            f"<td>{_fmt_pct(tr.get('mean'))}</td>"
            f"<td>{_fmt_pct(tr.get('median'))}</td>"
            f"<td>[{_fmt_pct(tr.get('lo'))}, {_fmt_pct(tr.get('hi'))}]</td>"
            f"<td>{_fmt_pct(m.get('max_drawdown', {}).get('median'))}</td>"
            f"<td>{m.get('violation_count_mean', 0):.1f}</td>"
            f"<td>{mde_s}</td>"
            f"</tr>"
        )

    paired_rows = []
    for name, p in paired.items():
        d = p.get("delta") or p.get("delta_return") or {}
        dr = p.get("delta_return") or {}
        wx = p.get("wilcoxon") or {}
        flag = "indistinguishable" if p.get("indistinguishable") else "signal?"
        if wx.get("significant_05"):
            flag = "Wilcoxon p<0.05"
        p_val = wx.get("p_value")
        p_s = f"{p_val:.3f}" if isinstance(p_val, float) and p_val == p_val else "—"
        paired_rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{p['n_pairs']}</td>"
            f"<td>{_fmt_num(d.get('mean'))}</td>"
            f"<td>[{_fmt_num(d.get('lo'))}, {_fmt_num(d.get('hi'))}]</td>"
            f"<td>{_fmt_pct(dr.get('mean'))}</td>"
            f"<td>{_fmt_pct(p.get('mde'))}</td>"
            f"<td>{p_s}</td><td>{flag}</td></tr>"
        )

    contam_rows = []
    for name, c in contamination.items():
        bp = c.get("blind_premium") or {}
        contam_rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{c.get('n_pairs', 0)}</td>"
            f"<td>{_fmt_pct(bp.get('mean'))}</td>"
            f"<td>[{_fmt_pct(bp.get('lo'))}, {_fmt_pct(bp.get('hi'))}]</td>"
            f"<td>{html.escape(c.get('note', ''))}</td></tr>"
        )

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Trading Bench — {html.escape(campaign)} comparison</title>
<style>
  :root {{ --bg:#0f1419; --card:#1a2332; --text:#e7ecf3; --muted:#8b9bb4; --accent:#3d9cf0; }}
  body {{ font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif; background:var(--bg); color:var(--text); margin:0; padding:2rem; }}
  h1,h2 {{ font-weight:600; }}
  .meta {{ color:var(--muted); margin-bottom:1.5rem; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card); border-radius:10px; overflow:hidden; margin-bottom:1.5rem; }}
  th,td {{ text-align:left; padding:0.55rem 0.75rem; border-bottom:1px solid #2a3548; font-variant-numeric:tabular-nums; }}
  th {{ color:var(--muted); font-weight:500; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.04em; }}
  .note {{ background:var(--card); border-left:3px solid var(--accent); padding:0.8rem 1rem; margin:1.2rem 0; color:#c5d0e0; }}
  .warn {{ border-left-color:#e0a030; }}
  footer {{ margin-top:2rem; color:var(--muted); font-size:0.8rem; }}
  code {{ background:#0d1117; padding:0.1em 0.35em; border-radius:4px; }}
</style>
</head>
<body>
  <h1>Campaign comparison</h1>
  <div class="meta">campaign: {html.escape(campaign)} · episodes: {aggregate.get('n_episodes', 0)} · primary metric: <code>{html.escape(primary)}</code></div>

  <div class="note">
    <strong>Selection alpha first.</strong> Cumulative return is mostly factor exposure (KTD-Fin).
    Primary column is factor-adjusted selection alpha when attribution ran; total return is secondary.
    Bootstrap 95% CIs and MDE are always shown. If model spread &lt; MDE, report <em>indistinguishable</em> — do not invent a ranking.
  </div>

  <div class="note warn">
    A null finding — no model beats <code>momentum_3m</code> — is a publishable result, not a failure.
    Design for that being the finding (MVP_SPEC §0, §12).
  </div>

  <h2>Leaderboard</h2>
  <table>
    <thead>
      <tr>
        <th>Model</th><th>N</th>
        <th>α mean</th><th>α med</th><th>α 95% CI</th>
        <th>Ret mean</th><th>Ret med</th><th>Ret 95% CI</th>
        <th>Med. MaxDD</th><th>Violations</th><th>MDE</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows) or '<tr><td colspan=11>No data</td></tr>'}
    </tbody>
  </table>

  <h2>Paired comparisons (same window × seed)</h2>
  <table>
    <thead>
      <tr>
        <th>Pair</th><th>N pairs</th><th>Δ primary</th><th>95% CI</th>
        <th>Δ return</th><th>MDE</th><th>Wilcoxon p</th><th>Status</th>
      </tr>
    </thead>
    <tbody>
      {''.join(paired_rows) or '<tr><td colspan=8>No paired data</td></tr>'}
    </tbody>
  </table>

  <h2>Contamination: blind premium (bright − blinded)</h2>
  <table>
    <thead>
      <tr><th>Model</th><th>N pairs</th><th>Premium mean</th><th>95% CI</th><th>Note</th></tr>
    </thead>
    <tbody>
      {''.join(contam_rows) or '<tr><td colspan=5>No matched bright/blinded cells in this campaign</td></tr>'}
    </tbody>
  </table>

  <h2>Raw aggregate JSON</h2>
  <pre style="background:#1a2332;padding:1rem;border-radius:8px;overflow:auto;font-size:0.75rem;">{html.escape(json.dumps(aggregate, indent=2, default=str))}</pre>

  <footer>Trading Bench v2 · locked harness · selection alpha primary · auditable</footer>
</body>
</html>
"""
    out_path.write_text(body)
    return out_path
