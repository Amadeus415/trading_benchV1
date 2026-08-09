"""CLI entrypoint: tradingbench <command>."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.group()
@click.version_option(package_name="tradingbench")
def main() -> None:
    """Trading Bench — paper-trading evaluation harness for LLM agents."""


@main.command("build-snapshot")
@click.option("--out", default="snapshots", show_default=True, help="Output parent directory")
@click.option("--seed", default=42, show_default=True, type=int)
def build_snapshot_cmd(out: str, seed: int) -> None:
    """Generate a frozen synthetic market snapshot."""
    from tradingbench.data.build_snapshot import build_snapshot

    dest = Path(out) / "_building"
    path = build_snapshot(dest, seed=seed)
    sid = (path / "SNAPSHOT_ID").read_text().strip() if (path / "SNAPSHOT_ID").exists() else path.name
    click.echo(f"Wrote snapshot {sid} → {path}")


@main.command("run-episode")
@click.option("--snapshot", required=True, type=click.Path(exists=True), help="Snapshot directory")
@click.option("--model", default="buy_and_hold", show_default=True, help="Baseline or model id")
@click.option("--provider", default="baseline", show_default=True)
@click.option("--model-name", default=None, help="API model name (if not baseline)")
@click.option("--start", default="2025-01-06", show_default=True, help="Episode start date")
@click.option("--window-id", default="w1", show_default=True)
@click.option("--seed", default=1, show_default=True, type=int)
@click.option("--mode", default="standard", show_default=True,
              type=click.Choice(["standard", "named_control", "blind", "synthetic"]))
@click.option("--steps", default=12, show_default=True, type=int)
@click.option("--runs-dir", default="runs", show_default=True)
@click.option("--campaign", default="adhoc", show_default=True)
def run_episode_cmd(
    snapshot: str,
    model: str,
    provider: str,
    model_name: str | None,
    start: str,
    window_id: str,
    seed: int,
    mode: str,
    steps: int,
    runs_dir: str,
    campaign: str,
) -> None:
    """Run a single episode and write artifacts + HTML report."""
    from tradingbench.data.store import load_snapshot
    from tradingbench.runner import run_episode

    snap = load_snapshot(snapshot)
    model_cfg = {
        "id": model,
        "provider": provider,
        "name": model_name or model,
        "temperature": 0.7,
        "max_tokens": 2000,
    }
    window = {"id": window_id, "start": start, "track": "historical"}
    result = run_episode(
        snap,
        model_cfg=model_cfg,
        window=window,
        seed=seed,
        mode=mode,
        steps=steps,
        runs_dir=Path(runs_dir),
        campaign=campaign,
    )
    click.echo(json.dumps({
        "run_id": result["run_id"],
        "path": result["path"],
        "total_return": result["metrics"]["total_return"],
        "final_nav": result["metrics"]["final_nav"],
        "report": str(Path(result["path"]) / "report.html"),
    }, indent=2))


@main.command("run-campaign")
@click.option("--config", required=True, type=click.Path(exists=True), help="Campaign YAML")
@click.option("--snapshot-dir", default=None, help="Override snapshot directory")
@click.option("--runs-dir", default=None, help="Override runs directory")
@click.option("--models", default=None, help="Comma-separated model ids to include")
@click.option("--max-episodes", default=None, type=int, help="Cap episodes (for smoke tests)")
def run_campaign_cmd(
    config: str,
    snapshot_dir: str | None,
    runs_dir: str | None,
    models: str | None,
    max_episodes: int | None,
) -> None:
    """Run a multi-episode campaign from YAML config."""
    from tradingbench.runner import run_campaign

    mf = models.split(",") if models else None
    result = run_campaign(
        config,
        snapshot_dir=snapshot_dir,
        runs_dir=runs_dir,
        models_filter=mf,
        max_episodes=max_episodes,
    )
    click.echo(f"Done. Report: {result['report']}")


@main.command("compare")
@click.option("--runs-dir", default="runs", show_default=True)
@click.option("--campaign", default="v1_baseline", show_default=True)
@click.option("--out", default=None, help="Output HTML path")
def compare_cmd(runs_dir: str, campaign: str, out: str | None) -> None:
    """Aggregate existing run metrics into a comparison report."""
    from tradingbench.eval.aggregate import aggregate_metrics
    from tradingbench.report.compare_report import write_compare_report

    root = Path(runs_dir)
    metrics = []
    for p in root.glob("*/metrics.json"):
        m = json.loads(p.read_text())
        # filter by campaign prefix in run folder name
        if campaign and not p.parent.name.startswith(campaign):
            continue
        metrics.append(m)
    if not metrics:
        # fall back to all
        for p in root.glob("*/metrics.json"):
            metrics.append(json.loads(p.read_text()))

    agg = aggregate_metrics(metrics)
    out_path = Path(out) if out else root / f"_campaign_{campaign}" / "compare_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_compare_report(out_path, campaign=campaign, aggregate=agg, episode_metrics=metrics)
    click.echo(f"Wrote {out_path} ({len(metrics)} episodes)")


if __name__ == "__main__":
    main()
