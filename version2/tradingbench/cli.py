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
@click.option(
    "--mask",
    default="bright",
    show_default=True,
    type=click.Choice(["bright", "stock_blind", "date_blind", "blinded"]),
    help="2×2 mask factorial (§9.1)",
)
@click.option(
    "--decision-mode",
    default="standard",
    show_default=True,
    type=click.Choice(["standard", "memory_only"]),
    help="standard or memory_only null-channel (§9.2a)",
)
@click.option(
    "--price-path",
    default="real",
    show_default=True,
    type=click.Choice(["real", "synthetic"]),
    help="real or block-bootstrap synthetic (§9.3)",
)
@click.option(
    "--mode",
    default=None,
    help="Legacy mode (standard|named_control|blind|synthetic); overrides --mask if set",
)
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
    mask: str,
    decision_mode: str,
    price_path: str,
    mode: str | None,
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
    kwargs: dict = {
        "model_cfg": model_cfg,
        "window": window,
        "seed": seed,
        "steps": steps,
        "runs_dir": Path(runs_dir),
        "campaign": campaign,
        "price_path": price_path,
    }
    if mode:
        kwargs["mode"] = mode
    else:
        kwargs["mask"] = mask
        kwargs["decision_mode"] = decision_mode
    result = run_episode(snap, **kwargs)
    click.echo(json.dumps({
        "run_id": result["run_id"],
        "path": result["path"],
        "total_return": result["metrics"]["total_return"],
        "selection_alpha": result["metrics"].get("selection_alpha"),
        "final_nav": result["metrics"]["final_nav"],
        "mask": result["metrics"].get("mask"),
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


@main.command("viz")
@click.option(
    "--run",
    "runs",
    multiple=True,
    type=click.Path(exists=True),
    help="Run directory (repeatable). Defaults to learn_* runs if present.",
)
@click.option("--runs-dir", default="runs", show_default=True, help="Search dir when --run omitted")
@click.option("--out", default="viz", show_default=True, help="Output directory for index.html + data.js")
@click.option("--open/--no-open", "open_browser", default=True, help="Open in browser")
@click.option("--serve/--no-serve", default=False, help="Serve on localhost (avoids file:// quirks)")
@click.option("--port", default=8765, show_default=True, type=int)
def viz_cmd(
    runs: tuple[str, ...],
    runs_dir: str,
    out: str,
    open_browser: bool,
    serve: bool,
    port: int,
) -> None:
    """Build an interactive learning visualization from episode run(s)."""
    import http.server
    import socketserver
    import threading
    import webbrowser

    from tradingbench.viz.export import export_run_bundle, write_data_js

    run_paths: list[Path] = [Path(r) for r in runs]
    if not run_paths:
        root = Path(runs_dir)
        # Prefer learn_* demos, then any completed run
        candidates = sorted(root.glob("learn_*/manifest.json"))
        if not candidates:
            candidates = sorted(root.glob("*/manifest.json"))
        # skip campaign aggregate folders
        run_paths = [c.parent for c in candidates if not c.parent.name.startswith("_")]
        if not run_paths:
            raise click.ClickException(
                f"No runs found under {root}. Run an episode first, or pass --run PATH."
            )
        # keep at most a few for the dropdown
        run_paths = run_paths[:6]

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure index.html is present
    src_index = Path(__file__).resolve().parent.parent / "viz" / "index.html"
    if not src_index.exists():
        # package-adjacent fallback
        src_index = Path(__file__).resolve().parents[1] / "viz" / "index.html"
    dest_index = out_dir / "index.html"
    if src_index.exists():
        dest_index.write_text(src_index.read_text())
    elif not dest_index.exists():
        raise click.ClickException(f"Missing viz template at {src_index}")

    catalog: dict = {}
    for rp in run_paths:
        bundle = export_run_bundle(rp, out_path=rp / "viz_bundle.json")
        catalog[bundle["run_id"]] = bundle
        click.echo(f"  bundled {bundle['run_id']} ({len(bundle['steps'])} steps)")

    # data.js exposes both catalog and default TB_DATA
    first = next(iter(catalog.values()))
    payload = (
        "window.TB_CATALOG = "
        + json.dumps(catalog, default=str)
        + ";\nwindow.TB_DATA = window.TB_CATALOG["
        + json.dumps(first["run_id"])
        + "];\n"
    )
    (out_dir / "data.js").write_text(payload)
    click.echo(f"Wrote {out_dir / 'index.html'} + data.js ({len(catalog)} run(s))")

    url = (out_dir / "index.html").resolve().as_uri()
    if serve:
        # serve out_dir so relative data.js loads cleanly
        handler = http.server.SimpleHTTPRequestHandler
        os_cwd = Path.cwd()

        class QuietHandler(handler):
            def log_message(self, format, *args):  # noqa: A003
                pass

        def _serve():
            import os

            os.chdir(out_dir)
            with socketserver.TCPServer(("127.0.0.1", port), QuietHandler) as httpd:
                httpd.serve_forever()

        t = threading.Thread(target=_serve, daemon=True)
        t.start()
        url = f"http://127.0.0.1:{port}/index.html"
        click.echo(f"Serving {url}  (Ctrl+C to stop)")
        if open_browser:
            webbrowser.open(url)
        try:
            t.join()
        except KeyboardInterrupt:
            click.echo("\nStopped.")
        finally:
            import os

            os.chdir(os_cwd)
    else:
        if open_browser:
            webbrowser.open(url)
        click.echo(f"Open: {url}")
        click.echo("Tip: use --serve if the browser blocks local data.js")


@main.command("forward-init")
@click.option("--out", default="forward/live_v1", show_default=True)
@click.option("--cash", default=1000.0, show_default=True, type=float)
@click.option(
    "--models",
    default="buy_and_hold",
    show_default=True,
    help="Comma-separated baseline ids to track live",
)
def forward_init_cmd(out: str, cash: float, models: str) -> None:
    """Phase 0: stand up the forward track (uncontaminated, N=1 market path)."""
    from tradingbench.forward import init_forward_portfolio

    path = init_forward_portfolio(
        out,
        starting_cash=cash,
        models=[m.strip() for m in models.split(",") if m.strip()],
    )
    click.echo(f"Forward portfolio ready at {path}")
    click.echo("Run weekly: tradingbench forward-step --portfolio {path} --snapshot SNAP")


@main.command("forward-step")
@click.option("--portfolio", required=True, type=click.Path(exists=True))
@click.option("--snapshot", required=True, type=click.Path(exists=True))
@click.option("--model", default="buy_and_hold", show_default=True)
@click.option("--as-of", default=None, help="Decision date (default: second-to-last bar)")
@click.option("--seed", default=1, show_default=True, type=int)
def forward_step_cmd(
    portfolio: str,
    snapshot: str,
    model: str,
    as_of: str | None,
    seed: int,
) -> None:
    """Append one weekly step to a forward-track portfolio."""
    from datetime import date as date_cls

    from tradingbench.forward import forward_step

    result = forward_step(
        portfolio,
        snapshot,
        model=model,
        as_of=date_cls.fromisoformat(as_of) if as_of else None,
        seed=seed,
    )
    click.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

