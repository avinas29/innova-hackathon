"""Command-line interface.

    veritas research "topic"      run a full research + verification pass
    veritas verify "claim"        verify a single claim
    veritas eval                  benchmark against the single-LLM baseline
    veritas calibrate             fit the confidence calibrator
    veritas serve                 start the API
    veritas doctor                check configuration and connectivity
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from veritas.config import env_summary, get_settings
from veritas.logging import configure_logging
from veritas.schemas import RunEvent, Verdict

app = typer.Typer(
    name="veritas",
    help="Autonomous multi-agent research and fact-verification.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_VERDICT_STYLE = {
    Verdict.SUPPORTED: "bold green",
    Verdict.REFUTED: "bold red",
    Verdict.NEI: "bold yellow",
}


def _setup(verbose: bool) -> None:
    settings = get_settings()
    configure_logging("DEBUG" if verbose else settings.log_level, settings.log_json)


@app.command()
def research(
    topic: str = typer.Argument(..., help="Research topic or question"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write full JSON report here"),
    markdown: Path | None = typer.Option(None, "--markdown", "-m", help="Write the report markdown here"),
    max_claims: int | None = typer.Option(None, "--max-claims", help="Cap on claims to verify"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a full research and verification pass."""
    _setup(verbose)
    settings = get_settings()
    if max_claims:
        settings.max_claims = max_claims

    console.print(Panel.fit(f"[bold]{topic}[/bold]", title="VERITAS", border_style="cyan"))
    console.print(f"[dim]provider={settings.resolved_provider}  "
                  f"entailment={settings.entailment_backend}[/dim]\n")

    from veritas.graph.build import run_research

    def sink(event: RunEvent) -> None:
        if event.node in {"verifier", "planner", "researcher", "report", "contradiction"}:
            console.print(f"[dim]{event.node:<14}[/dim] {event.message}")

    with console.status("[cyan]researching…", spinner="dots"):
        report = asyncio.run(run_research(topic, event_sink=sink))

    console.print()
    _print_report(report)

    if output:
        output.write_text(json.dumps(report.model_dump(mode="json"), indent=2, default=str))
        console.print(f"\n[green]JSON written to[/green] {output}")
    if markdown:
        markdown.write_text(report.body_markdown)
        console.print(f"[green]Markdown written to[/green] {markdown}")

    if report.status.value != "COMPLETED":
        raise typer.Exit(code=1)


def _print_report(report) -> None:
    metrics = report.metrics

    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_row("Status", report.status.value)
    summary.add_row("Claims extracted", str(metrics.total_claims))
    summary.add_row("Verified", str(metrics.checkworthy_claims))
    summary.add_row(
        "Verdicts",
        f"[green]{metrics.supported} supported[/green]  "
        f"[red]{metrics.refuted} refuted[/red]  "
        f"[yellow]{metrics.nei} uncertain[/yellow]",
    )
    summary.add_row("Retracted", str(metrics.retracted))
    summary.add_row("Mean confidence", f"{metrics.mean_confidence:.3f}")
    summary.add_row("Sources / domains", f"{metrics.unique_sources} / {metrics.unique_domains}")
    summary.add_row(
        "Evidence / independent clusters",
        f"{metrics.evidence_items} / {metrics.independent_clusters}",
    )
    summary.add_row("Contradictions", str(metrics.contradictions))
    summary.add_row("Duration", f"{metrics.duration_seconds:.1f}s")
    summary.add_row("Tokens", f"{metrics.tokens.total:,} ({metrics.tokens.calls} calls)")
    console.print(Panel(summary, title="Run metrics", border_style="cyan"))

    verified = [c for c in report.claims if c.evidence_ids or c.verdict is not Verdict.NEI]
    if verified:
        table = Table(title="Claim verdicts", show_lines=False, header_style="bold")
        table.add_column("Verdict", width=10)
        table.add_column("Conf", width=6, justify="right")
        table.add_column("Src", width=4, justify="right")
        table.add_column("Claim", overflow="fold")

        for claim in sorted(verified, key=lambda c: -c.confidence):
            text = claim.verify_text
            if claim.retracted:
                text = f"[strike]{text}[/strike]"
            table.add_row(
                f"[{_VERDICT_STYLE[claim.verdict]}]{claim.verdict.value}[/]",
                f"{claim.confidence:.2f}",
                str(len(claim.cluster_ids)),
                text[:130],
            )
        console.print(table)

    if report.contradictions:
        console.print("\n[bold yellow]Source conflicts[/bold yellow]")
        for conflict in report.contradictions[:10]:
            console.print(
                f"  [yellow]•[/yellow] {conflict.domain_a} vs {conflict.domain_b} "
                f"({conflict.score:.2f}): {conflict.explanation[:110]}"
            )

    if report.warnings:
        console.print("\n[bold]Warnings[/bold]")
        for warning in report.warnings[:8]:
            console.print(f"  [dim]•[/dim] {warning[:150]}")

    if report.body_markdown:
        console.print("\n")
        console.print(Panel(Markdown(report.body_markdown[:6000]), title="Report",
                            border_style="green"))


@app.command()
def verify(
    claim: str = typer.Argument(..., help="A single claim to verify"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Verify one claim without a full research run."""
    _setup(verbose)

    from veritas.graph.nodes import verify_single_claim
    from veritas.state import build_context

    async def main() -> None:
        context = await build_context(get_settings())
        try:
            with console.status("[cyan]verifying…", spinner="dots"):
                result = await verify_single_claim(context, claim, run_id="cli")
        finally:
            await context.aclose()

        verified = result["verified_claims"][0]
        evidence = result.get("evidence", [])
        clusters = result.get("clusters", [])

        console.print(
            Panel.fit(
                f"[{_VERDICT_STYLE[verified.verdict]}]{verified.verdict.value}[/] "
                f"— confidence [bold]{verified.confidence:.3f}[/bold]",
                title=claim[:90],
                border_style="cyan",
            )
        )
        console.print(f"\n[dim]{verified.rationale[:600]}[/dim]\n")

        if verified.minority_report:
            console.print(f"[yellow]Minority report:[/yellow] {verified.minority_report[:300]}\n")

        table = Table(title=f"Evidence ({len(evidence)} items → {len(clusters)} independent)",
                      header_style="bold")
        table.add_column("Stance", width=9)
        table.add_column("Score", width=6, justify="right")
        table.add_column("Domain", width=24)
        table.add_column("Dup", width=4)
        table.add_column("Snippet", overflow="ellipsis")

        for item in sorted(evidence, key=lambda e: -e.entailment_score)[:12]:
            colour = {"SUPPORTS": "green", "REFUTES": "red", "NEUTRAL": "dim"}[item.stance.value]
            table.add_row(
                f"[{colour}]{item.stance.value}[/]",
                f"{item.entailment_score:.2f}",
                item.domain[:24],
                "yes" if item.is_derivative else "",
                item.snippet[:100].replace("\n", " "),
            )
        console.print(table)

        features = Table(title="Confidence features", header_style="bold", box=None)
        features.add_column("Feature", width=16)
        features.add_column("Value", justify="right")
        for name, value in verified.features.model_dump().items():
            features.add_row(name, f"{value:.3f}")
        console.print(features)

    asyncio.run(main())


@app.command("eval")
def eval_command(
    dataset: str = typer.Option("builtin", "--dataset", "-d", help="'builtin' or a JSONL path"),
    limit: int = typer.Option(24, "--limit", "-n", help="Number of claims to evaluate"),
    no_baseline: bool = typer.Option(False, "--no-baseline", help="Skip the single-LLM control"),
    output: Path = typer.Option(Path("eval_results.json"), "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Benchmark the pipeline against a single-LLM baseline."""
    _setup(verbose)

    from veritas.eval.run import run_evaluation

    with console.status("[cyan]evaluating…", spinner="dots"):
        payload = asyncio.run(
            run_evaluation(
                dataset=dataset,
                limit=limit,
                include_baseline=not no_baseline,
                output_path=output,
            )
        )

    console.print(Panel.fit(
        f"[bold]{payload['dataset']}[/bold]\n"
        f"{payload['n']} claims · provider={payload['provider']} · "
        f"{payload['duration_seconds']}s",
        title="Evaluation", border_style="cyan",
    ))
    console.print()
    console.print(Markdown(payload["comparison_table"]))

    headline = payload["verdict"]
    if headline.get("comparable"):
        colour = "green" if headline["better_calibration"] else "yellow"
        console.print(f"\n[{colour}]{headline['summary']}[/{colour}]")

    console.print(f"\n[dim]Full results written to {output}[/dim]")
    console.print("[dim]Fit the calibrator on this run with: veritas calibrate[/dim]")


@app.command()
def calibrate(
    eval_file: Path = typer.Option(Path("eval_results.json"), "--from", "-f"),
    output: Path = typer.Option(Path("calibration.json"), "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fit the confidence calibrator from an evaluation run."""
    _setup(verbose)

    if not eval_file.exists():
        console.print(f"[red]No evaluation file at {eval_file}.[/red] Run `veritas eval` first.")
        raise typer.Exit(code=1)

    from veritas.eval.run import calibrate_from_eval

    payload = json.loads(eval_file.read_text())
    try:
        report = asyncio.run(calibrate_from_eval(payload, output))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    table = Table(title="Calibration", header_style="bold")
    table.add_column("Metric")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_row("ECE", f"{report['ece_before']:.4f}", f"{report['ece_after']:.4f}")
    table.add_row("Brier", f"{report['brier_before']:.4f}", f"{report['brier_after']:.4f}")
    console.print(table)
    console.print(f"\n[green]Calibration written to {output}[/green] — picked up automatically.")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the API server."""
    import uvicorn

    configure_logging(get_settings().log_level, get_settings().log_json)
    console.print(Panel.fit(
        f"[bold cyan]VERITAS API[/bold cyan]\nhttp://{host}:{port}/docs",
        border_style="cyan",
    ))
    uvicorn.run("veritas.api.app:app", host=host, port=port, reload=reload, log_config=None)


@app.command()
def doctor() -> None:
    """Check configuration and provider connectivity."""
    configure_logging("WARNING")
    summary = env_summary()

    table = Table(title="Configuration", header_style="bold", box=None)
    table.add_column("Setting", width=22)
    table.add_column("Value")
    for key, value in summary.items():
        table.add_row(key, str(value))
    console.print(table)

    problems: list[str] = []
    if summary["llm_provider"] == "fake":
        problems.append(
            "No model API key found — running the deterministic OFFLINE provider. "
            "Results will be heuristic, not model output. Set OPENAI_API_KEY, "
            "ANTHROPIC_API_KEY, or GEMINI_API_KEY (free: "
            "https://aistudio.google.com/apikey)."
        )

    settings = get_settings()
    if summary["llm_provider"] == "gemini":
        limits = settings.effective_rate_limits()
        # Calibrated against a measured run: 8 claims produced 72 calls, so
        # roughly 8 per claim plus fixed planning/synthesis/report overhead.
        # The fast model absorbs about 4 of every 5 (query generation and
        # per-evidence entailment dominate); the strong model handles planning,
        # adjudication, synthesis and the report.
        estimated_calls = 8 + settings.max_claims * 8
        fast_rpd = limits.get(settings.model_for("fast"), (0, 0))[1]
        strong_rpd = limits.get(settings.model_for("strong"), (0, 0))[1]

        if settings.profile != "free":
            problems.append(
                "Using Gemini without VERITAS_PROFILE=free. A default run spends "
                f"~{estimated_calls} requests; the free profile trims that to fit a "
                "free-tier daily allowance."
            )

        if strong_rpd:
            runs_per_day = min(
                fast_rpd // max(1, int(estimated_calls * 0.8)),
                strong_rpd // max(1, int(estimated_calls * 0.2)),
            )
            if runs_per_day < 3:
                problems.append(
                    f"Daily quota allows only ~{runs_per_day} run(s) at the current "
                    f"settings (~{estimated_calls} calls each). Lower "
                    "VERITAS_MAX_CLAIMS or set VERITAS_PROFILE=free."
                )
            else:
                console.print(
                    f"[dim]Estimated ~{estimated_calls} model calls per run "
                    f"(~{runs_per_day} runs/day within the free quota).[/dim]"
                )
    if not summary["search_providers"]:
        problems.append("No search providers available; retrieval will return nothing.")
    elif summary["search_providers"] == ["duckduckgo"]:
        problems.append(
            "Only the keyless DuckDuckGo fallback is configured. It is rate-limited "
            "and best-effort — set TAVILY_API_KEY, EXA_API_KEY or BRAVE_API_KEY for "
            "reliable retrieval."
        )

    calibration = Path("calibration.json")
    if not calibration.exists():
        problems.append(
            "No calibration.json — confidence uses prior weights. "
            "Run `veritas eval` then `veritas calibrate` for fitted scores."
        )

    console.print()
    if problems:
        for problem in problems:
            console.print(f"[yellow]![/yellow] {problem}")
    else:
        console.print("[green]✓ All checks passed.[/green]")

    console.print("\n[dim]Testing connectivity…[/dim]")
    asyncio.run(_connectivity_check())


async def _connectivity_check() -> None:
    from veritas.llm.client import LLMClient, user
    from veritas.tools.search import SearchClient

    settings = get_settings()

    llm = LLMClient(settings)
    try:
        result = await llm.chat([user("Reply with the single word: ok")], max_tokens=10)
        console.print(
            f"[green]✓[/green] model provider [bold]{llm.provider_name}[/bold] "
            f"responded ({len(result.text)} chars)"
        )
    except Exception as exc:
        console.print(f"[red]✗[/red] model provider failed: {str(exc)[:160]}")
    finally:
        await llm.aclose()

    search = SearchClient(settings)
    try:
        results = await search.search("test query", limit=2)
        if results:
            console.print(
                f"[green]✓[/green] search returned {len(results)} results "
                f"via [bold]{results[0].provider}[/bold]"
            )
        else:
            console.print("[yellow]![/yellow] search returned no results")
    except Exception as exc:
        console.print(f"[red]✗[/red] search failed: {str(exc)[:160]}")
    finally:
        await search.aclose()


if __name__ == "__main__":
    app()
