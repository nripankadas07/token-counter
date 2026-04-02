"""CLI interface for token-counter."""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.table import Table

from token_counter.main import SUPPORTED_MODELS, TokenCounter

console = Console()


@click.group()
@click.version_option(package_name="token-counter")
def main() -> None:
    """Count tokens for multiple LLM tokenizers."""


@main.command()
@click.argument("text", required=False)
@click.option(
    "-m",
    "--model",
    multiple=True,
    help="Model(s) to count for. Can be repeated. Default: all models.",
)
@click.option(
    "-f", "--file", "filepath", type=click.Path(exists=True), help="Read text from file."
)
@click.option("--cost/--no-cost", default=True, help="Show cost estimates.")
def count(
    text: str | None,
    model: tuple[str, ...],
    filepath: str | None,
    cost: bool,
) -> None:
    """Count tokens in text or a file across one or more models."""
    # Resolve input text
    if filepath:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    elif text is None:
        if not sys.stdin.isatty():
            text = sys.stdin.read()
        else:
            console.print("[red]Error:[/red] Provide text as argument, --file, or pipe via stdin.")
            raise SystemExit(1)

    models = list(model) if model else None

    # Validate models
    if models:
        for m in models:
            if m not in SUPPORTED_MODELS:
                console.print(f"[red]Error:[/red] Unknown model '{m}'.")
                console.print(f"Supported: {', '.join(SUPPORTED_MODELS)}")
                raise SystemExit(1)

    counter = TokenCounter()
    results = counter.count_multi(text, models)

    # Build table
    table = Table(title="Token Count Results", show_lines=False)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Provider", style="dim")
    table.add_column("Encoding", style="dim")
    table.add_column("Tokens", justify="right", style="green bold")
    if cost:
        table.add_column("Input Cost", justify="right", style="yellow")
        table.add_column("Output Cost", justify="right", style="yellow")

    for r in results:
        row = [r.model, r.provider, r.encoding_name, f"{r.token_count:,}"]
        if cost:
            row.append(f"${r.input_cost:.6f}" if r.input_cost else "free")
            row.append(f"${r.output_cost:.6f}" if r.output_cost else "free")
        table.add_row(*row)

    console.print()
    console.print(f"[dim]Text length:[/dim] {len(text):,} characters")
    console.print(f"[dim]Whitespace estimate:[/dim] ~{counter.estimate_whitespace(text):,} tokens")
    console.print()
    console.print(table)


@main.command("models")
def list_models() -> None:
    """List all supported models."""
    table = Table(title="Supported Models")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="dim")

    from token_counter.main import MODEL_REGISTRY

    for name in SUPPORTED_MODELS:
        info = MODEL_REGISTRY[name]
        table.add_row(name, info["provider"])

    console.print(table)


if __name__ == "__main__":
    main()
