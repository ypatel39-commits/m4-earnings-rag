"""One-time pipeline: fetch 10-Qs from SEC EDGAR, chunk, embed, index."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without pip install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import click

from m4_earnings_rag.config import TICKERS
from m4_earnings_rag.index import build_chunks_from_disk, collection_count, index_chunks
from m4_earnings_rag.transcripts import fetch_all, load_cached


@click.command()
@click.option("--tickers", "-t", default=",".join(TICKERS),
              help="Comma-separated list of tickers.")
@click.option("--limit", default=2, show_default=True,
              help="Recent 10-Q filings to fetch per ticker.")
@click.option("--skip-fetch", is_flag=True,
              help="Skip SEC fetch; only re-index existing transcripts.")
def main(tickers: str, limit: int, skip_fetch: bool) -> None:
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    click.echo(f"Tickers: {ticker_list}")

    if not skip_fetch:
        click.echo("Fetching 10-Qs from SEC EDGAR ...")
        filings = fetch_all(ticker_list, limit_per_ticker=limit)
        click.echo(f"Fetched/cached {len(filings)} filings.")
    else:
        click.echo("Skipping fetch.")

    cached = load_cached()
    click.echo(f"Transcripts on disk: {len(cached)}")

    click.echo("Chunking ...")
    chunks = build_chunks_from_disk()
    click.echo(f"Built {len(chunks)} chunks.")

    click.echo("Embedding + indexing into ChromaDB (this may take a minute) ...")
    n = index_chunks(chunks)
    click.echo(f"Indexed {n} chunks.")
    click.echo(f"Collection size: {collection_count()}")


if __name__ == "__main__":
    main()
