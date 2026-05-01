"""Fetch SEC EDGAR 10-Q filings and extract MD&A (Item 2) text.

Polite scraping (User-Agent + 0.15s rate limit). Caches plain-text MD&A
sections under data/transcripts/{TICKER}_{ACCESSION}_{DATE}.txt.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

from .config import (
    CIK_BY_TICKER,
    SEC_RATE_LIMIT_SEC,
    SEC_USER_AGENT,
    TICKERS,
    TRANSCRIPTS_DIR,
)


SEC_BASE = "https://data.sec.gov"
ARCHIVE_BASE = "https://www.sec.gov/Archives"


@dataclass
class Filing:
    ticker: str
    cik: str
    accession: str  # no dashes
    accession_dashed: str
    filing_date: str  # YYYY-MM-DD
    primary_document: str
    form: str
    url: str
    cached_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _headers() -> dict[str, str]:
    return {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def _sleep() -> None:
    time.sleep(SEC_RATE_LIMIT_SEC)


def _get(url: str, *, timeout: int = 30) -> requests.Response:
    _sleep()
    resp = requests.get(url, headers=_headers(), timeout=timeout)
    resp.raise_for_status()
    return resp


def list_recent_10q(ticker: str, *, limit: int = 2) -> list[Filing]:
    """Return up to `limit` most-recent 10-Q filings for a ticker."""
    cik = CIK_BY_TICKER.get(ticker.upper())
    if cik is None:
        raise KeyError(f"Unknown ticker: {ticker}")
    url = f"{SEC_BASE}/submissions/CIK{cik}.json"
    data = _get(url).json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accs = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    docs = recent.get("primaryDocument", [])
    out: list[Filing] = []
    for form, acc, date, doc in zip(forms, accs, dates, docs):
        if form != "10-Q":
            continue
        acc_nodash = acc.replace("-", "")
        out.append(
            Filing(
                ticker=ticker.upper(),
                cik=cik,
                accession=acc_nodash,
                accession_dashed=acc,
                filing_date=date,
                primary_document=doc,
                form=form,
                url=f"{ARCHIVE_BASE}/edgar/data/{int(cik)}/{acc_nodash}/{doc}",
            )
        )
        if len(out) >= limit:
            break
    return out


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r" ", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_MDA_START = re.compile(
    r"item\s*2\.?\s*management.?s\s+discussion\s+and\s+analysis",
    re.IGNORECASE,
)
_MDA_END = re.compile(
    r"item\s*3\.?\s*quantitative\s+and\s+qualitative\s+disclosures",
    re.IGNORECASE,
)


def extract_mdna(text: str) -> str:
    """Slice MD&A (Item 2) from a 10-Q's plain text.

    10-Q HTML typically has TWO matches for "Item 2. Management's Discussion":
    one in the TOC and one at the real section start. We prefer the LAST
    Item 2 occurrence (the real section) and the next Item 3 after it.
    Falls back to mid-doc if no clean section.
    """
    starts = list(_MDA_START.finditer(text))
    if starts:
        start = starts[-1].start()
        ends = [m for m in _MDA_END.finditer(text) if m.start() > start]
        end = ends[0].start() if ends else min(start + 80_000, len(text))
        section = text[start:end].strip()
        if len(section) >= 1000:  # sanity: real MD&A is huge
            return section
    # Fallback: take the middle 40k chars (10-Qs are huge; MD&A usually mid-doc)
    n = len(text)
    if n < 5000:
        return text
    mid = n // 2
    return text[max(0, mid - 20_000) : mid + 20_000].strip()


def fetch_and_cache(ticker: str, *, limit: int = 2,
                    out_dir: Path | None = None) -> list[Filing]:
    """Fetch latest 10-Qs for a ticker, extract MD&A, write to disk."""
    out_dir = out_dir or TRANSCRIPTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filings = list_recent_10q(ticker, limit=limit)
    for f in filings:
        fname = f"{f.ticker}_{f.filing_date}_{f.accession}.txt"
        path = out_dir / fname
        if path.exists() and path.stat().st_size > 1000:
            f.cached_path = str(path)
            continue
        try:
            html = _get(f.url).text
        except Exception as exc:  # pragma: no cover — network-dependent
            print(f"[WARN] fetch failed {f.ticker} {f.filing_date}: {exc}")
            continue
        text = _html_to_text(html)
        mdna = extract_mdna(text)
        header = (
            f"TICKER: {f.ticker}\nCIK: {f.cik}\nFORM: {f.form}\n"
            f"FILING_DATE: {f.filing_date}\nACCESSION: {f.accession_dashed}\n"
            f"SOURCE_URL: {f.url}\n\n"
        )
        path.write_text(header + mdna, encoding="utf-8")
        f.cached_path = str(path)
    return filings


def fetch_all(tickers: Iterable[str] | None = None, *,
              limit_per_ticker: int = 2) -> list[Filing]:
    """Fetch+cache for all tickers. Returns flat list of Filings."""
    tickers = list(tickers) if tickers else TICKERS
    all_filings: list[Filing] = []
    for t in tickers:
        try:
            all_filings.extend(fetch_and_cache(t, limit=limit_per_ticker))
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] {t}: {exc}")
    return all_filings


def load_cached() -> list[tuple[Path, dict]]:
    """Return (path, header_meta) for every cached transcript on disk."""
    if not TRANSCRIPTS_DIR.exists():
        return []
    out: list[tuple[Path, dict]] = []
    for p in sorted(TRANSCRIPTS_DIR.glob("*.txt")):
        meta = parse_header(p.read_text(encoding="utf-8"))
        out.append((p, meta))
    return out


def parse_header(text: str) -> dict:
    meta: dict = {}
    for line in text.splitlines()[:8]:
        if ":" not in line:
            break
        k, _, v = line.partition(":")
        meta[k.strip().lower()] = v.strip()
    return meta
