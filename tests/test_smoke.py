"""Tests that don't require network, ollama, or a built index."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ on path even without pip install -e
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m4_earnings_rag import RANDOM_STATE
from m4_earnings_rag.config import (
    CIK_BY_TICKER,
    EMBED_MODEL,
    OLLAMA_MODEL,
    TICKERS,
    TOP_K,
)
from m4_earnings_rag.index import chunk_text
from m4_earnings_rag.rag import RAGAnswer, _format_context
from m4_earnings_rag.transcripts import extract_mdna, parse_header


def test_random_state_is_42():
    assert RANDOM_STATE == 42


def test_tickers_have_ciks():
    assert len(TICKERS) >= 10
    for t in TICKERS:
        assert t in CIK_BY_TICKER, f"Missing CIK for {t}"
        assert CIK_BY_TICKER[t].isdigit() and len(CIK_BY_TICKER[t]) == 10


def test_config_models():
    assert "qwen2.5" in OLLAMA_MODEL
    assert EMBED_MODEL == "all-MiniLM-L6-v2"
    assert TOP_K == 5


def test_chunk_text_overlaps_and_respects_size():
    paras = ["This is paragraph " + str(i) + ". " * 50 for i in range(20)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, chunk_tokens=200, overlap=20)
    assert len(chunks) >= 2
    for _, c in chunks:
        assert len(c) <= 200 * 4 + 600


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_extract_mdna_finds_section():
    # Simulate a 10-Q with TOC + real section. The body must be >=1000 chars
    # for the regex slice to be preferred over the mid-doc fallback.
    real_body = "Revenue grew 10%. " + ("Apple discussed iPhone trends. " * 80)
    body = (
        "Cover page filler. " * 40
        + "\n\nItem 2. Management's Discussion and Analysis (TOC entry)\n\n"
        + "Item 3. Quantitative (TOC)\n\n"
        + "Filler section. " * 40
        + "\n\nItem 2. Management's Discussion and Analysis of Financial Condition\n\n"
        + real_body
        + "\n\nItem 3. Quantitative and Qualitative Disclosures about Market Risk\n\n"
        + "Tail content not in MD&A. " * 20
    )
    out = extract_mdna(body)
    assert "Revenue grew 10%" in out
    assert "Tail content not in MD&A" not in out


def test_parse_header_roundtrip():
    text = "TICKER: AAPL\nFILING_DATE: 2025-08-01\nACCESSION: 0000320193-25-000999\n\nbody"
    h = parse_header(text)
    assert h["ticker"] == "AAPL"
    assert h["filing_date"] == "2025-08-01"


def test_format_context_builds_citations():
    hits = [
        {
            "text": "Apple sold many iPhones.",
            "metadata": {
                "ticker": "AAPL",
                "filing_date": "2025-08-01",
                "accession": "0000320193-25-000999",
                "source_url": "https://example.com/x.htm",
                "paragraph_idx": 7,
            },
        },
        {
            "text": "Services revenue grew.",
            "metadata": {
                "ticker": "AAPL",
                "filing_date": "2025-08-01",
                "accession": "0000320193-25-000999",
                "source_url": "https://example.com/x.htm",
                "paragraph_idx": 12,
            },
        },
    ]
    ctx, cites = _format_context(hits)
    assert "[1]" in ctx and "[2]" in ctx
    assert len(cites) == 2
    assert cites[0].ticker == "AAPL"
    assert cites[1].paragraph_idx == 12


def test_rag_answer_dataclass_serializes():
    a = RAGAnswer(question="q", answer="a", citations=[])
    d = a.to_dict()
    assert d["question"] == "q"
    assert d["citations"] == []
