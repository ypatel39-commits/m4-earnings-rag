"""Shared configuration for paths, models, and tickers."""

from __future__ import annotations

from pathlib import Path

# Repo paths
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CHROMA_DIR = DATA_DIR / "chroma"

# Tickers we cover (CIK lookup happens in transcripts.py)
TICKERS: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "JPM",
    "BAC", "NVDA", "TSLA", "META", "NFLX",
]

# Hardcoded CIKs (10-digit zero-padded) — sourced from SEC EDGAR public lookup.
# Stable; embedded here so tests/build don't require a network ticker lookup.
CIK_BY_TICKER: dict[str, str] = {
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "GOOGL": "0001652044",
    "AMZN":  "0001018724",
    "JPM":   "0000019617",
    "BAC":   "0000070858",
    "NVDA":  "0001045810",
    "TSLA":  "0001318605",
    "META":  "0001326801",
    "NFLX":  "0001065280",
}

# SEC EDGAR
SEC_USER_AGENT = "M4 Earnings RAG yashpatel06050@gmail.com"
SEC_RATE_LIMIT_SEC = 0.15

# Models
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Chunking
CHUNK_TOKENS = 500
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 5

# Reproducibility
RANDOM_STATE = 42

# Chroma collection
CHROMA_COLLECTION = "earnings_chunks"
