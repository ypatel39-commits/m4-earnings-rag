# STATE — M4 Earnings RAG

## Status

`v0.1.0` — first working build. Local-only stack, no API keys.

## What works

- SEC EDGAR fetch (`transcripts.py`): polite User-Agent, 0.15s rate-limit,
  hardcoded CIKs for the 10 tickers, MD&A regex slice with mid-doc fallback.
- Chunker (`index.py`): 500-token / 50-token-overlap sliding window over
  paragraph splits; deterministic chunk IDs `TICKER_DATE_ACCESSION_PARA`.
- Embedding: `all-MiniLM-L6-v2` via `sentence-transformers`,
  `normalize_embeddings=True` so cosine in Chroma works correctly.
- Vector store: `chromadb.PersistentClient` at `data/chroma/`, cosine HNSW.
- RAG (`rag.py`): top-5 retrieve, formatted `[n]`-citation context block,
  `qwen2.5:7b` via Ollama HTTP API, `temperature=0.2 seed=42`.
- Streamlit chat UI: question → answer → collapsible citation cards with
  SEC filing URL.
- 9 pytest tests, all offline. CI workflow already in `.github/workflows/`.

## What's deliberately out of scope (v0.1)

- True earnings-call **transcripts** (Motley Fool / company IR) require
  scraping copyrighted material — replaced with 10-Q MD&A which is public
  domain and a strong proxy ("forward-looking commentary in own words").
- No reranker. Pure vector similarity is good enough for top-5 over MD&A.
- No streaming generation in UI yet — Ollama call is blocking. Roadmap.
- No eval harness. Manual sample Q&A in `docs/sample-qa.md`.

## Known limitations

- MD&A regex assumes US 10-Q format ("Item 2. Management's Discussion...").
  Foreign-private-issuer 6-K / 20-F not supported.
- Token approximation = 4 chars/token. Fine for English MD&A.
- First run downloads MiniLM weights (~90 MB) and `qwen2.5:7b` (~4.7 GB).

## Reproducibility

- `RANDOM_STATE = 42` (used for Ollama `seed` and recorded in Chroma metadata).
- Hardcoded CIKs prevent silent ticker drift.

## Next steps

1. Add an eval set: 20 hand-written Q&A pairs with gold-standard citations,
   measure top-5 hit rate.
2. Stream Ollama tokens to Streamlit.
3. Add quarter-over-quarter compare mode ("How did AAPL margins change Q-over-Q?").
