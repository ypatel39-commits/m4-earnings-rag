"""Streamlit chat UI for M4 Earnings RAG."""

from __future__ import annotations

import os

import streamlit as st

# Bridge Streamlit Cloud secrets -> env vars (Streamlit doesn't auto-export).
# Allows M4_DEMO_MODE = "1" in secrets.toml to flip the rag.py demo path.
try:
    for _k in ("M4_DEMO_MODE",):
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # st.secrets may not exist locally; harmless

from m4_earnings_rag.config import OLLAMA_HOST, OLLAMA_MODEL, TICKERS, TOP_K
from m4_earnings_rag.index import build_chunks_from_disk, collection_count, index_chunks
from m4_earnings_rag.rag import answer, check_ollama


def _demo_mode() -> bool:
    return os.environ.get("M4_DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}


st.set_page_config(
    page_title="M4 Earnings RAG", page_icon=None, layout="wide"
)

st.title("Earnings Call / 10-Q RAG Q&A")
st.caption(
    "Local stack: Ollama qwen2.5:7b + ChromaDB + sentence-transformers. "
    "Source: SEC EDGAR 10-Q MD&A."
)

with st.sidebar:
    st.subheader("Status")
    ok, info = check_ollama()
    if _demo_mode():
        st.info("🟦 Demo mode (mock LLM, no Ollama)")
    elif ok:
        st.success(f"Ollama OK (v{info})")
    else:
        st.error(info)
        st.markdown(
            "Start Ollama:\n```\nollama serve\nollama pull qwen2.5:7b\n```"
        )

    try:
        n_chunks = collection_count()
    except Exception as exc:
        n_chunks = 0
        st.warning(f"Index not built yet: {exc}")
    st.metric("Indexed chunks", n_chunks)

    # Streamlit Cloud first-run: empty chromadb. Build index from the in-repo
    # transcripts on click. Takes ~30-60s due to sentence-transformers cold load.
    if n_chunks == 0:
        if st.button("Build index from in-repo transcripts", type="primary"):
            with st.spinner("Embedding 10-Q transcripts (first run ~60s)..."):
                chunks = build_chunks_from_disk()
                indexed = index_chunks(chunks)
            st.success(f"Indexed {indexed} chunks across {len(chunks)} sources.")
            st.rerun()

    st.write(f"Model: `{OLLAMA_MODEL}`")
    st.write(f"Top-K: {TOP_K}")
    st.write(f"Tickers: {', '.join(TICKERS)}")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        for c in turn.get("citations", []):
            with st.expander(f"[{c['idx']}] {c['ticker']} 10-Q {c['filing_date']} para {c['paragraph_idx']}"):
                if c.get("source_url"):
                    st.markdown(f"[SEC filing]({c['source_url']})")
                st.write(c.get("snippet", ""))

prompt = st.chat_input("Ask about a company's MD&A: e.g. 'How did AAPL describe iPhone revenue trends?'")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            res = answer(prompt)
        st.markdown(res.answer)
        for c in res.citations:
            with st.expander(f"[{c.idx}] {c.ticker} 10-Q {c.filing_date} para {c.paragraph_idx}"):
                if c.source_url:
                    st.markdown(f"[SEC filing]({c.source_url})")
                st.write(c.snippet)
        st.session_state.history.append(
            {
                "role": "assistant",
                "content": res.answer,
                "citations": [c.to_dict() for c in res.citations],
            }
        )
