"""Retrieval-augmented generation over indexed 10-Q chunks via local Ollama."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests

from .config import OLLAMA_HOST, OLLAMA_MODEL, TOP_K
from .index import query as retrieve


SYSTEM_PROMPT = (
    "You are an equity-research assistant answering questions about US public "
    "companies using ONLY the provided 10-Q MD&A excerpts. Cite each claim "
    "inline with bracket numbers like [1], [2] that map to the Sources block. "
    "If the excerpts do not contain the answer, say so plainly. Be concise; "
    "prefer numbers and dates from the source over generalities."
)


@dataclass
class Citation:
    idx: int
    ticker: str
    filing_date: str
    accession: str
    paragraph_idx: int
    source_url: str
    snippet: str

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "ticker": self.ticker,
            "filing_date": self.filing_date,
            "accession": self.accession,
            "paragraph_idx": self.paragraph_idx,
            "source_url": self.source_url,
            "snippet": self.snippet,
        }


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    model: str = OLLAMA_MODEL
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "model": self.model,
            "error": self.error,
        }


def check_ollama(host: str = OLLAMA_HOST, *, timeout: int = 3) -> tuple[bool, str]:
    """Verify Ollama is reachable. Returns (ok, message)."""
    try:
        r = requests.get(f"{host}/api/version", timeout=timeout)
        r.raise_for_status()
        return True, r.json().get("version", "unknown")
    except Exception as exc:
        return False, f"Ollama unreachable at {host}: {exc}"


def _format_context(hits: list[dict]) -> tuple[str, list[Citation]]:
    citations: list[Citation] = []
    blocks: list[str] = []
    for i, h in enumerate(hits, start=1):
        m = h.get("metadata", {}) or {}
        snippet = (h.get("text") or "")[:280].replace("\n", " ").strip()
        c = Citation(
            idx=i,
            ticker=str(m.get("ticker", "?")),
            filing_date=str(m.get("filing_date", "")),
            accession=str(m.get("accession", "")),
            paragraph_idx=int(m.get("paragraph_idx", 0) or 0),
            source_url=str(m.get("source_url", "")),
            snippet=snippet,
        )
        citations.append(c)
        blocks.append(
            f"[{i}] {c.ticker} 10-Q filed {c.filing_date} (para {c.paragraph_idx}):\n"
            f"{h.get('text', '')}"
        )
    return "\n\n".join(blocks), citations


def _ollama_generate(prompt: str, *, host: str = OLLAMA_HOST,
                     model: str = OLLAMA_MODEL, timeout: int = 600) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "seed": 42},
    }
    r = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def answer(question: str, *, top_k: int = TOP_K,
           host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL) -> RAGAnswer:
    """End-to-end: retrieve, prompt, call Ollama, return answer + citations."""
    hits = retrieve(question, top_k=top_k)
    context, citations = _format_context(hits)
    if not hits:
        return RAGAnswer(
            question=question,
            answer="No indexed transcripts match this question. Run scripts/build_index.py first.",
            citations=[],
            model=model,
            error="empty_index",
        )

    prompt = (
        f"{SYSTEM_PROMPT}\n\nSources:\n{context}\n\n"
        f"Question: {question}\n\nAnswer (with [n] citations):"
    )
    ok, info = check_ollama(host=host)
    if not ok:
        return RAGAnswer(
            question=question,
            answer=(
                "Ollama is not reachable. Start it with `ollama serve` and "
                f"ensure model `{model}` is pulled. Citations below are still "
                "the retrieved sources."
            ),
            citations=citations,
            model=model,
            error=info,
        )
    try:
        text = _ollama_generate(prompt, host=host, model=model)
    except Exception as exc:
        return RAGAnswer(
            question=question,
            answer=f"Ollama call failed: {exc}",
            citations=citations,
            model=model,
            error=str(exc),
        )
    return RAGAnswer(
        question=question, answer=text, citations=citations, model=model
    )


def format_for_terminal(a: RAGAnswer) -> str:
    lines = [f"Q: {a.question}", "", f"A: {a.answer}", "", "Sources:"]
    for c in a.citations:
        lines.append(
            f"  [{c.idx}] {c.ticker} 10-Q {c.filing_date} para {c.paragraph_idx}"
        )
        if c.source_url:
            lines.append(f"      {c.source_url}")
    if a.error:
        lines += ["", f"(note: {a.error})"]
    return "\n".join(lines)
