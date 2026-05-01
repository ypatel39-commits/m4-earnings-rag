"""Chunk transcripts, embed with sentence-transformers, store in ChromaDB."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_TOKENS,
    EMBED_MODEL,
    RANDOM_STATE,
    TOP_K,
)
from .transcripts import load_cached, parse_header

# Approx 1 token ≈ 4 chars (English). Avoids tiktoken dep.
CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    chunk_id: str
    text: str
    ticker: str
    filing_date: str
    accession: str
    source_url: str
    paragraph_idx: int


def _split_paragraphs(body: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
    return paras


def chunk_text(text: str, *, chunk_tokens: int = CHUNK_TOKENS,
               overlap: int = CHUNK_OVERLAP) -> list[tuple[int, str]]:
    """Sliding-window chunker on chars (token-approx). Returns (paragraph_idx, chunk)."""
    chunk_chars = chunk_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap * CHARS_PER_TOKEN
    paras = _split_paragraphs(text)
    if not paras:
        return []

    out: list[tuple[int, str]] = []
    buf = ""
    buf_start_para = 0
    for i, p in enumerate(paras):
        if not buf:
            buf_start_para = i
        if len(buf) + len(p) + 2 <= chunk_chars:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            if buf:
                out.append((buf_start_para, buf))
            # Start next chunk with overlap from tail of previous
            tail = buf[-overlap_chars:] if overlap_chars and buf else ""
            buf = (tail + "\n\n" + p) if tail else p
            buf_start_para = i
    if buf:
        out.append((buf_start_para, buf))
    return out


def _strip_header(full: str) -> tuple[dict, str]:
    meta = parse_header(full)
    # Header ends at first blank line
    parts = full.split("\n\n", 1)
    body = parts[1] if len(parts) == 2 else full
    return meta, body


def build_chunks_from_disk() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path, _ in load_cached():
        full = path.read_text(encoding="utf-8")
        meta, body = _strip_header(full)
        ticker = meta.get("ticker", path.stem.split("_")[0])
        filing_date = meta.get("filing_date", "")
        accession = meta.get("accession", "")
        source_url = meta.get("source_url", "")
        for para_idx, ctext in chunk_text(body):
            cid = f"{ticker}_{filing_date}_{accession}_{para_idx}"
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=ctext,
                    ticker=ticker,
                    filing_date=filing_date,
                    accession=accession,
                    source_url=source_url,
                    paragraph_idx=para_idx,
                )
            )
    return chunks


def get_embedder():
    """Lazy import sentence-transformers to keep test imports cheap."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def get_chroma_collection(persist_dir: Path | None = None,
                          collection: str = CHROMA_COLLECTION):
    """Lazy import chromadb. Returns persistent collection."""
    import chromadb
    persist_dir = persist_dir or CHROMA_DIR
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(
        name=collection,
        metadata={"hnsw:space": "cosine", "random_state": str(RANDOM_STATE)},
    )


def index_chunks(chunks: Iterable[Chunk], *, batch_size: int = 64) -> int:
    """Embed and upsert chunks into ChromaDB. Returns count indexed."""
    chunks = list(chunks)
    if not chunks:
        return 0
    embedder = get_embedder()
    coll = get_chroma_collection()
    n = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        embs = embedder.encode(texts, show_progress_bar=False,
                               normalize_embeddings=True).tolist()
        coll.upsert(
            ids=[c.chunk_id for c in batch],
            documents=texts,
            embeddings=embs,
            metadatas=[
                {
                    "ticker": c.ticker,
                    "filing_date": c.filing_date,
                    "accession": c.accession,
                    "source_url": c.source_url,
                    "paragraph_idx": c.paragraph_idx,
                }
                for c in batch
            ],
        )
        n += len(batch)
    return n


def query(text: str, *, top_k: int = TOP_K) -> list[dict]:
    """Embed `text`, retrieve top_k chunks. Returns hit dicts."""
    embedder = get_embedder()
    coll = get_chroma_collection()
    q_emb = embedder.encode(
        [text], show_progress_bar=False, normalize_embeddings=True
    ).tolist()
    res = coll.query(query_embeddings=q_emb, n_results=top_k)
    hits: list[dict] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        hits.append(
            {
                "chunk_id": cid,
                "text": doc,
                "metadata": meta or {},
                "distance": float(dist) if dist is not None else None,
            }
        )
    return hits


def collection_count() -> int:
    return get_chroma_collection().count()
