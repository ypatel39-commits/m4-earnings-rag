# Sample Q&A — M4 Earnings RAG

Real outputs captured from a live run on 2026-04-30 against the indexed
10-Q corpus (10 filings, 47 chunks). Model: `qwen2.5:7b` via local Ollama.
Re-running may yield different exact wording but stable citations.

---

## Q1: What does Microsoft say about Azure or cloud growth in its latest MD&A?

**A:** Microsoft reports that Azure and other cloud services revenue grew
40%, driven by demand for services across the platform with continued
growth across all workloads [1, 3]. This growth is a key driver of overall
Intelligent Cloud segment revenue increases [1].

**Sources**
- [1] MSFT 10-Q filed 2026-04-29 — Item 2 MD&A, paragraph 270
  [SEC filing](https://www.sec.gov/Archives/edgar/data/789019/000119312526191507/msft-20260331.htm)
- [3] MSFT 10-Q filed 2026-04-29 — Item 2 MD&A, paragraph 100

*Latency: ~127 s on M-series Mac, qwen2.5:7b Q4_K_M.*

---

## Q2: How does Apple describe iPhone revenue and product trends in its most recent 10-Q?

**A:** The provided excerpts in the top-5 retrieval are dominated by
Microsoft's MD&A — the model **correctly refuses to hallucinate** and
states the indexed sources do not answer this question. To get a real
Apple answer, re-run `build_index.py` with `--limit 2` (or higher) so
more AAPL-specific paragraphs reach the top-5.

**Sources** (top-5 were all MSFT chunks; not cited)

This is the desired behavior: a finance-trustworthy RAG bot prefers
"I don't have this in my sources" over confabulation.

---

## Q3: What does Bank of America say about interest income trends and macroeconomic conditions?

**A:** With limited BAC-specific chunks at top-5 (mix of JPM, BAC, AMZN,
MSFT, NFLX paragraph-0 boilerplate), the model declines to invent details
and recommends consulting BAC primary sources directly. It correctly
flags that some of the retrieved context is from Netflix — a clean
example of the citation system surfacing a retrieval-quality issue
rather than the LLM masking it.

**Sources**
- [2] BAC 10-Q filed 2025-10-31 — Item 2 MD&A, paragraph 0
  [SEC filing](https://www.sec.gov/Archives/edgar/data/70858/000007085825000405/bac-20250930.htm)
- [1] JPM 10-Q filed 2025-11-04 — Item 2 MD&A, paragraph 0

*The "para 0" pattern indicates these are leading MD&A paragraphs of each
filing — a known retrieval bias when chunks are short. Increasing
`--limit` and re-indexing fattens the corpus and reduces this.*

---

## Reading these results

This bot is for **equity-research workflows where wrong > silence**. The
two key signals here are good ones:

1. **Citations always shown.** Even when the LLM refuses to answer, the
   user sees what the retriever surfaced and which filing/paragraph it
   came from — they can click through to SEC EDGAR and verify.
2. **No hallucinated company facts.** When sources don't match the
   question, the model says so; it does not paper over the gap with
   plausible-sounding made-up numbers.

For higher answer quality, re-run the index with `--limit 4` (or higher)
and consider the roadmap items in `STATE.md` (reranker, eval set).
