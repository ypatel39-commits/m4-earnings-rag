# Deploy Guide — M4 Earnings RAG

This app's local stack uses **Ollama** + **ChromaDB** + **sentence-transformers**.
**Streamlit Community Cloud cannot run Ollama** (no GPU, no daemon process,
no port 11434). To deploy a hosted demo without breaking local behavior, this
repo ships a **demo mode** flag that swaps the LLM call for a deterministic
mock response. Local Ollama path is unchanged.

---

## 1. The Ollama problem (and three options)

| Option | What you get | Effort | Cost |
| --- | --- | --- | --- |
| **A. Demo mode** *(recommended for portfolio)* | Hosted app shows real retrieval + citations, mock LLM answer. Banner reads "Demo mode — mock LLM". | 0 code changes (already wired). Set `M4_DEMO_MODE=1` in Streamlit Secrets. | Free. |
| **B. Hosted LLM API** (OpenAI / Anthropic) | Real synthesized answers. | ~10 lines in `rag.py` (`_ollama_generate` shim). | Pay per token. |
| **C. Self-hosted Ollama** (Render / Fly / RunPod) | Identical to local. | Significant — separate service, GPU recommended for qwen2.5:7b. | $20–$200/mo. |

This guide covers Option A. Option B notes are at the bottom.

## 2. Prerequisites

- GitHub account + this repo on `main` (already true).
- Streamlit Community Cloud account: <https://share.streamlit.io/>.
- A pre-built Chroma index committed to `data/chroma/` **or** a
  `scripts/build_index.py` invocation that runs at first launch (heavy —
  prefer pre-building locally and committing).

## 3. Quick deploy (Option A — demo mode, ~10 min)

1. **Build the index locally** so Streamlit can serve real retrieval:
   ```bash
   uv pip install -e .
   python scripts/build_index.py    # populates data/chroma/
   git add data/chroma && git commit -m "data: prebuilt chroma index for demo"
   git push origin main
   ```
   Note: `data/chroma/` may be ~50–200 MB. Confirm it's not in `.gitignore`
   before pushing.
2. Go to <https://share.streamlit.io/> → **New app**.
3. Repository: `ypatel39-commits/m4-earnings-rag`. Branch: `main`. Main file: `app.py`.
4. Click **Advanced settings → Secrets** and add:
   ```toml
   M4_DEMO_MODE = "1"
   ```
5. **Deploy.** First build takes 5–10 minutes (chromadb + sentence-transformers + torch
   are large).
6. Confirm the sidebar shows `Ollama OK (vdemo (mock LLM, no Ollama))` and the
   chat returns mock answers with real citations.

## 4. Files Streamlit Cloud reads

| File                       | Why                                                        |
| -------------------------- | ---------------------------------------------------------- |
| `requirements.txt`         | Pip install list (mirrors pyproject deps + `-e .`).        |
| `.streamlit/config.toml`   | Theme, server flags, telemetry off.                        |
| `app.py`                   | Entrypoint.                                                |
| `data/chroma/`             | Pre-built vector store (commit it, or build at first run). |
| Streamlit Secrets          | `M4_DEMO_MODE=1` to bypass Ollama.                         |

## 5. Environment variables

| Var              | Required for hosted? | Notes                                      |
| ---------------- | -------------------- | ------------------------------------------ |
| `M4_DEMO_MODE`   | **Yes** (set to `1`) | Skips Ollama HTTP call, returns mock LLM.  |
| `OLLAMA_HOST`    | No                   | Local default `http://localhost:11434`.    |
| `OPENAI_API_KEY` | No (Option B only)   | If you wire OpenAI as the LLM backend.     |
| `ANTHROPIC_API_KEY` | No (Option B only) | If you wire Claude as the LLM backend.     |

Set these via **App settings → Secrets** in the Streamlit dashboard. They are
read with `os.environ[...]` (already used by `_demo_mode_enabled()`).

## 6. Free-tier limits + expected resource usage

Streamlit Community Cloud free tier (~1 GB RAM, shared vCPU):

| Resource | Estimated usage | Risk |
| --- | --- | --- |
| RAM | 600–900 MB (sentence-transformers MiniLM + Chroma in-process) | **Tight.** May OOM if chunk count > ~5k. |
| CPU | Spike on each query (embed + ANN search + mock format) | Acceptable. |
| Storage | `data/chroma/` baked into repo: 50–200 MB | OK; well under repo size limits. |
| Cold start | 30–60 s (loading torch + MiniLM) | First user pays the price. |
| Sleep | Idle apps sleep after ~7 days of no traffic | Wakes on visit. |

**If you hit OOM:** switch the embedder to a smaller model (e.g.
`paraphrase-MiniLM-L3-v2`) or reduce indexed chunk count. Or move to Option B
(hosted LLM API) which lets you drop torch entirely if you also swap the
embedder for OpenAI's embeddings API.

## 7. Local Ollama still works

The demo-mode flag only activates when `M4_DEMO_MODE` is set to `1`/`true`/`yes`.
Locally (no env var) `app.py` and `rag.py` behave exactly as before — they
require `ollama serve` and `ollama pull qwen2.5:7b` and call `localhost:11434`.

Verify:
```bash
unset M4_DEMO_MODE
ollama serve &
streamlit run app.py    # uses real Ollama
```

```bash
M4_DEMO_MODE=1 streamlit run app.py    # mock LLM, no Ollama needed
```

## 8. Option B — wire OpenAI/Anthropic instead of Ollama

If you want real synthesized answers in the hosted app:

1. In `src/m4_earnings_rag/rag.py`, replace `_ollama_generate` with a hosted
   API call when an env flag like `M4_LLM_BACKEND=openai` or `=anthropic` is
   set. The contract is just: `prompt: str -> str`.
2. Add `openai>=1.0` (or `anthropic>=0.34`) to `requirements.txt`.
3. In Streamlit Secrets, set the API key:
   ```toml
   OPENAI_API_KEY = "sk-..."          # or
   ANTHROPIC_API_KEY = "sk-ant-..."
   M4_LLM_BACKEND = "openai"
   ```
4. **Do not** check the key into the repo — Streamlit Secrets handles it.

Cost estimate: ~$0.001–$0.01 per question with GPT-4o-mini or Claude Haiku.

## 9. Troubleshooting

- **`ModuleNotFoundError: m4_earnings_rag`** → confirm `-e .` is at the bottom
  of `requirements.txt`.
- **OOM on cold start** → MiniLM + Chroma + Streamlit barely fits in 1 GB.
  Try the L3 embedder or trim the index.
- **App says "Index not built yet"** → you didn't commit `data/chroma/` or it
  was gitignored. Either commit it or run `scripts/build_index.py` as a
  pre-deploy step in the Streamlit logs (slow).
- **`Ollama unreachable`** in hosted app → you forgot to set `M4_DEMO_MODE=1`
  in Secrets.

## 10. Updating the deployed app

Push to `main`. Auto-redeploy in ~1 minute.

---

Author: Yash Patel · Project M4.
