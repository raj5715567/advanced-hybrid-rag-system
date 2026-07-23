# Advanced RAG System

## Completion phase

The Streamlit application includes hybrid FAISS and BM25 retrieval with reciprocal-rank fusion, cross-encoder reranking, local multi-query expansion, context compression, short-term conversation memory, streamed answer display, multiple LLM providers, cited sources, and JSON chat export.

An educational, portfolio-ready Retrieval-Augmented Generation application built with Streamlit. The project grows from simple local document ingestion into hybrid retrieval (dense search + BM25), reciprocal-rank fusion, reranking, conversational memory, and cited answers.

## Current status

**Phase 5 — Basic cited RAG is complete.** After you build the FAISS index, ask a question in chat. The app retrieves relevant chunks, instructs the selected LLM to answer from those sources only, and displays the exact chunks used.

## Supported roadmap

1. Parse PDF, DOCX, TXT, and CSV documents.
2. Split documents into metadata-rich chunks.
3. Build embeddings and a persistent FAISS index.
4. Add basic cited RAG responses. ✅
5. Add BM25, reciprocal-rank fusion, and a cross-encoder reranker.
6. Add conversational memory, streaming, LLM provider selection, and chat export.

## Setup

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

For Gemini, set `RAG_LLM_PROVIDER=gemini`, `RAG_GEMINI_MODEL=gemini-3.6-flash`, and `GEMINI_API_KEY` in `.env`. For local LLMs, install and run [Ollama](https://ollama.com/), then pull your chosen model, for example `ollama pull llama3.2`. OpenAI remains optional: set `RAG_LLM_PROVIDER=openai` and `OPENAI_API_KEY` only if you choose it.

Never paste API keys into source code, screenshots, or chat. If a key is exposed, revoke it immediately and replace it in your untracked `.env` file.

## Embeddings

The default local model, `sentence-transformers/all-MiniLM-L6-v2`, downloads once when you build the index. The resulting FAISS vectors and source metadata are stored in `data/embeddings/`. The index is saved through Python byte-based I/O so it also works in Windows paths containing non-English characters. You can use a different compatible Sentence Transformer model by setting `RAG_EMBEDDING_MODEL` in `.env` before indexing.

## Ask questions

Choose **gemini** after setting `GEMINI_API_KEY` in your local `.env`, **ollama** for local answers after running `ollama pull llama3.2`, or **openai** after setting `OPENAI_API_KEY`. The application sends only retrieved chunks—not every uploaded document—to the selected LLM and displays those chunks as citations.

## Structure

```text
app.py          Streamlit interface
config.py       Configuration and storage paths
data/           Local uploads, processed documents, and indexes
ingestion/      Document parsers (Phase 2)
chunking/       Chunking logic (Phase 3)
embeddings/     Embedding and FAISS code (Phase 4)
retrieval/      Dense, BM25, hybrid, and reranking code (Phases 5–7)
llm/            Provider integrations and prompts
memory/         Conversation-memory code
tests/          Automated tests
```
