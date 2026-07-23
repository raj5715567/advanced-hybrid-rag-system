"""Streamlit entry point for the Advanced RAG System."""

import json
from pathlib import Path

import streamlit as st

from chunking import chunk_documents, load_processed_documents, save_chunks
from chunking.models import TextChunk
from config import settings
from embeddings import FaissVectorStore, embed_texts
from ingestion import parse_directory, save_processed_documents
from llm import SYSTEM_PROMPT, build_rag_prompt, generate_answer
from memory import format_conversation_history
from retrieval import compress_context, expand_query, reciprocal_rank_fusion, rerank_chunks, retrieve_hybrid


st.set_page_config(page_title="Advanced RAG", page_icon="📚", layout="wide")
settings.ensure_directories()


def initialise_session() -> None:
    defaults = {
        "messages": [],
        "uploaded_files": [],
        "parsed_document_count": 0,
        "chunk_count": 0,
        "index_chunk_count": 0,
        "llm_provider": settings.llm_provider,
        "dense_top_k": settings.dense_top_k,
        "bm25_top_k": settings.bm25_top_k,
        "rerank_candidates": getattr(settings, "rerank_candidates", 10),
        "final_top_k": settings.final_top_k,
        "use_multi_query": True,
        "use_context_compression": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def save_uploaded_files(files: list) -> None:
    """Persist uploaded files locally for the ingestion pipeline."""
    saved_files = []
    for uploaded_file in files:
        safe_name = Path(uploaded_file.name).name
        (settings.uploads_dir / safe_name).write_bytes(uploaded_file.getvalue())
        saved_files.append(safe_name)
    st.session_state.uploaded_files = saved_files


def render_sources(sources: list[dict]) -> None:
    """Show the exact chunks used to ground an answer."""
    with st.expander("Sources used"):
        for source in sources:
            st.markdown(f"**[{source['number']}] {source['source_name']} — {source['location']}**")
            st.caption(f"Cross-encoder score: {source['score']:.4f}")
            st.write(source["text"])


initialise_session()
processed_path = settings.processed_dir / "parsed_documents.json"
chunks_path = settings.processed_dir / "chunks.json"
index_path = settings.embeddings_dir / "chunks.faiss"

with st.sidebar:
    st.header("Knowledge base")
    uploaded_files = st.file_uploader(
        "Add documents", type=["pdf", "docx", "txt", "csv"], accept_multiple_files=True
    )
    if uploaded_files and st.button("Save documents", use_container_width=True):
        save_uploaded_files(uploaded_files)
        st.success(f"Saved {len(st.session_state.uploaded_files)} document(s).")

    if st.session_state.uploaded_files and st.button("Process documents", use_container_width=True):
        with st.spinner("Extracting text and metadata..."):
            parsed_documents, errors = parse_directory(settings.uploads_dir)
            save_processed_documents(parsed_documents, processed_path)
        st.session_state.parsed_document_count = len(parsed_documents)
        if parsed_documents:
            st.success(f"Extracted {len(parsed_documents)} document unit(s).")
        else:
            st.warning("No extractable text was found in the uploaded files.")
        for error in errors:
            st.error(error)

    if processed_path.exists() and st.button("Create chunks", use_container_width=True):
        with st.spinner("Splitting documents into retrieval-ready chunks..."):
            chunks = chunk_documents(
                load_processed_documents(processed_path), settings.chunk_size, settings.chunk_overlap
            )
            save_chunks(chunks, chunks_path)
        st.session_state.chunk_count = len(chunks)
        st.success(f"Created {len(chunks)} chunk(s) with source metadata.")

    if chunks_path.exists() and st.button("Build FAISS index", use_container_width=True):
        with st.spinner("Loading the embedding model and creating vectors..."):
            chunks = [TextChunk(**item) for item in json.loads(chunks_path.read_text(encoding="utf-8"))]
            embedding_model = getattr(settings, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            vectors = embed_texts([chunk.text for chunk in chunks], embedding_model)
            FaissVectorStore.from_chunks(chunks, vectors).save(settings.embeddings_dir)
        st.session_state.index_chunk_count = len(chunks)
        st.success(f"FAISS index built for {len(chunks)} chunk(s).")

    if st.session_state.uploaded_files:
        st.caption("Saved for indexing")
        for file_name in st.session_state.uploaded_files:
            st.write(f"• {file_name}")
    if st.session_state.parsed_document_count:
        st.caption(f"Parsed units: {st.session_state.parsed_document_count}")
    if st.session_state.chunk_count:
        st.caption(f"Retrieval-ready chunks: {st.session_state.chunk_count}")
    if st.session_state.index_chunk_count or index_path.exists():
        st.caption("FAISS index: ready")

    st.divider()
    st.header("Retrieval settings")
    st.selectbox("LLM provider", ["gemini", "ollama", "openai"], key="llm_provider")
    st.slider("Dense candidates", 1, 20, key="dense_top_k")
    st.slider("BM25 candidates", 1, 20, key="bm25_top_k")
    st.slider("Rerank candidates", 1, 20, key="rerank_candidates")
    st.slider("Final sources", 1, 10, key="final_top_k")
    st.checkbox("Use multi-query retrieval", key="use_multi_query")
    st.checkbox("Compress context before answering", key="use_context_compression")
    st.caption("FAISS and BM25 are fused, then a cross-encoder reranks the strongest candidates.")

    st.divider()
    st.header("Chat")
    st.download_button("Download chat", json.dumps(st.session_state.messages, ensure_ascii=False, indent=2), "rag-chat.json", "application/json", use_container_width=True)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if chunks_path.exists():
        with st.expander("Preview prepared chunks"):
            for chunk in json.loads(chunks_path.read_text(encoding="utf-8"))[:3]:
                st.caption(f"{chunk['source_name']} · {chunk['location']} · chunk {chunk['chunk_index'] + 1}")
                st.text(chunk["text"][:300] + ("…" if len(chunk["text"]) > 300 else ""))

st.title("📚 Advanced RAG System")
st.caption("Dense retrieval and cited answers — built incrementally.")

if not st.session_state.uploaded_files:
    st.info("Upload documents, process them, create chunks, and build the index before asking a question.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

question = st.chat_input("Ask your knowledge base a question")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not index_path.exists():
            answer, sources = "Build the FAISS index in the sidebar before asking a question.", []
            st.warning(answer)
        else:
            try:
                with st.spinner("Retrieving sources and generating an answer..."):
                    embedding_model = getattr(settings, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
                    queries = expand_query(question) if st.session_state.use_multi_query else [question]
                    candidate_lists = [
                        retrieve_hybrid(
                            query,
                            settings.embeddings_dir,
                            embedding_model,
                            st.session_state.dense_top_k,
                            st.session_state.bm25_top_k,
                            st.session_state.rerank_candidates,
                        )
                        for query in queries
                    ]
                    results = reciprocal_rank_fusion(candidate_lists, st.session_state.rerank_candidates)
                    if st.session_state.use_context_compression:
                        results = compress_context(question, results)
                    results = rerank_chunks(
                        question,
                        results,
                        st.session_state.final_top_k,
                        getattr(settings, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                    )
                    prompt = build_rag_prompt(question, results, format_conversation_history(st.session_state.messages))
                    provider = st.session_state.llm_provider
                    if provider == "gemini":
                        # ``getattr`` keeps a Streamlit hot reload safe when an
                        # older Settings instance is still held in memory.
                        model = getattr(settings, "gemini_model", "gemini-3.6-flash")
                    elif provider == "ollama":
                        model = settings.ollama_model
                    elif provider == "openai":
                        model = settings.openai_model
                    else:
                        raise ValueError(f"Unsupported LLM provider: {provider}")
                    answer = generate_answer(provider, SYSTEM_PROMPT, prompt, model)
                    sources = [
                        {
                            "number": number,
                            "source_name": chunk.source_name,
                            "location": chunk.location,
                            "text": chunk.text,
                            "score": score,
                        }
                        for number, (chunk, score) in enumerate(results, start=1)
                    ]
                st.write_stream(word + " " for word in answer.split())
                render_sources(sources)
            except Exception as error:
                answer, sources = f"I could not answer that question: {error}", []
                st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
