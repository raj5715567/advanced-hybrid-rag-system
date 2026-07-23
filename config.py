"""Central configuration for the Advanced RAG application."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return value


@dataclass(frozen=True)
class Settings:
    """Application settings read once from environment variables."""

    uploads_dir: Path = PROJECT_ROOT / "data" / "uploads"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    embeddings_dir: Path = PROJECT_ROOT / "data" / "embeddings"
    llm_provider: str = os.getenv("RAG_LLM_PROVIDER", "gemini").lower()
    ollama_model: str = os.getenv("RAG_OLLAMA_MODEL", "llama3.2")
    openai_model: str = os.getenv("RAG_OPENAI_MODEL", "gpt-4o-mini")
    gemini_model: str = os.getenv("RAG_GEMINI_MODEL", "gemini-3.6-flash")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = _positive_int("RAG_CHUNK_SIZE", 800)
    chunk_overlap: int = _positive_int("RAG_CHUNK_OVERLAP", 120)
    dense_top_k: int = _positive_int("RAG_DENSE_TOP_K", 8)
    bm25_top_k: int = _positive_int("RAG_BM25_TOP_K", 8)
    rerank_candidates: int = _positive_int("RAG_RERANK_CANDIDATES", 10)
    reranker_model: str = os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    final_top_k: int = _positive_int("RAG_FINAL_TOP_K", 5)

    def ensure_directories(self) -> None:
        for directory in (self.uploads_dir, self.processed_dir, self.embeddings_dir):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
