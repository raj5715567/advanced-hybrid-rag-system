"""Persistence helpers for prepared chunks."""

import json
from pathlib import Path

from chunking.models import TextChunk
from ingestion.models import ParsedDocument


def load_processed_documents(path: Path) -> list[ParsedDocument]:
    """Load Phase 2 extraction output into document objects."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ParsedDocument(**item) for item in data]


def save_chunks(chunks: list[TextChunk], path: Path) -> Path:
    """Save chunks as JSON for inspection and the embedding phase."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2), encoding="utf-8")
    return path

