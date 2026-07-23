"""Data model for text chunks used by retrieval."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TextChunk:
    """A retrievable text segment that retains its original source details."""

    id: str
    text: str
    source_name: str
    source_type: str
    location: str
    chunk_index: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

