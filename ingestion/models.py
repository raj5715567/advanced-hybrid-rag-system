"""Shared data models for document ingestion."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ParsedDocument:
    """One extractable unit from an uploaded source (page, row, or file)."""

    text: str
    source_name: str
    source_type: str
    location: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

