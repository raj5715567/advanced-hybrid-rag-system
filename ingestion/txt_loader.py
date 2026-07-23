"""Plain-text extraction with encoding fallbacks."""

from pathlib import Path

from ingestion.models import ParsedDocument


def load_txt(path: Path) -> list[ParsedDocument]:
    """Return a single document from a UTF-8 or common Windows text file."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            text = path.read_text(encoding=encoding).strip()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode {path.name} as text.")

    if not text:
        return []
    return [ParsedDocument(text, path.name, "txt", "entire file", {"encoding": encoding})]

