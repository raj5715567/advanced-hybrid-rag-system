"""File-type dispatch and persistence for parsed documents."""

import json
from pathlib import Path

from ingestion.csv_loader import load_csv
from ingestion.docx_loader import load_docx
from ingestion.models import ParsedDocument
from ingestion.pdf_loader import load_pdf
from ingestion.txt_loader import load_txt


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}


def parse_file(path: Path) -> list[ParsedDocument]:
    """Extract content from one supported local file."""
    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".txt": load_txt,
        ".csv": load_csv,
    }
    try:
        return loaders[path.suffix.lower()](path)
    except KeyError as error:
        raise ValueError(f"Unsupported file type: {path.suffix or 'no extension'}") from error


def parse_directory(directory: Path) -> tuple[list[ParsedDocument], list[str]]:
    """Parse all supported files, retaining errors for UI feedback."""
    documents: list[ParsedDocument] = []
    errors: list[str] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            documents.extend(parse_file(path))
        except Exception as error:  # Keep the rest of the knowledge base usable.
            errors.append(f"{path.name}: {error}")
    return documents, errors


def save_processed_documents(documents: list[ParsedDocument], destination: Path) -> Path:
    """Save extracted text and metadata for inspection and later chunking."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([document.to_dict() for document in documents], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination

