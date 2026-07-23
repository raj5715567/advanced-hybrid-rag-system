"""PDF text extraction."""

from pathlib import Path

from pypdf import PdfReader

from ingestion.models import ParsedDocument


def load_pdf(path: Path) -> list[ParsedDocument]:
    """Return one document per non-empty PDF page."""
    reader = PdfReader(str(path))
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(
                ParsedDocument(
                    text=text,
                    source_name=path.name,
                    source_type="pdf",
                    location=f"page {page_number}",
                    metadata={"page_number": page_number},
                )
            )
    return documents

