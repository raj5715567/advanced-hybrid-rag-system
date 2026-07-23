"""Dependency-free, metadata-preserving recursive text splitting."""

from hashlib import sha256

from ingestion.models import ParsedDocument

from chunking.models import TextChunk


SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def _split_by_length(text: str, chunk_size: int) -> list[str]:
    """Use progressively smaller natural separators before a hard character split."""
    if len(text) <= chunk_size:
        return [text]

    for separator in SEPARATORS:
        parts = list(text) if separator == "" else text.split(separator)
        if len(parts) <= 1:
            continue
        chunks: list[str] = []
        current = ""
        joiner = "" if separator == "" else separator
        for part in parts:
            candidate = f"{current}{joiner if current else ''}{part}"
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(part) > chunk_size:
                    chunks.extend(_split_by_length(part, chunk_size))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current.strip())
        if chunks and all(len(chunk) <= chunk_size for chunk in chunks):
            return chunks

    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prefix each later chunk with a tail from the preceding chunk."""
    if not chunks or overlap == 0:
        return chunks
    overlapped = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        prefix = previous[-overlap:].strip()
        overlapped.append(f"{prefix}\n{current}" if prefix else current)
    return overlapped


def chunk_documents(
    documents: list[ParsedDocument], chunk_size: int, chunk_overlap: int
) -> list[TextChunk]:
    """Create bounded overlapping chunks while retaining source citations."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[TextChunk] = []
    for document_number, document in enumerate(documents):
        pieces = _add_overlap(_split_by_length(document.text.strip(), chunk_size), chunk_overlap)
        for chunk_index, text in enumerate(pieces):
            if not text:
                continue
            chunk_id = sha256(f"{document.source_name}:{document.location}:{document_number}:{chunk_index}".encode()).hexdigest()[:16]
            chunks.append(
                TextChunk(
                    id=chunk_id,
                    text=text,
                    source_name=document.source_name,
                    source_type=document.source_type,
                    location=document.location,
                    chunk_index=chunk_index,
                    metadata={**document.metadata, "document_index": document_number},
                )
            )
    return chunks

