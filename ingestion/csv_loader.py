"""CSV extraction."""

import csv
from pathlib import Path

from ingestion.models import ParsedDocument


def load_csv(path: Path) -> list[ParsedDocument]:
    """Convert each CSV row into a readable, metadata-rich document."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(file, dialect=dialect)
        if not reader.fieldnames:
            return []

        documents = []
        for row_number, row in enumerate(reader, start=2):
            values = [f"{column}: {value.strip()}" for column, value in row.items() if value and value.strip()]
            if values:
                documents.append(
                    ParsedDocument(
                        text="\n".join(values),
                        source_name=path.name,
                        source_type="csv",
                        location=f"row {row_number}",
                        metadata={"row_number": row_number, "columns": reader.fieldnames},
                    )
                )
    return documents

