from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, List, Sequence

from src.ingestion.pdf_loader import PageDocument


@dataclass(frozen=True)
class Chunk:
    """A retrieval chunk with the metadata required by the assignment."""

    chunk_id: str
    section: str
    source: str
    page_number: int
    text: str
    start_char: int
    end_char: int

    def to_metadata(self) -> dict:
        return asdict(self)


class Chunker:
    """Split page-level documents into overlapping character chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: Sequence[PageDocument]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for page in pages:
            chunks.extend(self._chunk_page(page))
        return chunks

    def _chunk_page(self, page: PageDocument) -> Iterable[Chunk]:
        text = page.text or ""
        if not text:
            return []

        start = 0
        chunk_index = 0
        text_length = len(text)
        while start < text_length:
            end = min(text_length, start + self.chunk_size)
            chunk_text = text[start:end].strip()
            if chunk_text:
                yield Chunk(
                    chunk_id=f"p{page.page_number:03d}_c{chunk_index:03d}",
                    section=page.section_title,
                    source=page.source,
                    page_number=page.page_number,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                )
                chunk_index += 1
            if end >= text_length:
                break
            start = max(0, end - self.chunk_overlap)
