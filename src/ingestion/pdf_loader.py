from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz

from src.ingestion.text_cleaner import clean_text


@dataclass(frozen=True)
class PageDocument:
    """Single cleaned PDF page with lightweight metadata."""

    page_number: int
    section_title: str
    source: str
    text: str


class PdfLoader:
    """Read the financial policy PDF and infer basic section metadata."""

    def __init__(self, pdf_path: Path) -> None:
        self.pdf_path = pdf_path

    def load_pages(self) -> List[PageDocument]:
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        documents: List[PageDocument] = []
        doc = fitz.open(str(self.pdf_path))
        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                raw_text = page.get_text("text") or ""
                page_number = page_index + 1
                section_title = self._infer_section_title(raw_text, page_number)
                documents.append(
                    PageDocument(
                        page_number=page_number,
                        section_title=section_title,
                        source=f"Page {page_number}",
                        text=clean_text(raw_text),
                    )
                )
        finally:
            doc.close()
        return documents

    def _infer_section_title(self, text: str, page_number: int) -> str:
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        for line in lines[:8]:
            if self._looks_like_heading(line):
                return line.strip("_*")
        return f"Page {page_number}"

    def _looks_like_heading(self, line: str) -> bool:
        if len(line) < 8 or len(line) > 120:
            return False
        if line.lower().startswith("table "):
            return True
        if line.isupper() and len(line.split()) <= 12:
            return True
        words = line.split()
        title_case_words = sum(word[:1].isupper() for word in words if word)
        return title_case_words >= max(2, len(words) // 2)
