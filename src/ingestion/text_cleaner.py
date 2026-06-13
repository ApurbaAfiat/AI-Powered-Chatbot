from __future__ import annotations

import re
import unicodedata


_HEADER_PATTERNS = (
    re.compile(r"^\s*page\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*financial policy objectives and strategies statement\s*$", re.IGNORECASE),
    re.compile(r"^\s*budget paper no\.\s*\d+\s*$", re.IGNORECASE),
)


def normalize_text(text: str) -> str:
    """Normalize whitespace and Unicode punctuation."""

    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



def remove_headers_and_footers(text: str) -> str:
    """Drop repeated header/footer lines from PDF extraction output."""

    cleaned_lines = []
    for line in (text or "").splitlines():
        candidate = line.strip()
        if not candidate:
            cleaned_lines.append("")
            continue
        if any(pattern.match(candidate) for pattern in _HEADER_PATTERNS):
            continue
        cleaned_lines.append(candidate)
    return "\n".join(cleaned_lines)



def clean_text(text: str) -> str:
    """Apply the standard cleaning pipeline used during ingestion."""

    text = remove_headers_and_footers(text)
    text = normalize_text(text)
    return text
