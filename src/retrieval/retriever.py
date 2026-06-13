from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.embeddings.vector_store import FaissVectorStore, SearchHit


@dataclass(frozen=True)
class RetrievedChunk:
    """Chunk plus a compact display snippet for answer generation."""

    hit: SearchHit

    @property
    def snippet(self) -> str:
        text = self.hit.text.replace("\n", " ").strip()
        return text[:260] + ("…" if len(text) > 260 else "")


class Retriever:
    """Semantic retriever built on top of the FAISS vector store."""

    def __init__(self, vector_store: FaissVectorStore, default_top_k: int = 5) -> None:
        self.vector_store = vector_store
        self.default_top_k = default_top_k

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
        hits = self.vector_store.search(query, top_k=top_k or self.default_top_k)
        return [RetrievedChunk(hit=hit) for hit in hits]

    def build_context(self, results: List[RetrievedChunk]) -> str:
        lines: List[str] = []
        for index, result in enumerate(results, start=1):
            hit = result.hit
            lines.append(
                f"[{index}] Source: {hit.section} | Page: {hit.page_number} | Chunk: {hit.chunk_id}\n{hit.text.strip()}"
            )
        return "\n\n".join(lines)

    def source_summary(self, results: List[RetrievedChunk]) -> str:
        if not results:
            return ""
        entries = []
        seen = set()
        for result in results:
            hit = result.hit
            label = f"{hit.section} (Page {hit.page_number}, Chunk {hit.chunk_id})"
            if label not in seen:
                seen.add(label)
                entries.append(label)
        return "; ".join(entries)
