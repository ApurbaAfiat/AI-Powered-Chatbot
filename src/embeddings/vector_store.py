from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
import json

import faiss
import numpy as np

from src.embeddings.embedding_model import EmbeddingModel
from src.ingestion.chunker import Chunk


@dataclass(frozen=True)
class SearchHit:
    """A ranked search result returned from FAISS."""

    chunk_id: str
    section: str
    source: str
    page_number: int
    text: str
    score: float
    start_char: int
    end_char: int


class FaissVectorStore:
    """Persist chunks and embeddings in a FAISS index plus JSON metadata."""

    def __init__(self, index_path: Path, metadata_path: Path, embedding_model: EmbeddingModel) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model = embedding_model
        self._index: faiss.Index | None = None
        self._chunks: List[dict] = []

    @property
    def chunks(self) -> List[dict]:
        return list(self._chunks)

    def build(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build a vector store without chunks")

        texts = [chunk.text for chunk in chunks]
        vectors = np.asarray(self.embedding_model.embed_texts(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.size == 0:
            raise ValueError("Embedding model returned an invalid matrix")

        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self._index = index
        self._chunks = [chunk.to_metadata() for chunk in chunks]
        self._save()

    def load(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("FAISS index or metadata file is missing")
        self._index = faiss.read_index(str(self.index_path))
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            self._chunks = json.load(handle)

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        if self._index is None:
            self.load()

        assert self._index is not None
        if not self._chunks:
            return []

        query_vector = np.asarray(self.embedding_model.embed_texts([query]), dtype=np.float32)
        faiss.normalize_L2(query_vector)
        scores, indices = self._index.search(query_vector, min(top_k, len(self._chunks)))

        hits: List[SearchHit] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            record = self._chunks[index]
            hits.append(
                SearchHit(
                    chunk_id=record["chunk_id"],
                    section=record["section"],
                    source=record["source"],
                    page_number=int(record["page_number"]),
                    text=record["text"],
                    score=float(score),
                    start_char=int(record["start_char"]),
                    end_char=int(record["end_char"]),
                )
            )
        return hits

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(self._chunks, handle, indent=2, ensure_ascii=False)
