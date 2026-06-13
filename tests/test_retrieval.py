from pathlib import Path

from src.embeddings.vector_store import FaissVectorStore
from src.ingestion.chunker import Chunk
from src.retrieval.retriever import Retriever


class DummyEmbeddingModel:
    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "debt" in lowered:
                vectors.append([1.0, 0.0])
            elif "tax" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return vectors


def test_retriever_returns_relevant_chunk(tmp_path: Path) -> None:
    index_path = tmp_path / "policy.index"
    metadata_path = tmp_path / "policy_chunks.json"
    store = FaissVectorStore(index_path, metadata_path, DummyEmbeddingModel())
    store.build(
        [
            Chunk("p001_c000", "Debt", "Page 1", 1, "Debt should remain low.", 0, 24),
            Chunk("p002_c000", "Taxation", "Page 2", 2, "Taxation should remain stable.", 0, 31),
        ]
    )

    retriever = Retriever(store, default_top_k=1)
    results = retriever.retrieve("Tell me about debt")

    assert len(results) == 1
    assert results[0].hit.section == "Debt"
    assert "Debt" in retriever.build_context(results)
