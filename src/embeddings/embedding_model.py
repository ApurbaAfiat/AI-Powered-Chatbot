from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np

from src.utils.config import AppConfig

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:  # pragma: no cover - optional dependency
    OpenAIEmbeddings = None

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - hard dependency for local mode
    SentenceTransformer = None
    _sentence_transformers_error = exc
else:
    _sentence_transformers_error = None


class EmbeddingModel(Protocol):
    """Protocol for pluggable embedding backends."""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        ...


@dataclass
class SentenceTransformerEmbeddingModel:
    """Local embedding model backed by sentence-transformers."""

    model_name: str
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        if SentenceTransformer is None:  # pragma: no cover - environment specific
            raise ImportError("sentence-transformers is required for local embeddings") from _sentence_transformers_error
        self._model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=self.normalize_embeddings)
        return np.asarray(vectors, dtype=np.float32).tolist()


@dataclass
class OpenAIEmbeddingModel:
    """Embedding backend that uses OpenAI when credentials are configured."""

    model_name: str

    def __post_init__(self) -> None:
        if OpenAIEmbeddings is None:  # pragma: no cover - optional dependency
            raise ImportError("langchain-openai is required for OpenAI embeddings")
        self._embeddings = OpenAIEmbeddings(model=self.model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)


class EmbeddingFactory:
    """Create the configured embedding backend with a local fallback."""

    @staticmethod
    def create(config: AppConfig) -> EmbeddingModel:
        if config.openai_chat_model and config_env_has_openai_key():
            if OpenAIEmbeddings is not None:
                return OpenAIEmbeddingModel(model_name=os_env_openai_embedding_model())
        return SentenceTransformerEmbeddingModel(model_name=config.sentence_embedding_model)



def config_env_has_openai_key() -> bool:
    import os

    return bool(os.getenv("OPENAI_API_KEY", "").strip())



def os_env_openai_embedding_model() -> str:
    import os

    return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
