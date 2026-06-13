from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    """Central application settings."""

    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    docs_dir: Path = project_root / "docs"
    vector_db_dir: Path = project_root / "vector_db"
    default_pdf_name: str = "financial_policy.pdf"
    fallback_pdf_name: str = "For Task - Policy file.pdf"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    faiss_index_name: str = "policy.index"
    faiss_metadata_name: str = "policy_chunks.json"
    sentence_embedding_model: str = "all-MiniLM-L6-v2"
    openai_chat_model: str = "gpt-4o-mini"
    local_generation_model: str = "google/flan-t5-small"
    max_context_chunks: int = 4
    max_memory_turns: int = 5

    @property
    def pdf_path(self) -> Path:
        preferred = self.data_dir / self.default_pdf_name
        if preferred.exists():
            return preferred
        fallback = self.project_root / self.fallback_pdf_name
        return fallback

    @property
    def faiss_index_path(self) -> Path:
        return self.vector_db_dir / self.faiss_index_name

    @property
    def faiss_metadata_path(self) -> Path:
        return self.vector_db_dir / self.faiss_metadata_name

    @property
    def env_path(self) -> Path:
        return self.project_root / ".env"



def load_config() -> AppConfig:
    """Load environment variables and return the application configuration."""

    load_dotenv()
    return AppConfig(
        sentence_embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        local_generation_model=os.getenv("LOCAL_GENERATION_MODEL", "google/flan-t5-small"),
    )
