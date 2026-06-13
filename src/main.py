from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel

from src.chatbot.chatbot_engine import ChatbotEngine, build_generator
from src.embeddings.embedding_model import EmbeddingFactory
from src.embeddings.vector_store import FaissVectorStore
from src.ingestion.chunker import Chunker
from src.ingestion.pdf_loader import PdfLoader
from src.memory.conversation_memory import ConversationMemory
from src.retrieval.retriever import Retriever
from src.utils.config import AppConfig, load_config
from src.utils.logger import get_logger


console = Console()
logger = get_logger(__name__)


def build_index(config: AppConfig) -> FaissVectorStore:
    """Ingest the PDF and persist a FAISS index plus chunk metadata."""

    loader = PdfLoader(config.pdf_path)
    pages = loader.load_pages()
    chunker = Chunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    chunks = chunker.chunk_pages(pages)

    embedding_model = EmbeddingFactory.create(config)
    store = FaissVectorStore(config.faiss_index_path, config.faiss_metadata_path, embedding_model)
    store.build(chunks)

    logger.info("Indexed %s pages into %s chunks", len(pages), len(chunks))
    logger.info("Vector store saved to %s", config.vector_db_dir)
    return store


def build_chatbot(config: AppConfig) -> ChatbotEngine:
    """Create a ready-to-run chatbot engine."""

    embedding_model = EmbeddingFactory.create(config)
    store = FaissVectorStore(config.faiss_index_path, config.faiss_metadata_path, embedding_model)
    retriever = Retriever(store, default_top_k=config.top_k)
    memory = ConversationMemory(max_turns=config.max_memory_turns)
    generator = build_generator(config.openai_chat_model, config.local_generation_model)
    return ChatbotEngine(retriever=retriever, memory=memory, generator=generator)



def run_chat(config: AppConfig) -> None:
    """Start a simple interactive command-line chat loop."""

    if not config.faiss_index_path.exists() or not config.faiss_metadata_path.exists():
        console.print("[yellow]No index found. Building a fresh one first...[/yellow]")
        build_index(config)

    chatbot = build_chatbot(config)
    console.print(Panel.fit("Financial Policy Chatbot", title="Ready", subtitle="Type 'exit' to quit"))

    while True:
        question = console.input("[bold cyan]You[/bold cyan]: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        response = chatbot.answer(question, top_k=config.top_k)
        console.print(Panel(response.answer, title="Bot", subtitle=response.sources[0] if response.sources else "No source"))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Financial policy chatbot")
    parser.add_argument("command", nargs="?", choices={"build-index", "chat"}, default="chat")
    return parser.parse_args()



def main() -> None:
    config = load_config()
    args = parse_args()
    if args.command == "build-index":
        build_index(config)
        return
    run_chat(config)


if __name__ == "__main__":
    main()
