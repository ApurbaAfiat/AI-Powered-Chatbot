from src.ingestion.chunker import Chunker
from src.ingestion.pdf_loader import PageDocument


def test_chunker_preserves_metadata() -> None:
    page = PageDocument(
        page_number=3,
        section_title="Debt Policy",
        source="Page 3",
        text="A" * 620,
    )
    chunker = Chunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_pages([page])

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "p003_c000"
    assert chunks[0].section == "Debt Policy"
    assert chunks[0].source == "Page 3"
    assert chunks[0].end_char - chunks[0].start_char == 500
    assert chunks[1].start_char == 450
