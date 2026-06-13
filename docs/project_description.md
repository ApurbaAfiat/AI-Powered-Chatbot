# Project Description

This project refactors a notebook-based proof of concept into a modular Python application for answering questions about a financial policy document.

## Core Capabilities

- PDF ingestion with cleaning and section inference.
- Character-based chunking with overlap.
- FAISS-backed vector search.
- Conversation memory for follow-up questions.
- Answer generation with source citations.

## Metadata Captured

Each chunk stores:

- `section`
- `source`
- `chunk_id`
- `page_number`
- `start_char`
- `end_char`

## Operational Flow

1. Load the policy PDF.
2. Clean and chunk the text.
3. Embed chunks and store them in FAISS.
4. Retrieve the most relevant chunks for a query.
5. Use conversation memory to preserve context.
6. Generate a concise answer with cited source chunks.
