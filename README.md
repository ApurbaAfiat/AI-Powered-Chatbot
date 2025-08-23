# Financial Policy Q&A Chatbot (ChromaDB + Sentence-Transformers)

A lightweight, notebook-first Retrieval-Augmented Generation (RAG) chatbot.  
It reads a **financial policy PDF**, builds a **local ChromaDB** index with **sentence-transformers** embeddings, and answers questions with **page citations**. Runs fully offline—no API keys required.

> **Stack:** Python · Jupyter/VS Code Notebooks · PyMuPDF · ChromaDB · sentence-transformers · ipywidgets (optional UI)

---

## Features

- **One-pass ingest:** Read PDF → chunk → embed → store in **Chroma** (persisted on disk).
- **Local embeddings:** `BAAI/bge-small-en-v1.5` (fast; no API).
- **Citations:** Answers include inline `(p. X)` tags + a `Sources: p. …` line.
- **Conversation memory:** Rolling summary to make follow-ups work.
- **Two UIs:** 
  - Widget chat (send button / Enter to submit; scrollable, wrapped output)
  - Console loop fallback (works without ipywidgets)
- **Tunable retrieval:** `MAX_CHARS`, `OVERLAP`, and `Top‑k` for precision/recall trade‑offs.

---

Install packages:
```bash
pip install -U chromadb sentence-transformers pymupdf ipywidgets tqdm pandas rich python-dotenv
```

> If using classic Jupyter Notebook (not Lab), you may need:
> ```bash
> jupyter nbextension enable --py widgetsnbextension
> ```

---

## Quickstart (how to run the chatbot)

1. **Open** `chatbot.ipynb` in Jupyter/VS Code.
2. **Place your PDF** at `data/raw/policy.pdf` **or** set `PDF_PATH` to your file path.
3. **Run the ingest cell** (Phases 2+3 one‑pass) to build the index. Set these knobs for good recall:
   ```python
   MAX_CHARS  = 700     # chunk size (smaller => more precise retrieval)
   OVERLAP    = 150     # overlap to preserve context across chunks
   CLEAR_OLD  = True    # rebuild index cleanly
   BATCH_SIZE = 16
   ```
4. **Load retrieval + answerer (Phases 4–6)**. These define:
   - `retrieve(query, k)`
   - `rag_answer_v2` + `answer_with_memory_v2`
   - `ConversationMemory` (rolling context)
5. **Run the chat UI (Phase 8)**  
   - **Widget mode** (preferred): press **Enter** or click **Send** to submit. Answers wrap and scroll.
   - **Console mode** (fallback): supports `/k N` (Top‑k), `/reset`, `/quit`.

### Sample questions
- *What is the financial policy about, in one paragraph?*  
- *What are the short‑term financial objectives?*  
- *What does the policy say about managing fiscal risk prudently?*  
- *What is the superannuation funding target and timeframe?*  
- *Show taxation as a percentage of GSP by year.*

---

##  How it works (phases)

1. **Phase 1 – Read PDF:** PyMuPDF extracts text per page and cleans whitespace.
2. **Phase 2 – Chunk:** Split pages into ~700‑char chunks with 150‑char overlap (tunable).
3. **Phase 3 – Index:** Embed chunks with `BAAI/bge-small-en-v1.5` and store in **Chroma** (persisted).
4. **Phase 4 – Retrieve:** Embed query and return top‑k chunk texts + metadata (page numbers).
5. **Phase 5 – Answer:** Offline extractive RAG (`rag_answer_v2`), with MMR selection & formatting.
6. **Phase 6 – Memory:** Rolling summary augments follow‑up queries.
7. **Phase 7 – Eval (optional):** Small harness for Recall@k and citation overlap.
8. **Phase 8 – UI:** ipywidgets chat (or console loop) to interact locally.

---

##  Key configuration

- **Chunking**
  - `MAX_CHARS` — chunk length (lower → more precise; higher → faster ingest)
  - `OVERLAP` — preserve cross‑chunk context (100–180 typical)
- **Retrieval**
  - **Top‑k** slider in the UI (3–12). Increase when answers miss context.
- **Persistence/Cache**
  - `PERSIST_DIR` — where Chroma stores its DuckDB/Parquet
  - `HF_HOME` — HuggingFace cache (put on SSD to save C: space)

---

##  Troubleshooting

**“PDF not found …”**  
Check `PDF_PATH` spelling, spaces, and extension. If using Windows, raw strings (`r"…"`).

**`AttributeError: 'Client' object has no attribute 'persist'`**  
Use the newer API:
```python
from chromadb import PersistentClient
client = PersistentClient(path=str(PERSIST_DIR))  # no client.persist() call
```

**Widget submits on each keystroke**  
Ensure the input uses **`.on_submit()`** (Enter) or a **Send** button—no `observe(..., names="value")`.

**Long answers get cut off**  
Use the HTML output panel with `white-space: pre-wrap;` (already in the improved Phase‑8 cell).

**Answers are generic**  
You likely indexed too few chunks. Re‑ingest with:
```python
MAX_CHARS = 600–800
OVERLAP   = 140–180
CLEAR_OLD = True
```
Aim for **≥35** chunks; then try again with `Top‑k = 8–10`.

---

##  Data & privacy
All indexing and querying happens locally on your machine. No document text or embeddings leave your environment.

---

##  FAQ

**Q: Can I use a different PDF?**  
Yes—set `PDF_PATH` to the new file and re‑ingest (`CLEAR_OLD=True`).

**Q: How do I reuse the index next time?**  
Just reload via:
```python
from chromadb import PersistentClient
client = PersistentClient(path=str(PERSIST_DIR))
collection = client.get_or_create_collection("policy")
```
No need to re‑embed unless the source document changed.

**Q: Can I move everything off C:\\?**  
Yes—point both `PERSIST_DIR` and `HF_HOME` to your SSD (e.g., `E:\policy-chatbot\…`).

---

##  Contributing
PRs welcome—add new chunkers, retriever variants, or UI tweaks (e.g., citations as clickable anchors).
