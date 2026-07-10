Here's a README.md for DocSensei — copy and paste as-is:

```markdown
# DocSensei — Enterprise Multi-Document Adaptive RAG Assistant

DocSensei is a production-grade Retrieval-Augmented Generation (RAG) assistant that answers questions strictly from user-uploaded documents, with page-and-file level citations. It combines dense + sparse hybrid retrieval, an adaptive query router, contextual compression, and conversation memory into a single Jupyter notebook pipeline that mirrors a deployed `app.py` backend.

## Features

- **Multi-format ingestion** — PDF, DOCX, PPTX, and TXT files in a single session
- **Grounded answers only** — responds strictly from uploaded documents, never general knowledge, with explicit fallback when the answer isn't found
- **Adaptive RAG Decision Layer** — an LLM-based router classifies each query (Simple, Complex, Comparison, Summarization, Reasoning, Multi-document, Follow-up) and adjusts retrieval depth (`top_k`) accordingly
- **Hybrid retrieval** — dense vector search (Pinecone) fused with sparse keyword search (BM25) using weighted score normalization
- **Cross-encoder reranking** — candidate pool reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2` before compression
- **Contextual compression** — LLM extracts only query-relevant sentences from top-ranked chunks, discarding filler
- **Conversation memory + query rewriting** — maintains a sliding window of past turns and rewrites follow-up questions (with pronouns/references) into standalone queries
- **Structured document summarization** — auto-generates executive summaries, key topics, important facts, and key figures per document
- **Suggested questions** — auto-generates 5 questions spanning Basic → Analytical difficulty per document
- **OCR fallback** — automatically OCRs PDF pages with insufficient extractable text (via Tesseract + Poppler)
- **Session reset** — full wipe of vectors, BM25 index, chunks, memory, and cached summaries on demand

## Architecture

```
Documents (PDF/DOCX/PPTX/TXT)
        │
   Document Loaders + OCR fallback
        │
   Clean & Chunk (structure-aware, 1000/150 overlap)
        │
   Embed (all-MiniLM-L6-v2) ──► Pinecone (dense index)
        │
   Tokenize ──► BM25 (sparse index)
        │
   ┌────────────────────────┐
   │  User Query             │
   └────────────────────────┘
        │
   Adaptive Router (classify query → top_k)
        │
   Query Rewriter (standalone check + rewrite using memory)
        │
   Hybrid Retrieval (dense + BM25, weighted fusion)
        │
   Cross-Encoder Reranking
        │
   Contextual Compression (top-5 chunks)
        │
   Groq LLM (llama-3.3-70b-versatile) ──► Answer + Citations
        │
   Conversation Memory (last 6 turns)
```

## Tech Stack

| Component            | Technology                                      |
|-----------------------|--------------------------------------------------|
| LLM                   | Groq — `llama-3.3-70b-versatile`                 |
| Dense retrieval       | Pinecone (serverless, AWS `us-east-1`)           |
| Sparse retrieval      | BM25Okapi (`rank_bm25`)                          |
| Embeddings            | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Reranker              | `cross-encoder/ms-marco-MiniLM-L-6-v2`           |
| Document parsing      | `pypdf`, `python-docx`, `python-pptx`            |
| OCR                   | `pytesseract`, `pdf2image`, Poppler, Tesseract   |
| Config/env            | `python-dotenv`                                  |

## Configuration

Key parameters (in `Config`):

| Parameter             | Default                          | Description                              |
|------------------------|-----------------------------------|--------------------------------------------|
| `CHUNK_SIZE`           | 1000                              | Characters per chunk                        |
| `CHUNK_OVERLAP`        | 150                               | Overlap between chunks                      |
| `DENSE_WEIGHT`         | 0.6                               | Weight for Pinecone score in fusion        |
| `BM25_WEIGHT`          | 0.4                               | Weight for BM25 score in fusion            |
| `RERANK_CANDIDATE_POOL`| 15                                | Chunks fed to reranker                      |
| `RERANK_TOP_N`         | 8                                 | Chunks kept after reranking                 |
| `COMPRESSION_TOP_N`    | 5                                 | Chunks passed through compression           |
| `MEMORY_WINDOW`        | 6                                 | Past conversation turns retained            |
| `ENABLE_OCR_FALLBACK`  | True                              | OCR pages with low extractable text        |
| `OCR_MIN_CHARS_PER_PAGE`| 20                               | Threshold to trigger OCR                    |

Router `top_k` by query type: Simple=3, Complex=8, Comparison=8, Summarization=10, Reasoning=8, Multi-document=10, Follow-up=5.

## Setup

### 1. Install dependencies
```bash
pip install -q groq pinecone-client rank_bm25 sentence-transformers
pip install -q pypdf python-docx python-pptx
pip install -q pytesseract pdf2image pillow
pip install -q python-dotenv
apt-get -qq install -y poppler-utils tesseract-ocr
```

### 2. Set environment variables
Create a `.env` file in the project root:
```
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

### 3. Run the notebook
Execute cells top to bottom. The Pinecone index (`docsensei-index`) is created automatically if it doesn't already exist.

## Usage

**Ingest documents:**
```python
documents = ["your_file.pdf", "your_file.pptx"]
session.ingest(documents)
```
This chunks, embeds, indexes (Pinecone + BM25), and generates summaries + suggested questions for each file. Each new ingest starts a fresh session (previous vectors/chunks are cleared).

**Ask questions (interactive loop):**
```python
result = session.answer("your question")
print(result["answer"])
print(result["citations"])
```
Or run the built-in interactive loop cell, which prompts for questions until you type `exit`.

**Reset the session:**
```python
reset_session()
```
Wipes all Pinecone vectors, BM25 index, chunks, memory, summaries, and suggested questions without ingesting new files.

## Output Format

Each answer returns:
- `answer` — plain-prose response grounded only in retrieved context
- `citations` — list of `{source_file, page}` references

If the answer isn't in the documents, the assistant responds exactly with:
> "I could not find this information in the uploaded documents."

## Notes

- The notebook's logic mirrors a deployed `app.py` Flask/Gradio backend, so behavior in the notebook matches production.
- Designed to be dependency-light where possible — chunking, dense retrieval, and document loading are custom implementations rather than heavy framework abstractions (e.g., no LangChain).
```

Want me to also generate this as an actual `.md` file and/or a matching `.pptx`/`.docx` version?
