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
