# DocSensei — Enterprise Multi-Document Adaptive RAG Assistant

DocSensei is a production-grade Retrieval-Augmented Generation (RAG) assistant that answers questions strictly from user-uploaded documents, with page-and-file level citations. It combines dense + sparse hybrid retrieval, an adaptive query router, contextual compression, and conversation memory into a modular pipeline built from independent, swappable components.

## Features

- **Multi-format ingestion** — PDF, DOCX, PPTX, and TXT files in a single session
- **Grounded answers only** — responds strictly from uploaded documents, never general knowledge, with explicit fallback when the answer isn't found
- **Adaptive RAG Decision Layer** — an LLM-based router classifies each query and adjusts retrieval depth accordingly
- **Hybrid retrieval** — dense vector search (Pinecone) fused with sparse keyword search (BM25)
- **Cross-encoder reranking** — candidate pool reranked before compression
- **Contextual compression** — LLM extracts only query-relevant sentences from top-ranked chunks
- **Conversation memory + query rewriting** — rewrites follow-up questions into standalone queries
- **Structured document summarization** — auto-generates summaries per document
- **Suggested questions** — auto-generates leveled questions per document
- **OCR fallback** — automatically OCRs PDF pages with insufficient extractable text
- **Session reset** — full wipe of vectors, BM25 index, chunks, memory, and cached summaries on demand

## Components

### 1. Document Loaders
Parses PDF, DOCX, PPTX, and TXT files into page/slide-level text.
- PDF text is extracted natively via `pypdf`, page by page.
- Pages with little or no extractable text (e.g. scanned images) are transparently retried through **OCR** (`pytesseract` + `pdf2image`, rendered at `OCR_DPI=200`), but only if the page falls below `OCR_MIN_CHARS_PER_PAGE=20` characters and OCR binaries (Tesseract + Poppler) are actually available on the system.
- DOCX/PPTX loaders preserve structural markers like headings, tables (`[Table...]`), and speaker notes (`[Speaker notes...]`) so they survive into chunking.

### 2. Text Cleaner
`clean_text()` normalizes Unicode (NFKC), standardizes line endings, collapses redundant whitespace, and removes duplicate blank lines — while preserving heading/bullet markers produced by the loaders.

### 3. Raw Document Builder
`build_raw_documents()` concatenates each file's cleaned pages into one full document string per file. These full texts feed the Summarizer and Suggested Question Generator (not the chunker, which works from `loaded` pages directly).

### 4. Chunker
A dependency-free, structure-aware chunker:
- `_presegment_structure_aware()` first groups lines into blocks so headings stay attached to their following paragraph, and consecutive bullet lines stay grouped as one block — before any character-level splitting happens.
- `recursive_character_split()` then splits each block into chunks of `CHUNK_SIZE=1000` characters with `CHUNK_OVERLAP=150` characters, recursively backing off across separators to avoid cutting mid-sentence.

### 5. Embedder
Uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). `embed_texts()` batches inputs (`batch_size=32`) and L2-normalizes embeddings so cosine similarity in Pinecone behaves correctly.

### 6. Dense Index (Pinecone)
- A serverless Pinecone index (`docsensei-index`, AWS `us-east-1`, cosine metric) is created automatically if it doesn't exist.
- `upsert_chunks_to_pinecone()` embeds and upserts all chunks in batches.
- `delete_all_vectors()` wipes the entire index at the start of every new ingest, so no stale document can ever be retrieved after a fresh upload.

### 7. Sparse Index (BM25)
- `build_bm25_index()` tokenizes every chunk (`_tokenize()` — lowercase alphanumeric regex) and builds an in-memory `BM25Okapi` index, rebuilt from scratch after every ingest.
- `bm25_retrieve()` scores the query against the corpus and returns the top-k chunks with `score > 0`.

### 8. Hybrid Retriever
`hybrid_retrieve()` fuses both indexes:
1. Pulls `top_k * 2` candidates from each of dense and BM25 retrieval.
2. Min-max normalizes scores within each result set to `[0, 1]` (`_min_max_normalize()`), falling back to a constant `1.0` when all scores tie.
3. Combines them into a single weighted score per chunk: `DENSE_WEIGHT=0.6` × normalized dense score + `BM25_WEIGHT=0.4` × normalized BM25 score. Chunks found by both retrievers are tagged `"hybrid"`.
4. Returns the top-k chunks by combined weighted score.

### 9. Adaptive Router
Classifies every incoming query into one of seven labels — **Simple, Complex, Comparison, Summarization, Reasoning, Multi-document, Follow-up** — via an LLM call (`ROUTER_SYSTEM_PROMPT`, temperature 0.0), which instructs the model to respond with only the single label.
- Each label maps to a retrieval depth: Simple=3, Complex=8, Comparison=8, Summarization=10, Reasoning=8, Multi-document=10, Follow-up=5.
- If the LLM call fails (e.g. an API outage), `_heuristic_classify_query()` provides a fast, dependency-free fallback using keyword/length rules, so retrieval never hard-stops.

### 10. Query Rewriter
Turns context-dependent follow-ups into standalone queries, in two steps:
1. `_needs_rewrite()` asks the LLM (`STANDALONE_CHECK_PROMPT`) whether the query already stands alone without conversation history; the more expensive rewrite step is skipped if it does. Defaults to "needs rewrite" on error (the safer choice).
2. If needed, `rewrite_query()` calls the LLM again (`REWRITE_SYSTEM_PROMPT`) with the conversation history and the follow-up question, and returns a rewritten, standalone version that preserves original intent.

### 11. Cross-Encoder Reranker
Loads `cross-encoder/ms-marco-MiniLM-L-6-v2` and reorders the top `RERANK_CANDIDATE_POOL=15` hybrid-retrieval candidates by relevance to the query, keeping the top `RERANK_TOP_N=8`.

### 12. Contextual Compressor
`compress_context()` shrinks the top `COMPRESSION_TOP_N=5` reranked chunks down to only their query-relevant sentences:
- Numbers each chunk (`[CHUNK i]`) and sends them all together with the query to the LLM.
- System prompt instructs the model to extract only relevant sentences per chunk, outputting an empty string for chunks with nothing relevant, and to respond with **only a JSON object** mapping chunk index → extracted text (no markdown fences, no commentary).
- The response is parsed as JSON; each chunk's text is replaced with its extracted excerpt (or left unchanged if extraction is empty).
- Chunks ranked below the top 5 pass through untouched.
- On any failure (bad JSON, API error), it falls back silently to the original, uncompressed chunks.

### 13. Answer Generator
Uses `ANSWER_SYSTEM_PROMPT` to instruct the Groq LLM (`llama-3.3-70b-versatile`, temperature 0.1) to answer **only** from the provided (compressed) context — never outside knowledge — in plain prose, with no inline citation markers (citations are attached separately as structured data). If the context doesn't contain the answer, it must respond exactly: *"I could not find this information in the uploaded documents."*

### 14. Conversation Memory
`ConversationMemory` wraps a `deque(maxlen=MEMORY_WINDOW)` (default 6 turns). Stores `{user, assistant}` pairs, exposes them as formatted text (`as_text()`) for the rewriter, and supports `clear()` on reset.

### 15. Document Summarizer
`summarize_document()` sends each document's full text (truncated to 12,000 characters) to the LLM with `SUMMARY_SYSTEM_PROMPT`, requesting a structured JSON object with exactly these keys: `executive_summary` (2–4 sentences), `key_topics` (3–7 strings), `important_facts` (3–6 strings), `key_numbers` (array, empty if none), and `conclusion` (1–3 sentences). Falls back to a minimal placeholder summary if the LLM call or JSON parsing fails.

### 16. Suggested Question Generator
`generate_suggested_questions()` sends a text sample (truncated to 12,000 characters) to the LLM with `SUGGESTED_QUESTIONS_SYSTEM_PROMPT`, requesting exactly 5 questions — one each for **Basic, Intermediate, Advanced, Comparison, Analytical** — as a JSON array of `{category, question}` objects. Returns an empty list on failure.

### 17. Session Manager (`RAGSession`)
Orchestrates the full lifecycle:
- **`ingest(filepaths)`** — clears all previous state (Pinecone vectors, BM25 index, chunks, memory, summaries, suggested questions), then loads → chunks → embeds/upserts → builds BM25 → summarizes → generates suggested questions, all timed via a `timeit()` context manager.
- **`answer(query)`** — runs the full query pipeline: route → rewrite → hybrid retrieve → rerank → compress → generate answer → attach citations → update memory.
- **`reset_session()`** — wipes vectors, BM25, chunks, memory, summaries, and suggested questions without re-ingesting, for a clean slate.

## Architecture
