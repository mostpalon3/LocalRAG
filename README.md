# LocalRag

One-line summary: a fully local RAG system that indexes documents, retrieves relevant evidence with vector search, and answers/evaluates queries via a browser chat UI and CLI.

![Python 3.13.5](https://img.shields.io/badge/python-3.13.5-blue)
![License Unspecified](https://img.shields.io/badge/license-unspecified-lightgrey)
![Last Commit](https://img.shields.io/badge/last%20commit-2026--07--06-blue)

![rag-image](./rag-design-basic.png)

## 2. Demo
- Fully offline - no cloud API dependency.

## 3. System Architecture

LocalRag implements this pipeline:

Document Ingestion -> Chunking -> Embedding -> Vector Storage -> Query Processing -> Retrieval -> LLM Generation -> Response

### Stage-by-stage technical flow

1. Document Ingestion
- Entry points:
  - CLI: `main.py` (`add`, `run` commands)
  - UI: `streamlit_app.py` (path-based indexing and upload-and-index)
- Input is a list of file paths; uploaded files are persisted into `.streamlit_uploads/` before indexing.

2. Chunking
- `src/impl/indexer.py` uses `DocumentConverter` + `HybridChunker` from Docling.
- Each chunk is transformed into a `DataItem` with:
  - `content`: headings + body text
  - `source`: filename + chunk index
- This adds provenance metadata while keeping chunk text self-descriptive.

3. Embedding
- `src/impl/datastore.py` embeds each chunk with Ollama (`nomic-embed-text` by default).
- Embedding dimension is configurable via `OLLAMA_EMBED_DIMENSIONS` (default `768`) and used to define LanceDB schema.

4. Vector Storage
- LanceDB table: `data/sample-lancedb/rag-table`.
- Schema includes `vector`, `content`, `source`.
- Upsert behavior is implemented via `merge_insert("source")`, which avoids naive append-only duplicates for the same source key.

5. Query Processing
- User query enters through:
  - CLI `query` command, or
  - Streamlit chat input.
- Query text is embedded with the same embedding model to maintain embedding-space consistency.

6. Retrieval
- `src/impl/retriever.py` delegates to datastore vector search (`top_k`, default 10 in retriever).
- Returned payload is list of retrieved chunk text.
- Current build is fully local: no external reranker.

7. LLM Generation
- `src/impl/response_generator.py` composes a structured prompt:
  - `<context>...</context>` with retrieved chunks
  - `<question>...</question>` with user query
- `src/util/invoke_ai.py` calls Ollama chat model (`qwen3:8b` default) with a strict system prompt to avoid hallucinated facts.

8. Response
- CLI prints the response.
- Streamlit renders response in chat history (`st.session_state.messages`) and can run evaluation over custom QA sets.

### ASCII architecture diagram

```text
                  +-----------------------------+
                  |      streamlit_app.py       |
                  |  (chat, upload, evaluation) |
                  +--------------+--------------+
                                 |
                                 v
 +---------+      +--------------------------+      +-------------------+
 | main.py |----->|   src/rag_pipeline.py    |<-----| create_parser.py  |
 | (CLI)   |      | (orchestration layer)    |      | (CLI commands)    |
 +----+----+      +-----+------+-----+-------+      +-------------------+
      |                 |      |     |
      | add/reset/query |      |     | evaluate
      v                 v      v     v
 +-------------+   +---------+  +-----------+   +----------------+
 | Indexer     |   |Retriever|  |Generator  |   | Evaluator      |
 | (Docling)   |   |(vector) |  |(Ollama)   |   |(LLM grading)   |
 +------+------+   +----+----+  +-----+-----+   +--------+-------+
        |               |             |                  |
        v               |             |                  |
 +-------------+        |             |                  |
 | Datastore   |<-------+-------------+------------------+
 | (LanceDB +  |
 |  embeddings)|
 +------+------+ 
        |
        v
 +-----------------------------+
 | data/sample-lancedb/rag-table|
 +-----------------------------+
```

### Why these technologies

- LanceDB over ChromaDB
  - Reason in this codebase: table-like schema control (`pyarrow`), explicit local file path, and predictable merge/upsert path (`merge_insert`).
  - This aligns with the project goal of local, inspectable storage with simple persistence semantics.

- Ollama over OpenAI API
  - Reason in this codebase: fully local inference for both chat and embeddings, no cloud API keys, and consistent local deployment behavior for demos/interviews.

- Docling over ad-hoc PDF parsing
  - Reason in this codebase: integrated document conversion + chunking pipeline with metadata support (`headings`, `origin.filename`).

- Streamlit over a custom frontend stack
  - Reason in this codebase: minimal glue code from pipeline to usable UI (chat + indexing + evaluation) while preserving one-language (Python) workflow.

## 4. Tech Stack

| Component | Technology | Reason |
|---|---|---|
| LLM serving | Ollama (`qwen3:8b`) | Local chat inference, no cloud dependency |
| Embedding model | Ollama (`nomic-embed-text`) | Local embeddings aligned with local-first constraint |
| Vector DB | LanceDB | File-backed local vector store with explicit schema and merge semantics |
| Document parsing/chunking | Docling (`DocumentConverter`, `HybridChunker`) | Structured conversion and chunk metadata from document sources |
| UI layer | Streamlit | Fast path to interactive chat, upload, and evaluation in browser |

## 5. Core Design Decisions

1. Decision: Keep retrieval fully local and remove hosted reranking.
- Alternatives Considered: Cohere rerank API, local cross-encoder reranker, hybrid lexical/vector reranking.
- Why This Choice: The project objective is offline operation with zero cloud API dependency. Current retriever delegates directly to datastore vector search to preserve that property.

2. Decision: Single embedding model for both indexing and querying (`nomic-embed-text`).
- Alternatives Considered: separate query/document embedding models, OpenAI embedding endpoints.
- Why This Choice: Shared embedding space reduces mismatch risk and simplifies operational setup. It also avoids additional runtime/services.

3. Decision: Embedding dimensionality is environment-configurable but schema-bound (`OLLAMA_EMBED_DIMENSIONS`, default 768).
- Alternatives Considered: infer dimensions dynamically per run, hardcode dimensions with no override.
- Why This Choice: LanceDB schema requires fixed vector size; explicit env configuration makes model swaps possible while keeping schema deterministic.

4. Decision: Chunk payload includes headings in addition to body text.
- Alternatives Considered: body text only, larger monolithic chunks per page.
- Why This Choice: Headings add topical context that improves retrieval precision for entity-heavy or section-specific questions.

5. Decision: Use `source` as merge key in datastore upsert (`merge_insert("source")`).
- Alternatives Considered: append-only writes, synthetic UUID for each chunk.
- Why This Choice: Source-based merge keeps indexing idempotent for same logical chunk identity and avoids unbounded duplicates from repeated runs.

6. Decision: Evaluate with LLM-as-judge using strict XML tags (`<reasoning>`, `<result>`).
- Alternatives Considered: exact-string match, semantic similarity-only scoring, external eval service.
- Why This Choice: Enables qualitative reasoning output with a simple parser (`extract_xml_tag`) and supports near-correct paraphrase detection beyond exact match.

7. Decision: Concurrency in both embedding insertion and evaluation.
- Alternatives Considered: fully sequential indexing/evaluation.
- Why This Choice: Network/model-bound calls are parallelized (`ThreadPoolExecutor`) to improve throughput on multi-question eval and multi-item embedding generation.

8. Decision: Streamlit session state as transient application state.
- Alternatives Considered: DB-backed conversation persistence, stateless request-response only.
- Why This Choice: Keeps implementation lightweight while supporting real UX requirements (chat history and latest evaluation results) for a resume-grade demo.

## 6. Project Structure

```text
simple-rag-pipeline/
├── main.py                         # CLI entrypoint wiring reset/add/query/evaluate commands
├── create_parser.py                # argparse command definitions and shared CLI args
├── streamlit_app.py                # browser UI: chat, upload/index, dynamic evaluation
├── requirements.txt                # runtime dependencies
├── .gitignore                      # local artifacts ignored in git
├── rag-design-basic.png            # architecture image used in README
├── sample_data/
│   ├── source/                     # sample source docs for indexing
│   └── eval/
│       └── sample_questions.json   # sample QA pairs for CLI evaluation
├── src/
│   ├── rag_pipeline.py             # orchestration layer for pipeline operations
│   ├── impl/                       # concrete component implementations
│   │   ├── datastore.py            # LanceDB + embedding client + vector search
│   │   ├── indexer.py              # Docling converter/chunker -> DataItem generation
│   │   ├── retriever.py            # retrieval policy (currently datastore search only)
│   │   ├── response_generator.py   # prompt assembly and model invocation
│   │   ├── evaluator.py            # LLM-based correctness evaluator
│   │   └── __init__.py             # exports concrete implementation classes
│   ├── interface/                  # abstract interfaces/contracts
│   │   ├── base_datastore.py       # DataItem model + datastore contract
│   │   ├── base_indexer.py         # indexer contract
│   │   ├── base_retriever.py       # retriever contract
│   │   ├── base_response_generator.py # generator contract
│   │   ├── base_evaluator.py       # evaluator contract + EvaluationResult model
│   │   └── __init__.py             # exports interface symbols
│   └── util/
│       ├── invoke_ai.py            # Ollama chat wrapper
│       └── extract_xml.py          # XML tag extraction helper used by evaluator
└── data/                           # local LanceDB files (generated at runtime)
```

Grouping summary:
- Core pipeline: `main.py`, `streamlit_app.py`, `src/rag_pipeline.py`, `create_parser.py`
- Data layer: `src/impl/datastore.py`, `data/`
- Interface layer: `src/interface/*`
- Config/runtime surface: `requirements.txt`, `.gitignore`, environment variables

## 7. Getting Started

### Prerequisites

- Python 3.13+
- Ollama installed and running

Pull required models:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

Optional env vars:

```bash
export OLLAMA_HOST='http://localhost:11434'
export OLLAMA_CHAT_MODEL='qwen3:8b'
export OLLAMA_EMBED_MODEL='nomic-embed-text'
export OLLAMA_EMBED_DIMENSIONS='768'
```

### Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Index your own documents

CLI path-based indexing:

```bash
PYTHONPATH=src python3 main.py add -p /absolute/or/relative/path/to/docs
```

Browser upload indexing:

```bash
streamlit run streamlit_app.py
```

Then use the sidebar `Upload documents` -> `Upload and index`.

### Run queries

CLI query:

```bash
PYTHONPATH=src python3 main.py query "Your question"
```

Full pipeline run:

```bash
PYTHONPATH=src python3 main.py run
```

## 8. Performance & Benchmarks

Current measured/observed data in this repository state:

- Indexed chunk count (sample corpus run): 41 items added to datastore.
- End-to-end pipeline (`PYTHONPATH=src python3 main.py run`) completed successfully on local setup.
- No dedicated latency instrumentation is currently implemented in code (no timing middleware/logging hooks).
- No memory profiling hooks are currently implemented.

Recommended benchmark additions (for rigorous reporting):

- Retrieval latency (P50/P95) for fixed question set
- Generation latency split by model token count
- Indexing throughput (chunks/sec)
- Peak RSS during indexing and evaluation

## 9. Limitations & Future Improvements

Current limitations:

- Retrieval uses vector search only; no secondary reranking stage.
- Prompt context can become large because top-k chunks are concatenated without token budgeting.
- Evaluation relies on LLM-generated XML tags; malformed output can reduce eval stability.
- CLI defaults still point to sample evaluation file unless explicitly overridden.

Planned improvements:

- Add optional local reranker (cross-encoder) for improved ordering precision.
- Implement context-window management (token-aware truncation/selection).
- Add structured observability (timings, retrieval stats, model call telemetry).
- Add persistent run artifacts for evaluation history and regression tracking.

## 10. What I Learned

- Interface-first design significantly reduces refactor cost when swapping providers (OpenAI -> Ollama, hosted reranker -> local-only retrieval).
- Retrieval quality depends as much on chunk construction and metadata strategy as on embedding model choice.
- Offline-first constraints force better dependency discipline and make demos more reproducible.
- Evaluation pipelines need explicit output contracts (e.g., XML tags) plus robust fallback handling for malformed model outputs.