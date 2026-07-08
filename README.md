# 💬 LocalRAG

A fully **local** Retrieval-Augmented Generation (RAG) chatbot — no cloud APIs, no data leaving your machine. Ask questions about your own documents using open-source models running entirely on your hardware via [Ollama](https://ollama.com).

---

## ✨ Features

- **100% Local** — all inference runs through Ollama; no data leaves your machine
- **Multi-format ingestion** — PDFs, images (OCR), source code, plain text, zip archives
- **Intelligent chunking** — Docling-powered hybrid chunking for PDFs; line-overlap chunking for text/code
- **Query rewriting** — LLM rewrites your question for better retrieval before searching
- **Streaming responses** — answers stream token-by-token in the Streamlit UI
- **Persistent chat history** — multi-session chat with SQLite-backed history, rename & delete support
- **Dynamic evaluation** — paste or upload a JSON Q&A set and score your RAG pipeline in one click
- **CLI interface** — scriptable `main.py` for headless indexing, querying, and evaluation

---

## 🏗️ Architecture

```
  +---------------------------+              +---------------------------+
  |        WEB USER           |              |        CLI USER           |
  |  INPUT:                   |              |  INPUT:                   |
  |  · type a chat query      |              |  $ python main.py query   |
  |  · upload file / .zip     |              |  $ python main.py add     |
  |  · paste eval JSON        |              |  $ python main.py run     |
  |  · set documents path     |              |  $ python main.py reset   |
  +-------------+-------------+              +-------------+-------------+
                | (browser)                                | (terminal)
                v                                          v
  +-------------+--------------------+     +---------------+----------+   +------------------+
  |        streamlit_app.py          |     |         main.py          |<--| create_parser.py |
  |  chat UI · upload · eval panel   |     |     (CLI entry point)    |   | argparse cmds:   |
  +------+---------------------------+     +---------------+----------+   | add/reset/query/ |
         |                     |                           |              | evaluate/run     |
         | save/load           | calls pipeline directly   |              +------------------+
         | messages            +---------------------------+
         v                                 |
  +---------------------------+            v
  |  chat_history_store.py    |  +---------+-----------------+
  |  (SQLite persistence)     |  |    src/rag_pipeline.py    |
  |  data/chat_history.sqlite3|  |    (orchestration layer)  |
  +---------------------------+  +-----+-------+------+------+
                                       |       |      |
                                  index|  query|      | evaluate
                                       v       v      v
  +-------------+   +--------------+   +----------+   +------------------+
  |   Indexer   |   |  Retriever   |   | Generator|   |   Evaluator      |
  | · Docling   |   | · LLM query  |   | · Ollama |   | · LLM-as-judge   |
  | · EasyOCR   |   |   rewrite    |   |   LLM    |   | · parallel (x10) |
  | · hybrid    |   | · vector     |   | · token  |   | · Q&A scoring    |
  |   chunking  |   |   search     |   |   stream |   | · reasoning tags |
  +------+------+   +------+-------+   +----+-----+   +--------+---------+
         |                 |                |                   |
         v                 |                |                   |
  +------+------+          |                |                   |
  |  Datastore  |<---------+----------------+-------------------+
  | · LanceDB   |
  | · Ollama    |
  |   embeddings|
  +------+------+
         |
         v
  +-------------------------------+
  | data/sample-lancedb/rag-table |
  +-------------------------------+

  OUTPUT (Web):  streamed answer + 📄 source chunks → rendered in Streamlit chat
                 eval score + per-item reasoning → shown in results panel
  OUTPUT (CLI):  answer printed to stdout · eval score/pass-fail to terminal
```

### Components

| Component | File | Role |
|---|---|---|
| `RAGPipeline` | `src/rag_pipeline.py` | Orchestrates all stages; timing logs |
| `Indexer` | `src/impl/indexer.py` | Chunks docs via Docling or line-overlap |
| `Datastore` | `src/impl/datastore.py` | LanceDB vector store; parallel embedding via Ollama |
| `Retriever` | `src/impl/retriever.py` | LLM query rewriter → vector search |
| `ResponseGenerator` | `src/impl/response_generator.py` | Context-grounded answer generation & streaming |
| `Evaluator` | `src/impl/evaluator.py` | Parallel LLM-as-judge over Q&A pairs |
| `chat_history_store` | `src/util/chat_history_store.py` | SQLite-backed multi-session chat persistence |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com)** installed and running

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd LocalRag
```

### 2. One-command setup & launch

```bash
chmod +x start_localrag.sh
./start_localrag.sh
```

This script will:
1. Create a `.venv` virtual environment
2. Install all dependencies
3. Pull the default models (`qwen3:8b` + `nomic-embed-text`)
4. Launch the Streamlit app at `http://localhost:8501`

### 3. Manual setup (alternative)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull qwen3:8b
ollama pull nomic-embed-text

streamlit run streamlit_app.py
```

---

## ⚙️ Configuration

All settings are controlled via **environment variables** (with sensible defaults):

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `qwen3:8b` | Ollama chat model for generation & evaluation |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `MAX_TOKENS` | `300` | Max tokens per chunk (embedding model limit) |
| `DB_PATH` | `data/sample-lancedb` | Path to the LanceDB vector store |
| `EMBED_DIMENSIONS` | `768` | Embedding vector size (must match model) |

**Example — swap to a different model:**

```bash
LLM_MODEL=llama3.2 EMBED_MODEL=mxbai-embed-large streamlit run streamlit_app.py
```

---

## 📁 Document Support

| Type | Method |
|---|---|
| PDF | Docling structural chunking → OCR fallback |
| Images (PNG, JPG, TIFF, WEBP, …) | EasyOCR text extraction |
| Source code & plain text | Character-aware line-overlap chunking |
| Zip archives | Auto-unpacked then indexed |
| Any other file | Docling → raw text fallback |

---

## 🖥️ Streamlit UI

Open `http://localhost:8501` after launch.

**Sidebar**
- **New chat** / **🗑️ Delete chat** — create or permanently remove a conversation
- **Conversation selector** — switch between saved chat sessions
- **System Info** — live document/chunk counts and current model info
- **Reset index / Index documents** — manage the vector store
- **Upload documents** — drag & drop files or `.zip` archives to index
- **Clear chat** — wipe messages from the current session
- **Dynamic Evaluation** — paste or upload a JSON Q&A list and score your pipeline

**Main area**
- Streaming chat interface with source attribution (expander shows retrieved chunks per answer)

---

## 🖱️ CLI Usage

```bash
# Full pipeline: reset DB, index docs, then evaluate
python main.py run -p ./my_docs -f ./sample_data/eval/sample_questions.json

# Index a directory
python main.py add -p ./my_docs

# Ask a single question
python main.py query "What is the refund policy?"

# Reset the vector store
python main.py reset

# Evaluate with a custom Q&A file
python main.py evaluate -f ./my_eval.json
```

**Evaluation JSON format:**

```json
[
  {"question": "What is X?", "answer": "X is ..."},
  {"question": "How does Y work?", "answer": "Y works by ..."}
]
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `ollama` | Local LLM & embedding client |
| `lancedb` | Vector database |
| `docling` | PDF & document structural chunking |
| `easyocr` | OCR for images and scanned PDFs |
| `pypdfium2` | PDF rendering for OCR fallback |
| `pydantic` | Data validation |
| `filetype` | MIME-type detection |
| `charset-normalizer` | Encoding-safe text reading |

---

## 📂 Project Structure

```
LocalRag/
├── streamlit_app.py          # Streamlit web UI
├── main.py                   # CLI entry point
├── config.py                 # Environment-variable configuration
├── create_parser.py          # CLI argument parser
├── requirements.txt
├── start_localrag.sh         # One-command setup & launch script
└── src/
    ├── rag_pipeline.py       # Pipeline orchestrator
    ├── impl/
    │   ├── datastore.py      # LanceDB vector store
    │   ├── indexer.py        # Document chunking & indexing
    │   ├── retriever.py      # Query rewriting & vector search
    │   ├── response_generator.py  # LLM response & streaming
    │   └── evaluator.py      # LLM-as-judge evaluation
    ├── interface/            # Abstract base classes
    └── util/
        ├── chat_history_store.py  # SQLite chat persistence
        ├── extract_content.py     # OCR, PDF, text extraction
        ├── file_discovery.py      # Directory walker
        ├── invoke_ai.py           # Ollama chat wrapper
        └── extract_xml.py        # XML tag parser
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).
