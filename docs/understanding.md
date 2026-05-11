# Reddit Sentiment RAG System: Project Understanding

This project is a **multi-agent Retrieval-Augmented Generation (RAG) system** designed to analyze sentiments and opinions from Reddit data, specifically focused on the `r/ManchesterUnited` subreddit. It combines vector search, LLM-powered analysis, and a premium web dashboard.

## System Architecture

The system follows a five-stage pipeline: **Extraction**, **Embedding/Storage**, **Retrieval**, **Multi-Agent Analysis**, and **Visualization**.

```
[Reddit Data] → [ChromaDB Vectors] → [Hybrid Retriever] → [LangGraph Agents] → [Web Dashboard]
```

### 1. Data Extraction (Scraper)
- **Tooling**: Python with `requests` and `Cloudflare WARP`.
- **Logic**:
    - Uses Cloudflare WARP as a proxy to rotate IPs and avoid rate limiting.
    - Scrapes top posts and their associated comment trees.
    - **Authority Injection**: Automatically tags comments from moderators (`🛡️ [MODERATOR]`) and the original poster (`🔴 [OP/CREATOR]`) to preserve context of authority.
    - **Metadata Capture**: Records scores, flairs, dates, and URLs.
- **Output**: Structured JSON files stored in `rag_ready_data/`.

### 2. Vector Storage (ChromaDB)
- **Script**: `embed_to_chroma.py`
- **Embedding Model**: `all-MiniLM-L6-v2` (via Sentence Transformers).
- **Processing Logic**:
    - **Context-Aware Flattening**: Nested comments are flattened into individual documents. To maintain context, each reply includes a snippet of its parent comment.
    - **Deduplication**: Generates unique MD5 hashes based on content and metadata to prevent duplicate entries.
    - **Flair Metadata**: Post flair is stored alongside each document for filtering.
    - **Persistence**: Data is stored in a local ChromaDB collection named `reddit_sentiment`.
- **Database Path**: `chroma_db/`

### 3. Advanced Retrieval (Phase 2)
- **Module**: `retriever.py`
- **Strategies**:
    - **Multi-Query Retrieval**: Uses Gemini LLM to generate 4 query variations, executes each against ChromaDB, and fuses/deduplicates results.
    - **Metadata-Filtered Retrieval**: Supports filtering by score, flair, and document type.
    - **Hybrid Pipeline**: Combines multi-query + metadata filtering + score-based reranking for optimal results.

### 4. Multi-Agent Analysis (Phase 3 — LangGraph)
- **Module**: `agents.py`
- **Pipeline**: `retrieve → extract → analyze_sentiment → synthesize`
- **Specialized Agents**:
    - **Retriever Node**: Fetches relevant documents via the hybrid pipeline.
    - **Extractor Agent**: Pulls key facts, direct quotes, and points of agreement/disagreement.
    - **Sentiment Agent**: Evaluates emotional tone, confidence, dominant emotions, and sarcasm detection.
    - **Synthesizer Agent**: Combines everything into a Pydantic-validated structured JSON report.

### 5. Interactive Dashboard (Phase 4)
- **Backend**: FastAPI (`app.py`) serving REST API + static dashboard.
- **Frontend**: Premium single-page HTML/CSS/JS dashboard (`static/index.html`).
- **Design**: Dark mode, glassmorphism, animated orbs, gradient typography, micro-animations.
- **Features**:
    - Quick Search (simple RAG) and Deep Analysis (multi-agent) modes
    - Sentiment gauge, theme pills, quote cards, emotion breakdown bars
    - Analysis history tracking
    - Typewriter effect on summaries

---

## Repository Structure

| Path | Description |
| :--- | :--- |
| `config.py` | Centralized configuration (paths, models, API keys, helpers). |
| `schemas.py` | Pydantic models for structured sentiment output. |
| `generator.py` | Phase 1 — Basic RAG chain with Gemini LLM. |
| `retriever.py` | Phase 2 — Multi-query and hybrid retrieval strategies. |
| `agents.py` | Phase 3 — LangGraph multi-agent orchestration pipeline. |
| `app.py` | Phase 4 — FastAPI server with REST API endpoints. |
| `static/index.html` | Phase 4 — Premium glassmorphism web dashboard. |
| `embed_to_chroma.py` | Script for processing JSON data, generating embeddings, and querying. |
| `requirements.txt` | All project dependencies. |
| `.env` | Environment variables (API keys) — not committed. |
| `.env.example` | Template for `.env` configuration. |
| `rag_ready_data/` | Directory containing raw scraped JSON files from Reddit. |
| `chroma_db/` | Local persistent storage for the ChromaDB vector database. |
| `rough.ipynb` | Jupyter notebook used for scraping experiments and WARP configuration. |
| `.venv/` | Python virtual environment with dependencies. |

## Usage

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### To Embed New Data:
```bash
python embed_to_chroma.py
```

### To Launch the Dashboard:
```bash
python app.py
# Open http://localhost:8000
```

### To Query Programmatically:
```python
# Quick RAG query
from generator import query_rag
result = query_rag("What do fans think about the manager?")

# Full multi-agent analysis
from agents import run_analysis
report = run_analysis("What is the sentiment about team ownership?")
```
