# Project Understanding & Architecture

This document provides a brief overview of the codebase and how the different components interact.

## 🏗️ Directory Structure

```text
Reddit-Sentiment-RAG/
├── agents/             # Multi-agent logic (LangGraph)
│   ├── pipeline.py     # Orchestration of specialized agents
│   └── schemas.py      # Pydantic models for structured output
├── rag/                # Core RAG components
│   ├── embedder.py     # JSON -> Vector DB processing
│   ├── retriever.py    # Agentic Router, Hybrid Search & RRF
│   └── generator.py    # Quick Mode RAG implementation
├── scraper/            # Data collection
│   └── reddit_scraper.py # Reddit JSON API scraper
├── static/             # Frontend UI (HTML/JS/CSS)
├── data/               # Persistent storage (Git-ignored)
│   ├── chroma_db/      # Vector database files
│   └── rag_ready_data/ # Raw JSON posts
├── app.py              # Main FastAPI server & entry point
├── config.py           # Centralized configuration & helpers
└── requirements.txt    # Project dependencies
```

## 🔄 Data Flow

1.  **Scrape**: `scraper/reddit_scraper.py` fetches posts from a subreddit and saves them as JSON files in `data/rag_ready_data/`.
2.  **Embed**: `rag/embedder.py` loads these JSON files, chunks the text (keeping parent-child context), and stores them in ChromaDB (`data/chroma_db/`).
3.  **Analyze**:
    *   **Quick Mode**: Uses `rag/generator.py` for a standard RAG response, now enhanced with the Agentic Router.
    *   **Deep Mode**: Uses `agents/pipeline.py` (LangGraph) to run a 4-stage pipeline:
        *   `Retrieve`: Fetches docs via the **Smart Router** in `rag/retriever.py`.
        *   `Extract`: Pulls facts and quotes.
        *   `Analyze`: Evaluates sentiment and emotions.
        *   `Synthesize`: Compiles a structured JSON report.

## 🧠 Advanced Retrieval Pipeline (`retriever.py`)

The system uses a state-of-the-art **Agentic Retrieval** strategy:

1.  **Agentic Router**: Every query is first analyzed by an LLM to decide between two paths:
    *   `SEMANTIC`: Pure vector search for abstract/conceptual queries.
    *   `HYBRID`: Combined keyword + vector search for entity-specific queries (e.g., player names).
2.  **Hybrid Search (BM25 + Vector)**: Integrates lexical matching (BM25) for high precision on names and dense vector search for high recall on intent.
3.  **Reciprocal Rank Fusion (RRF)**: Mathematically fuses the results from different retrieval paths into a single optimized list.
4.  **Consensus Filter**: Uses Reddit's metadata (net upvote scores) to prioritize authoritative community opinions over "noise."

## ⚙️ Core Configuration (`config.py`)
This file is the single source of truth for:
- Path definitions (`BASE_DIR`, `DATA_DIR`).
- LLM initialization (Google vs OpenRouter).
- Shared constants (Embedding models, search limits).
- Shared utilities (Get Chroma collection, Get LLM).

## 🎓 Academic Features
This project demonstrates several advanced AI and Information Retrieval (IR) concepts:
- **Modular Multi-Agent Systems**: Using LangGraph to separate concerns (Extraction vs. Analysis).
- **Hybrid Retrieval**: Solving the "Vocabulary Mismatch" problem in RAG.
- **Agentic Decision Making**: Using an LLM as a router for system optimization.
- **Metadata Re-ranking**: Incorporating domain-specific social signals (Reddit scores) into search relevance.
