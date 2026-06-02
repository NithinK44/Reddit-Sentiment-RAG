# Reddit Sentiment Intelligence (RAG) 🚀

An AI-powered multi-agent system that scrapes, indexes, and deeply analyzes Reddit community sentiment. Built on a **Retrieval-Augmented Generation (RAG)** backbone with specialized LangGraph agents, hybrid retrieval, and real-time observability.

---

## ✨ Features

- **Verified Scraping** — Real-time subreddit existence validation with a confirmation workflow to prevent invalid scrapes.
- **Query Intelligence** — A single combined LLM call replaces three separate calls (orchestration router + retrieval router + query rewriter) for maximum latency efficiency.
- **Advanced RAG Architecture** — Hybrid retrieval pipeline combining:
  - Multi-query semantic expansion (parallelized)
  - SQLite FTS5 keyword search
  - Reciprocal Rank Fusion (RRF)
  - Cross-Encoder semantic reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **4-Agent LangGraph Pipeline** — Modular agents with a reflection & self-correction loop:
  1. **Retriever** — Fetches top-15 documents from ChromaDB + SQLite FTS5
  2. **Extractor** — Domain-agnostic positive/negative fact extraction
  3. **Sentiment Analyst** — Emotion map, controversy score, and sentiment trajectory
  4. **Synthesizer** — Validates and produces a typed `UnifiedAnalysisReport` JSON
  - Automatic **Reflection Node** retries up to 2× on schema validation failures before falling back gracefully.
- **Multi-Model Support** — Tiered model routing:
  - Standard: `gemini-3.1-flash-lite`
  - Deep Analysis: `gemini-3-flash-preview`
  - Upgraded Analysis: `gemini-3.5-flash`
  - Fallback: `gemma-4-31b-it` (via Google AI Studio or OpenRouter)
- **Dual Provider Engine** — Toggle between **Google AI Studio (Gemini)** and **OpenRouter** with a single env var.
- **Automatic Rate-Limit Retry** — Both sync and async LLM clients retry up to 3× with 60-second backoff on 429 errors, with per-job notifications and telemetry.
- **FastMCP Server** — MCP integration for Claude Desktop: expose scraping, analysis, and DB status as native tools.
- **Persistent Storage** — ChromaDB (vector store) + SQLite with FTS5 full-text search, dual-table sync via triggers.
- **Job Tracking** — Async job store with streaming SSE events, per-job model usage, retry/fallback counters.
- **Advanced Observability** — LangSmith tracing with pre-seeded tracer contexts to ensure unified parent-child spans across async stream-generators and parallelized retrievers.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Orchestrator (Query Intelligence)
    │   Single LLM call: route + rewrite + retrieval strategy + query variants
    │
    ├─► [route=scraper] → Scraper Agent → Processor Agent (embed)
    │
    └─► RAG Synthesis Pipeline (LangGraph)
            │
            ├─ Node 1: Retrieve   (ChromaDB vector + SQLite FTS5 + RRF + CrossEncoder)
            ├─ Node 2: Extract    (Positive/Negative fact extraction)
            ├─ Node 3: Sentiment  (Emotion map, controversy, trajectory)
            └─ Node 4: Synthesize → Validate → [Reflect & Fix ×2] → UnifiedAnalysisReport
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Agent Framework** | LangGraph, LangChain |
| **API Server** | FastAPI + Uvicorn |
| **MCP Integration** | FastMCP |
| **Vector Store** | ChromaDB (cosine similarity) |
| **Keyword Search** | SQLite FTS5 (BM25 ranking) |
| **Embedding Model** | `all-MiniLM-L6-v2` (Sentence Transformers) |
| **Reranking Model** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLMs (Google)** | Gemini 3.1 Flash Lite, Gemini 3 Flash, Gemini 3.5 Flash, Gemma 4 31B |
| **LLMs (OpenRouter)** | Any OpenRouter-compatible model (default: `gemma-4-31b-it:free`) |
| **Observability** | LangSmith |
| **Frontend** | Vanilla HTML / CSS / JS |
| **Scraping** | Playwright (headless browser) |

---

## 📐 Development Workflow (SDD)

This project follows a **Spec-Driven Development** protocol. All features are first defined in specifications before implementation:
- **Master Spec**: [MASTER_SPEC.md](.agents/specs/MASTER_SPEC.md)
- **Feature Specs**: Located in `.agents/specs/features/`

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Reddit-Sentiment-RAG

# Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser (for scraping)
playwright install chromium
```

### 2. Configuration

Create a `.env` file in the root directory:

```env
# LangSmith Tracing (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=Reddit

# Google AI Studio (Gemini) — primary provider
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=gemini-3.1-flash-lite
DEEP_ANALYSIS_MODEL=gemini-3-flash-preview
UPGRADED_ANALYSIS_MODEL=gemini-3.5-flash
FALLBACK_MODEL=gemma-4-31b-it

# OpenRouter (alternative provider)
OPENROUTER_API_KEY=your_openrouter_api_key
LLM_MODEL=google/gemma-4-31b-it:free

# Provider Toggle — set True for Google AI Studio, False for OpenRouter
USE_GOOGLE_STUDIO=True

# Force fallback model for deep analysis agents (avoids quota burn on expensive models)
FORCE_FALLBACK_FOR_DEEP=True

# Scraper concurrency
SCRAPER_MAX_WORKERS=8
```

### 3. Run the App

#### Web Interface (FastAPI):
```bash
python app.py
```
Visit `http://localhost:8000` to use the web UI.

#### MCP Server (for Claude Desktop):
```bash
python mcp_server.py
```
See [`claude_desktop_config.example.json`](claude_desktop_config.example.json) for setup instructions.

---

## 📂 Project Structure

```
Reddit-Sentiment-RAG/
├── app.py                   # FastAPI application (web UI + API endpoints)
├── mcp_server.py            # FastMCP server (Claude Desktop integration)
├── config.py                # Centralized config, LLM factory, ChromaDB helpers
│
├── agents/                  # Multi-agent LangGraph orchestration
│   ├── orchestrator.py      # Query Intelligence + routing entry point
│   ├── rag_agent.py         # 4-node LangGraph pipeline (Retrieve → Extract → Sentiment → Synthesize)
│   ├── schemas.py           # Pydantic schemas (UnifiedAnalysisReport, etc.)
│   ├── scraper_agent.py     # LangGraph scraper agent
│   ├── processor_agent.py   # Embedding/processing agent
│   └── tools/               # Agent tools (scrape, embed, search)
│
├── rag/                     # RAG pipeline components
│   ├── retriever.py         # Hybrid retrieval (Vector + FTS5 + RRF + CrossEncoder)
│   ├── embedder.py          # Document ingestion & embedding
│   └── generator.py         # Context formatting helpers
│
├── core/                    # Shared infrastructure
│   ├── db.py                # SQLite + FTS5 schema, upsert, keyword search
│   ├── job_store.py         # Async job store with SSE event streaming
│   ├── llm_utils.py         # JSON parsing helpers
│   ├── logging_config.py    # Logging setup
│   └── manifest_manager.py  # Scrape manifest tracking
│
├── scraper/                 # Reddit data collection
│   └── reddit_scraper.py    # Playwright-based scraper with [OP]/[MOD] tagging
│
├── prompts/                 # LLM prompt templates
│   ├── query_intelligence.txt
│   ├── extractor.txt
│   ├── sentiment.txt
│   ├── synthesizer.txt
│   ├── reflection.txt
│   └── ...
│
├── static/                  # Frontend (HTML/CSS/JS)
├── data/                    # Local storage (SQLite DB, ChromaDB, scraped JSON)
└── .agents/                 # SDD specs and agent skills
```

---

## 🔌 MCP Tools (Claude Desktop)

| Tool | Description |
|---|---|
| `analyze_reddit_sentiment(query)` | Runs the full multi-agent pipeline. Automatically decides whether to scrape fresh data or use the existing database. |
| `get_reddit_database_stats(collection_name)` | Returns the document count for a given ChromaDB collection. |

---

## 📊 Analysis Output Schema

Every analysis returns a structured `UnifiedAnalysisReport` with:

- **`verdict`** — Overall sentiment, confidence score, net sentiment score
- **`executive_summary`** — Natural language summary
- **`sentiment_distribution`** — Positive / Negative / Neutral percentages (must sum to 100)
- **`positive_signals`** / **`negative_signals`** — Top themes, quotes, and percentages
- **`emotion_map`** — Detected emotions with intensities
- **`key_entities`** — Named entities with sentiment context (min. 3)
- **`actionable_insights`** — Per-team recommendations (product, marketing, support)
- **`meta`** — Report ID, timestamp, model used, fallback/retry info, data freshness dates

---

*Created with ❤️ by the Antigravity Team*
