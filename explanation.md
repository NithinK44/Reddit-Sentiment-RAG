# Technical Explanation: Reddit Sentiment RAG

This document outlines the high-level technical architecture and data flow of the system for use in technical presentations and reviews.

## 1. System Architecture: The Multi-Agent RAG
The project implements a **Retrieval-Augmented Generation (RAG)** architecture, further enhanced by a **Multi-Agent Pipeline** (orchestrated via LangGraph). This separates the concerns of raw data retrieval, fact extraction, emotional tone analysis, and final reporting.

---

## 2. Advanced Retrieval Engine (Agentic Hybrid Search)
This is the "brain" of the search process, solving the common RAG problem where simple vector search misses specific keywords.

### A. Agentic Routing
Before searching, a **Router Agent** (LLM-based) analyzes the query to decide on the most efficient retrieval path:
- **Path 1: Semantic (Dense Vector)**: Uses `all-MiniLM-L6-v2` embeddings + Cosine Similarity. Best for abstract concepts.
- **Path 2: Hybrid (Vector + Lexical)**: Combines Vector search with **BM25 Keyword Matching**. Best for specific entities (e.g., "Casemiro").

### B. Reciprocal Rank Fusion (RRF)
When Hybrid search is used, we apply **RRF** to combine results. This is a robust mathematical formula:
`Score = Σ 1 / (k + rank)`
It ensures that documents ranked highly by *both* algorithms are prioritized without needing to normalize the scores of different search types.

### C. Transparency and Exploration
- **Router Visibility**: The chosen retrieval strategy (Semantic vs Hybrid) is surfaced dynamically in the UI badges, making the agentic decision transparent to the user.
- **Data Exploration (Word Cloud)**: After data is embedded, the backend analyzes the knowledge base to extract frequent, significant terms, generating a dynamic word cloud. This guides users on what questions might yield the best insights before they even type a query.

---

## 3. The Data Pipeline (ETL)
1.  **Extraction**: `scraper/reddit_scraper.py` uses the Reddit API to fetch structured threads.
2.  **Transformation**: `rag/embedder.py` performs **Structural Splitting**. Instead of character-based splitting, it uses the natural hierarchy of comments. 
    - **Contextual Injection**: Child comments are "wrapped" with parent context to preserve conversation flow.
3.  **Loading**: Documents are stored in **ChromaDB** using **HNSW** (Hierarchical Navigable Small World) for fast approximate nearest-neighbor search.

---

## 4. Multi-Agent Analysis Pipeline (LangGraph)
Once documents are retrieved, they pass through a directed graph of agents:
1.  **Extractor Node**: Isolates raw facts, entities, and direct quotes.
2.  **Sentiment Node**: Analyzes emotional tone, intensity, and sarcasm/irony.
3.  **Synthesizer Node**: Aggregates the previous outputs into a structured JSON schema for UI visualization.

---

## 5. Metadata-Driven Consensus
The system prioritizes "Community Truth" over "Individual Noise" by leveraging Reddit-specific signals:
- **Upvote Score Filtering**: Discards downvoted/low-quality noise.
- **Authority Weighting**: Gives higher priority to Moderators and Original Posters (OP).

---

## 6. Technical Stack Summary
- **LLM**: Gemini-1.5-Flash (Low latency / High reasoning).
- **Embeddings**: `all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Vector DB**: ChromaDB.
- **Keyword Algorithm**: BM25Okapi.
- **Backend**: FastAPI (Python).
- **Frontend**: Vanilla JS + CSS (Glassmorphism design).
