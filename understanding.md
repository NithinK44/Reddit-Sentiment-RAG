# Reddit Sentiment RAG System: Project Understanding

This project is a Retrieval-Augmented Generation (RAG) system designed to analyze sentiments and opinions from Reddit data, specifically focused on the `r/ManchesterUnited` subreddit.

## 🚀 System Architecture

The system follows a three-stage pipeline: **Extraction**, **Embedding/Storage**, and **Retrieval**.

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
    - **Persistence**: Data is stored in a local ChromaDB collection named `reddit_sentiment`.
- **Database Path**: `chroma_db/`

### 3. Retrieval & Analysis
- **Functionality**: Provides a semantic search interface to query the database.
- **Features**:
    - **Semantic Search**: Finds relevant posts and comments based on query meaning rather than just keywords.
    - **Quality Filtering**: Supports filtering results by a minimum comment score to ensure high-quality retrieval.

---

## 📁 Repository Structure

| Path | Description |
| :--- | :--- |
| `rag_ready_data/` | Directory containing raw scraped JSON files from Reddit. |
| `chroma_db/` | Local persistent storage for the ChromaDB vector database. |
| `embed_to_chroma.py` | Main script for processing JSON data, generating embeddings, and querying. |
| `rough.ipynb` | Jupyter notebook used for scraping experiments and WARP configuration. |
| `.venv/` | Python virtual environment with dependencies (Chromadb, Transformers, etc.). |

## 🛠️ Usage

### To Embed New Data:
```bash
python embed_to_chroma.py
```

### To Query the System:
The `query_sentiment` function in `embed_to_chroma.py` can be used to retrieve relevant snippets:
```python
results = query_sentiment("What is the sentiment about team ownership?", n_results=5, min_score=10)
```
