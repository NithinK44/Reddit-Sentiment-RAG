# Reddit Sentiment Intelligence (RAG) 🚀

An AI-powered multi-agent system designed to scrape, index, and analyze sentiments from Reddit communities. Built with a focus on deep qualitative analysis using Retrieval-Augmented Generation (RAG) and specialized AI agents.

![Premium Dashboard Mockup](static/index.html) <!-- Placeholder for image if available -->

## ✨ Features
- **Verified Scraping**: NEW! Real-time subreddit existence validation and confirmation workflow to prevent invalid scrapes.
- **Smart Data Collection**: Extract post and comment trees from any subreddit with authority injection ([OP] and [MOD] tags).
- **Advanced RAG Architecture**: Hybrid retrieval combining multi-query expansion, metadata filtering (flair, score), and vector search via ChromaDB.
- **Multi-Agent Orchestration**: Specialized LangGraph agents for extraction, sentiment evaluation, and report synthesis.
- **Dual Engine Support**: Seamlessly toggle between **Google Gemini (AI Studio)** and **DeepSeek/OpenRouter**.
- **Modern Dashboard**: A glassmorphic, real-time UI for monitoring scrapes, vector builds, and performing deep analysis.

## 🛠️ Tech Stack
- **Backend**: FastAPI, LangChain, LangGraph
- **Vector Store**: ChromaDB
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLMs**: Google Gemini 1.5 Flash (Optimized for Gemini 3 Flash)
- **Frontend**: Vanilla HTML/JS/CSS (Premium Glassmorphism)

## 📐 Development Workflow (SDD)
This project follows a **Spec-Driven Development** protocol. All features are first defined in specifications before implementation:
- **Master Spec**: [MASTER_SPEC.md](.agents/specs/MASTER_SPEC.md)
- **Feature Specs**: Located in `.agents/specs/features/`
- **Tracking**: Status tracked in [TODO.md](TODO.md)

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd Reddit-Sentiment-RAG

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory:
```env
# AI Providers
GOOGLE_API_KEY=your_google_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

# Optional: Toggle provider (default is False/OpenRouter)
USE_GOOGLE_STUDIO=True
```

### 3. Run the App
```bash
python app.py
```
Visit `http://localhost:8000` to access the dashboard.

## 📂 Project Structure
- `scraper/`: Reddit data collection logic.
- `rag/`: Embedding, retrieval, and generation logic.
- `agents/`: Multi-agent LangGraph orchestration.
- `data/`: Local storage for JSON data and ChromaDB.
- `static/`: Frontend dashboard files.

---
*Created with ❤️ by the Antigravity Team*
