"""
FastAPI Application — Reddit Sentiment Intelligence
----------------------------------------------------
Serves the REST API (scrape, embed, analyze) and the premium static dashboard.
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
# Ensure config status is printed on startup
config.print_config_status()

# Import needed constants and settings
from config import get_chroma_collection, OPENROUTER_API_KEY, GOOGLE_API_KEY, USE_GOOGLE_STUDIO

app = FastAPI(
    title="Reddit Sentiment Intelligence",
    description="Multi-agent sentiment analysis for Reddit communities",
    version="2.0.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# History file path
HISTORY_FILE = config.BASE_DIR / "data" / "analysis_history.json"
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================

class AnalyzeRequest(BaseModel):
    query: str
    mode: str = "deep"  # "deep" = multi-agent, "quick" = simple RAG
    collection: str = "reddit_sentiment"


class EmbedRequest(BaseModel):
    subreddit: str


class ScrapeRequest(BaseModel):
    subreddit: str = ""
    post_limit: int = 25
    sort_by: str = "top"
    time_filter: str = "year"


class SubredditValidateRequest(BaseModel):
    subreddit: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_history() -> list:
    """Load analysis history from disk."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history: list):
    """Save analysis history to disk."""
    history = history[-50:]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


# ============================================================================
# API ENDPOINTS — DASHBOARD
# ============================================================================

@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard HTML."""
    return FileResponse("static/index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


# ============================================================================
# API ENDPOINTS — STATS
# ============================================================================

@app.get("/api/stats")
async def get_stats(collection_name: str = "reddit_sentiment"):
    """Get collection statistics."""
    try:
        collection = get_chroma_collection(collection_name)
        count = collection.count()

        sample = collection.peek(limit=100)
        dates, flairs, scores = [], {}, []

        if sample and sample.get("metadatas"):
            for meta in sample["metadatas"]:
                if meta.get("post_date"):
                    dates.append(meta["post_date"])
                flair = meta.get("flair", "Unknown")
                flairs[flair] = flairs.get(flair, 0) + 1
                scores.append(meta.get("comment_score", 0))

        return {
            "total_documents": count,
            "date_range": {
                "earliest": min(dates) if dates else "N/A",
                "latest": max(dates) if dates else "N/A",
            },
            "flair_distribution": flairs,
            "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "api_key_configured": bool(GOOGLE_API_KEY) if USE_GOOGLE_STUDIO else bool(OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_api_key_here"),
            "provider": "Google AI Studio" if USE_GOOGLE_STUDIO else "OpenRouter",
        }
    except Exception as e:
        return {"error": str(e), "total_documents": 0}


# ============================================================================
# API ENDPOINTS — SCRAPING
# ============================================================================

@app.post("/api/validate_subreddit")
async def validate_sub(request: SubredditValidateRequest):
    """Validate if a subreddit exists and return its metadata."""
    from scraper.reddit_scraper import validate_subreddit
    result = validate_subreddit(request.subreddit)
    if not result["success"]:
        return JSONResponse(status_code=400, content=result)
    return result


@app.post("/api/scrape")
async def start_scrape(request: ScrapeRequest):
    """Start a Reddit scraping job in the background."""
    from scraper.reddit_scraper import scrape_subreddit, get_scrape_status

    status = get_scrape_status()
    if status["running"]:
        raise HTTPException(status_code=409, detail="A scrape job is already running")

    def run_scrape():
        scrape_subreddit(
            subreddit=request.subreddit,
            post_limit=request.post_limit,
            sort_by=request.sort_by,
            time_filter=request.time_filter,
        )

    thread = threading.Thread(target=run_scrape, daemon=True)
    thread.start()

    return {"status": "started", "message": f"Scraping r/{request.subreddit} ({request.post_limit} posts)"}


@app.get("/api/scrape/status")
async def scrape_status():
    """Get the current scrape job status."""
    from scraper.reddit_scraper import get_scrape_status
    return get_scrape_status()


# ============================================================================
# API ENDPOINTS — EMBEDDING
# ============================================================================

@app.post("/api/embed")
async def start_embed(request: EmbedRequest):
    """Trigger embedding of scraped data into ChromaDB."""
    from rag.embedder import embed_to_chromadb, get_embed_status

    status = get_embed_status()
    if status["running"]:
        raise HTTPException(status_code=409, detail="An embed job is already running")

    def run_embed():
        embed_to_chromadb(request.subreddit)

    thread = threading.Thread(target=run_embed, daemon=True)
    thread.start()

    return {"status": "started", "message": f"Embedding data for r/{request.subreddit} into ChromaDB"}


@app.get("/api/embed/status")
async def embed_status():
    """Get the current embed job status."""
    from rag.embedder import get_embed_status
    return get_embed_status()


@app.get("/api/data/count")
async def data_file_count(subreddit: str = ""):
    """Count how many JSON files are ready for embedding."""
    from rag.embedder import count_json_files
    return {"count": count_json_files(subreddit)}


# ============================================================================
# API ENDPOINTS — ANALYSIS
# ============================================================================

@app.post("/api/analyze")
def analyze_sentiment(request: AnalyzeRequest):
    """Run sentiment analysis — both modes return {mode, report: UnifiedAnalysisReport}."""
    if USE_GOOGLE_STUDIO:
        if not GOOGLE_API_KEY:
            raise HTTPException(status_code=400, detail="GOOGLE_API_KEY not configured. Set it in your .env file.")
    else:
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_api_key_here":
            raise HTTPException(status_code=400, detail="OPENROUTER_API_KEY not configured. Set it in your .env file.")

    try:
        if request.mode == "quick":
            from rag.generator import query_rag
            report = query_rag(request.query, n_results=8, collection_name=request.collection)
        else:
            from agents.pipeline import run_analysis
            report = run_analysis(request.query, collection_name=request.collection)

        # Uniform response envelope — both modes identical shape
        response = {
            "mode": request.mode,
            "report": report,
        }

        # Save to history with enriched fields
        history = load_history()
        verdict = report.get("verdict", {})
        history_entry = {
            "query": request.query,
            "mode": request.mode,
            "timestamp": datetime.now().isoformat(),
            "sentiment": verdict.get("overall_sentiment", "N/A"),
            "confidence": verdict.get("confidence", 0.0),
            "net_sentiment_score": verdict.get("net_sentiment_score", 0),
        }
        history.append(history_entry)
        save_history(history)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_history():
    """Get recent analysis history."""
    history = load_history()
    return {"history": list(reversed(history))}

@app.get("/api/collections")
async def get_collections():
    """Get list of available subreddits (Chroma collections)."""
    import chromadb
    from config import CHROMA_PERSIST_DIR
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        collections = client.list_collections()
        deduped = {}
        for c in collections:
            name = c.name if hasattr(c, 'name') else c
            coll = client.get_collection(name)
            count = coll.count()
            if count > 0:
                key = name.lower()
                if key not in deduped or count > deduped[key]["docs"]:
                    deduped[key] = {"name": name, "docs": count}
        return {"collections": list(deduped.values())}
    except Exception as e:
        return {"error": str(e), "collections": []}


# ============================================================================
# STATIC FILES (must be mounted last to not override API routes)
# ============================================================================

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("REDDIT SENTIMENT INTELLIGENCE")
    print("=" * 60)
    print(f"\n  Dashboard: http://localhost:8000")
    print(f"  API Docs:  http://localhost:8000/docs")
    if USE_GOOGLE_STUDIO:
        api_status = "Configured (Google)" if GOOGLE_API_KEY else "Not set"
    else:
        api_status = "Configured (OpenRouter)" if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_api_key_here" else "Not set"
    print(f"  API Key:   {api_status}")
    print(f"\n{'=' * 60}\n")

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
