"""
FastAPI Application — Phase 4: Dashboard & API
------------------------------------------------
Serves both the REST API and the premium static dashboard.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_chroma_collection, OPENROUTER_API_KEY

# ============================================================================
# APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Reddit Sentiment Intelligence",
    description="Multi-agent sentiment analysis for Reddit communities",
    version="1.0.0",
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
HISTORY_FILE = Path("./analysis_history.json")


# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================

class AnalyzeRequest(BaseModel):
    query: str
    mode: str = "deep"  # "deep" = multi-agent, "quick" = simple RAG


class QuickSearchRequest(BaseModel):
    query: str
    n_results: int = 5


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
    # Keep only last 50 entries
    history = history[-50:]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard HTML."""
    return FileResponse("static/index.html")


@app.get("/api/stats")
async def get_stats():
    """Get collection statistics."""
    try:
        collection = get_chroma_collection()
        count = collection.count()
        
        # Sample some metadata to get date range and flair distribution
        sample = collection.peek(limit=100)
        dates = []
        flairs = {}
        scores = []
        
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
            "api_key_configured": bool(OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_api_key_here"),
        }
    except Exception as e:
        return {"error": str(e), "total_documents": 0}


@app.post("/api/analyze")
async def analyze_sentiment(request: AnalyzeRequest):
    """Run full multi-agent sentiment analysis."""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_api_key_here":
        raise HTTPException(
            status_code=400,
            detail="OPENROUTER_API_KEY not configured. Set it in your .env file."
        )
    
    try:
        if request.mode == "quick":
            # Phase 1: Simple RAG chain
            from generator import query_rag
            result = query_rag(request.query, n_results=5)
            response = {
                "type": "quick",
                "query": request.query,
                "answer": result,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            # Phase 3: Full multi-agent pipeline
            from agents import run_analysis
            report = run_analysis(request.query)
            response = {
                "type": "deep",
                "query": request.query,
                "report": report,
                "timestamp": datetime.now().isoformat(),
            }
        
        # Save to history
        history = load_history()
        history_entry = {
            "query": request.query,
            "mode": request.mode,
            "timestamp": datetime.now().isoformat(),
            "sentiment": response.get("report", {}).get("overall_sentiment", "N/A") if request.mode == "deep" else "N/A",
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
    return {"history": list(reversed(history))}  # Most recent first


# ============================================================================
# STATIC FILES (must be mounted last to not override API routes)
# ============================================================================

# Create static directory if it doesn't exist
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
    api_status = "Configured" if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_api_key_here" else "Not set"
    print(f"  API Key:   {api_status}")
    print(f"\n{'=' * 60}\n")
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
