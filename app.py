import json
import os
import re
import threading
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb

from core.db import get_analysis_history, save_analysis_history, get_documents_paginated

logger = logging.getLogger(__name__)

import config
# Ensure config status is printed on startup
config.print_config_status()

# Import needed constants and settings
from config import get_chroma_collection, OPENROUTER_API_KEY, GOOGLE_API_KEY, USE_GOOGLE_STUDIO, get_llm
from core.job_store import analysis_jobs
import asyncio

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
    strategy: str = "agentic"  # "agentic", "semantic", "hybrid"


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

_history_lock = threading.Lock()
_analysis_semaphore = threading.Semaphore(2)

def load_history() -> list:
    """Load analysis history from SQLite database (chronological: oldest first)."""
    hist = get_analysis_history(limit=50)
    return list(reversed(hist))


def save_history(history: list):
    """Save analysis history to SQLite database."""
    if not history:
        return
    # Save the most recent entry
    latest = history[-1]
    save_analysis_history(
        query=latest.get("query"),
        mode=latest.get("mode"),
        timestamp=latest.get("timestamp"),
        sentiment=latest.get("sentiment"),
        confidence=latest.get("confidence"),
        net_sentiment_score=latest.get("net_sentiment_score"),
        report_id=latest.get("report_id"),
        data_freshness=latest.get("data_freshness")
    )


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
# API ENDPOINTS — WORD CLOUD
# ============================================================================

STOPWORDS = {"the", "and", "to", "of", "a", "in", "for", "is", "on", "that", "it", "with", "as", "was", "this", "but", "they", "are", "have", "be", "not", "we", "you", "at", "from", "or", "by", "an", "if", "my", "so", "all", "about", "can", "has", "do", "what", "just", "their", "like", "there", "out", "would", "up", "who", "more", "when", "some", "one", "them", "which", "will", "your", "than", "me", "how", "he", "been", "only", "no", "get", "because", "people", "even", "now", "any", "other", "very", "also", "then", "into", "could", "much", "think", "see", "make", "really", "know", "good", "time", "well", "way", "why", "did", "were", "had", "should", "over", "those", "these", "where", "its", "i", "it's", "don't", "i'm", "that's", "can't"}

@app.get("/api/wordcloud")
async def get_wordcloud(collection_name: str = "reddit_sentiment"):
    """Get word frequencies for a word cloud."""
    try:
        collection = get_chroma_collection(collection_name)
        # Fetch up to 200 documents to generate the word cloud quickly
        result = collection.get(limit=200, include=["documents"])
        documents = result.get("documents", [])
        
        if not documents:
            return {"words": []}
            
        text = " ".join(documents).lower()
        words = re.findall(r'\b[a-z]{3,}\b', text)
        filtered_words = [w for w in words if w not in STOPWORDS]
        
        counts = Counter(filtered_words)
        # Get top 100 words to send to LLM
        top_common = counts.most_common(100)
        candidate_words = [word for word, count in top_common]
        
        # Call LLM to filter words
        try:
            llm = get_llm(temperature=0.1)
            prompt = f"""
            You are an expert at entity recognition. Filter the following list of words extracted from Reddit posts.
            Return ONLY the words that are reasonable entities (such as people, organizations, products, locations, specific technologies, games, movies, or distinct concepts).
            Exclude:
            - Common filler words or stop words.
            - Purely emotional words (e.g., 'good', 'bad', 'great', 'terrible', 'love', 'hate', 'awesome').
            - Generic verbs or adjectives (e.g., 'running', 'big', 'small', 'make', 'think').
            - Generic nouns that don't represent specific entities or concepts (e.g., 'thing', 'way', 'time').
            
            Input words: {json.dumps(candidate_words)}
            
            Return the filtered list as a JSON array of strings. Do not include any other text or code blocks, just the JSON array.
            """
            
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            from core.llm_utils import parse_llm_json_list
            filtered_entities = parse_llm_json_list(content)
            
            # Filter the top words to only include those approved by the LLM
            # and limit to top 40 for display
            top_words = []
            for word, count in top_common:
                if word in filtered_entities:
                    top_words.append({"text": word, "value": count})
                    if len(top_words) >= 40:
                        break
                        
            # If no entities found (unlikely), fallback to top 20 simple words
            if not top_words:
                top_words = [{"text": word, "value": count} for word, count in counts.most_common(20)]
                        
        except Exception as llm_err:
            logger.warning(f"LLM filtering failed: {llm_err}. Falling back to simple frequency.")
            # Fallback to simple frequency if LLM fails
            top_words = [{"text": word, "value": count} for word, count in counts.most_common(40)]
        
        return {"words": top_words}
    except Exception as e:
        return {"error": str(e), "words": []}


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
async def start_analyze_sentiment(request: AnalyzeRequest):
    """Start sentiment analysis in background (Quick or Deep mode)."""
    if USE_GOOGLE_STUDIO:
        if not GOOGLE_API_KEY:
            raise HTTPException(status_code=400, detail="GOOGLE_API_KEY not configured. Set it in your .env file.")
    else:
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_api_key_here":
            raise HTTPException(status_code=400, detail="OPENROUTER_API_KEY not configured. Set it in your .env file.")

    job_id = analysis_jobs.create_job()

    def run_analysis_task():
        # Semaphore protects concurrent LLM and memory usage
        with _analysis_semaphore:
            try:
                if request.mode == "quick":
                    from rag.generator import query_rag
                    report = query_rag(request.query, n_results=8, collection_name=request.collection, strategy=request.strategy)
                else:
                    from agents.orchestrator import run_orchestrated_analysis
                    report = run_orchestrated_analysis(request.query, collection_name=request.collection, job_id=job_id)

                # Save to history with enriched fields
                history = load_history()
                verdict = report.get("verdict", {})
                report_id = report.get("meta", {}).get("report_id")
                
                history_entry = {
                    "query": request.query,
                    "mode": request.mode,
                    "timestamp": datetime.now().isoformat(),
                    "sentiment": verdict.get("overall_sentiment", "N/A"),
                    "confidence": verdict.get("confidence", 0.0),
                    "net_sentiment_score": verdict.get("net_sentiment_score", 0),
                    "report_id": report_id,
                    "data_freshness": report.get("meta", {}).get("data_freshness", {})
                }
                history.append(history_entry)
                save_history(history)

                # Cache the full report
                if report_id:
                    reports_dir = config.BASE_DIR / "data" / "reports"
                    reports_dir.mkdir(parents=True, exist_ok=True)
                    with open(reports_dir / f"{report_id}.json", "w", encoding="utf-8") as f:
                        json.dump(report, f, indent=2, ensure_ascii=False)

                # Wrap final response
                final_response = {
                    "mode": request.mode,
                    "report": report,
                }
                
                job = analysis_jobs.get_job(job_id)
                if job:
                    job.complete(final_response)

            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                job = analysis_jobs.get_job(job_id)
                if job:
                    job.fail(str(e))

    thread = threading.Thread(target=run_analysis_task, daemon=True)
    thread.start()
    
    return {"status": "started", "job_id": job_id, "message": "Analysis started"}


@app.get("/api/analyze/status/{job_id}")
async def get_analyze_status(job_id: str):
    job = analysis_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = job.get_status()
    if job.status == "completed":
        status["result"] = job.result
    return status


@app.get("/api/analyze/stream/{job_id}")
async def stream_analyze_events(request: Request, job_id: str):
    job = analysis_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        last_idx = 0
        while True:
            if await request.is_disconnected():
                break

            events = job.get_events_since(last_idx)
            for e in events:
                yield f"data: {json.dumps(e)}\n\n"
                last_idx += 1
                
            if job.status in ["completed", "failed"]:
                break
                
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/report/{report_id}")
async def get_report(report_id: str):
    """Get a cached analysis report by ID."""
    reports_dir = config.BASE_DIR / "data" / "reports"
    report_file = reports_dir / f"{report_id}.json"
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                report = json.load(f)
                return {"mode": report.get("meta", {}).get("mode", "deep"), "report": report}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load report: {e}")
    raise HTTPException(status_code=404, detail="Report not found")

@app.get("/api/report/{report_id}/download")
async def download_report(report_id: str, format: str = "markdown"):
    """Download a cached analysis report as a formatted Markdown or HTML file."""
    reports_dir = config.BASE_DIR / "data" / "reports"
    report_file = reports_dir / f"{report_id}.json"
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                report = json.load(f)
            
            verdict = report.get("verdict", {})
            meta = report.get("meta", {})
            exec_sum = report.get("executive_summary", "")
            
            # Make the executive summary a bit more brief (keep first 2 sentences if longer)
            if exec_sum:
                sentences = re.split(r'(?<=[.!?])\s+', exec_sum)
                if len(sentences) > 2:
                    exec_sum = " ".join(sentences[:2])
                    
            insights = report.get("actionable_insights", {})
            query = meta.get("query", "Reddit Sentiment Analysis")

            if format.lower() == "html":
                # Sentiment Distribution & Emotion Map
                dist = report.get("sentiment_distribution", {}) or {}
                pos_pct = dist.get("positive_pct", 0) or 0
                neg_pct = dist.get("negative_pct", 0) or 0
                neu_pct = dist.get("neutral_pct", 0) or 0
                dominant_emotions = dist.get("dominant_emotions", []) or []
                emap = dist.get("emotion_map", {}) or {}

                emotions_html = "".join([f'<span class="emotion-tag">{e}</span>' for e in dominant_emotions])

                emotion_colors = {
                    "anger": "#ef4444", "frustration": "#f97316", "hope": "#10b981",
                    "satisfaction": "#22d3ee", "disappointment": "#8b5cf6", "excitement": "#f59e0b",
                    "sarcasm": "#ec4899", "resignation": "#6b7280"
                }
                emotion_rows_html = ""
                for emotion in ["anger", "frustration", "hope", "satisfaction", "disappointment", "excitement", "sarcasm", "resignation"]:
                    val = emap.get(emotion, 0) or 0
                    color = emotion_colors.get(emotion.lower(), "#8b5cf6")
                    emotion_rows_html += f"""
                    <div class="emotion-bar-row">
                        <div class="emotion-bar-label">{emotion}</div>
                        <div class="emotion-bar-track">
                            <div class="emotion-bar-fill" style="width: {val}%; background-color: {color};"></div>
                        </div>
                        <div class="emotion-bar-val">{val:.0f}%</div>
                    </div>
                    """

                dist_and_map_html = f"""
        <section>
            <h2>Sentiment Distribution & Emotion Map</h2>
            <div class="grid-2col">
                <div class="report-card">
                    <h3 style="font-size: 1.1rem; font-weight: 600; color: var(--accent-orange); margin-bottom: 1rem;">Sentiment Share</h3>
                    <div class="breakdown-bar">
                        <div class="bar-positive" style="width: {pos_pct}%"></div>
                        <div class="bar-negative" style="width: {neg_pct}%"></div>
                        <div class="bar-neutral" style="width: {neu_pct}%"></div>
                    </div>
                    <div class="breakdown-labels">
                        <span>🟢 Positive: {pos_pct:.0f}%</span>
                        <span>🔴 Negative: {neg_pct:.0f}%</span>
                        <span>⚪ Neutral: {neu_pct:.0f}%</span>
                    </div>
                    <div class="emotions-list">
                        {emotions_html}
                    </div>
                </div>
                
                <div class="report-card">
                    <h3 style="font-size: 1.1rem; font-weight: 600; color: var(--accent-orange); margin-bottom: 1rem;">Emotion Map</h3>
                    <div>
                        {emotion_rows_html}
                    </div>
                </div>
            </div>
        </section>
        """

                html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligence Report: {query}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0E0C0B;
            --bg-secondary: #161412;
            --card-bg: rgba(255, 248, 244, 0.04);
            --border-color: rgba(232, 210, 195, 0.1);
            --text-primary: #EDE9E4;
            --text-secondary: #9A8F87;
            --text-muted: #665E59;
            --accent-orange: #E8490F;
            --accent-steel: #3D7EBF;
            --sentiment-pos: #2E8B57;
            --sentiment-neg: #C0392B;
            --sentiment-neu: #9A8F87;
            --sentiment-mixed: #C17D2E;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            background-image:
                radial-gradient(ellipse 900px 600px at 15% 0%, rgba(232,73,15,0.06) 0%, transparent 70%),
                radial-gradient(ellipse 700px 500px at 85% 100%, rgba(61,126,191,0.05) 0%, transparent 70%);
            background-attachment: fixed;
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
            padding: 2rem 1rem;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 3rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }}
        header {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 2rem;
            margin-bottom: 2.5rem;
        }}
        h1 {{
            font-family: 'Inter', sans-serif;
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-orange) 0%, var(--accent-steel) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }}
        .meta-item {{
            background: rgba(14, 12, 11, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 0.8rem 1.2rem;
        }}
        .meta-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.2rem;
        }}
        .meta-value {{
            font-weight: 600;
            font-size: 0.95rem;
        }}
        .verdict-card {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.8rem;
            margin-bottom: 2.5rem;
        }}
        .verdict-info h3 {{
            font-size: 1.1rem;
            color: var(--text-secondary);
            margin-bottom: 0.3rem;
        }}
        .verdict-badge {{
            font-size: 1.6rem;
            font-weight: 700;
            font-family: 'Inter', sans-serif;
            padding: 0.2rem 1rem;
            border-radius: 8px;
        }}
        .badge-positive {{ color: var(--sentiment-pos); }}
        .badge-negative {{ color: var(--sentiment-neg); }}
        .badge-mixed {{ color: var(--sentiment-mixed); }}
        .badge-neutral {{ color: var(--sentiment-neu); }}
        
        .score-badge {{
            font-size: 1.2rem;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.05);
            padding: 0.5rem 1rem;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}
        section {{
            margin-bottom: 3rem;
        }}
        h2 {{
            font-family: 'Inter', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 1.2rem;
            border-left: 4px solid var(--accent-orange);
            padding-left: 0.8rem;
            color: var(--text-primary);
        }}
        p {{
            color: var(--text-secondary);
            margin-bottom: 1.2rem;
            font-size: 0.98rem;
        }}
        .insight-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            margin-top: 1rem;
        }}
        .insight-group h3 {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--accent-orange);
            margin-bottom: 0.8rem;
            margin-top: 1rem;
        }}
        .insight-group:first-of-type h3 {{
            margin-top: 0;
        }}
        .insight-group ul {{
            list-style-type: none;
            padding-left: 0.5rem;
        }}
        .insight-group li {{
            position: relative;
            padding-left: 1.2rem;
            margin-bottom: 0.6rem;
            font-size: 0.95rem;
            color: var(--text-secondary);
        }}
        .insight-group li::before {{
            content: "→";
            position: absolute;
            left: 0;
            color: var(--accent-steel);
        }}
        .quote-card {{
            border-left: 3px solid var(--accent-orange);
            background: rgba(232, 73, 15, 0.05);
            padding-left: 1rem;
            margin: 1rem 0;
            font-style: italic;
            color: var(--text-secondary);
        }}
        .quote-score {{
            font-size: 0.8rem;
            color: var(--accent-steel);
            font-weight: 600;
            margin-top: 0.3rem;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        /* ── Sentiment Distribution & Emotion Map ── */
        .grid-2col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 3rem;
        }}
        @media(max-width: 768px) {{
            .grid-2col {{
                grid-template-columns: 1fr;
            }}
        }}
        .report-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
        }}
        .breakdown-bar {{
            display: flex;
            height: 12px;
            border-radius: 6px;
            overflow: hidden;
            margin: 1.2rem 0;
            background: rgba(232, 210, 195, 0.05);
        }}
        .bar-positive {{
            background: var(--sentiment-pos);
        }}
        .bar-negative {{
            background: var(--sentiment-neg);
        }}
        .bar-neutral {{
            background: var(--sentiment-neu);
        }}
        .breakdown-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 0.82rem;
            color: var(--text-secondary);
        }}
        .emotion-tag {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.75rem;
            background: rgba(61, 126, 191, 0.1);
            color: var(--accent-steel);
            border: 1px solid rgba(61, 126, 191, 0.2);
            text-transform: capitalize;
            margin-right: 6px;
            margin-bottom: 6px;
        }}
        .emotions-list {{
            display: flex;
            flex-wrap: wrap;
            margin-top: 1.2rem;
        }}
        .emotion-bar-row {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}
        .emotion-bar-label {{
            width: 100px;
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-transform: capitalize;
            flex-shrink: 0;
        }}
        .emotion-bar-track {{
            flex: 1;
            height: 8px;
            background: rgba(232, 210, 195, 0.08);
            border-radius: 4px;
            overflow: hidden;
        }}
        .emotion-bar-fill {{
            height: 100%;
            border-radius: 4px;
        }}
        .emotion-bar-val {{
            width: 32px;
            text-align: right;
            font-size: 0.78rem;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
        }}
        .controversy-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 1.5rem;
            padding-top: 1.2rem;
            border-top: 1px solid var(--border-color);
        }}
        .controversy-label {{
            font-size: 0.82rem;
            color: var(--text-secondary);
        }}
        .controversy-track {{
            flex: 1;
            height: 6px;
            background: rgba(232, 210, 195, 0.08);
            border-radius: 3px;
            overflow: hidden;
        }}
        .controversy-fill {{
            height: 100%;
            background: var(--sentiment-mixed);
        }}
        .controversy-val {{
            font-size: 0.82rem;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{query}</h1>
            <p style="color: var(--text-secondary); font-size: 1.1rem; font-style: italic; margin-bottom: 1.5rem;">{verdict.get('one_line_summary', '')}</p>
            <div class="meta-grid">
                <div class="meta-item">
                    <div class="meta-label">Analyzed Date</div>
                    <div class="meta-value">{meta.get('timestamp', 'N/A')[:10]}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">RAG Strategy</div>
                    <div class="meta-value">{meta.get('retrieval_strategy', 'N/A')}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Docs Analyzed</div>
                    <div class="meta-value">{meta.get('documents_analyzed', 0)} posts/comments</div>
                </div>
            </div>
        </header>
        
        <div class="verdict-card">
            <div class="verdict-info">
                <h3>Overall Verdict Sentiment</h3>
                <div class="verdict-badge badge-{verdict.get('overall_sentiment', 'neutral').lower()}">
                     {verdict.get('overall_sentiment', 'Unknown')}
                </div>
            </div>
            <div class="score-badge">
                Net Sentiment Score: <span style="color: { 'var(--sentiment-pos)' if verdict.get('net_sentiment_score', 0) >= 0 else 'var(--sentiment-neg)' }; font-weight: 700;">{verdict.get('net_sentiment_score', 0)}</span>
            </div>
        </div>
        
        <section>
            <h2>Executive Summary</h2>
            <p>{exec_sum}</p>
        </section>
"""
                pos_signals = report.get("positive_signals", {})
                if pos_signals:
                    html_content += f"""
        <section>
            <h2>Positive Signals (Share: {pos_signals.get('percentage', 0)}%)</h2>
            <p><strong>Key Driver:</strong> {pos_signals.get('headline', '')}</p>
        """
                    for theme in pos_signals.get("top_themes", []):
                        html_content += f"""
            <div style="margin-top: 1rem; border-bottom: 1px solid rgba(255,255,255,0.03); padding-bottom: 1rem;">
                <h4 style="color: var(--sentiment-pos); font-size: 1.05rem; margin-bottom: 0.3rem;">{theme.get('theme')} (Frequency: {theme.get('frequency')})</h4>
                <p style="font-size: 0.95rem; margin-bottom: 0.5rem;">{theme.get('description')}</p>
            """
                        req = theme.get("representative_quote")
                        if req and req.get("text"):
                            html_content += f"""
                <div class="quote-card">
                    "{req.get('text')}"
                    <div class="quote-score">Upvotes: {req.get('score', 0)} | Source: {req.get('source_post', 'Reddit comment')}</div>
                </div>
                """
                        html_content += "</div>"
                    html_content += "</section>"

                neg_signals = report.get("negative_signals", {})
                if neg_signals:
                    html_content += f"""
        <section>
            <h2>Negative Signals (Share: {neg_signals.get('percentage', 0)}%)</h2>
            <p><strong>Key Driver:</strong> {neg_signals.get('headline', '')}</p>
        """
                    for theme in neg_signals.get("top_themes", []):
                        html_content += f"""
            <div style="margin-top: 1rem; border-bottom: 1px solid rgba(255,255,255,0.03); padding-bottom: 1rem;">
                <h4 style="color: var(--sentiment-neg); font-size: 1.05rem; margin-bottom: 0.3rem;">{theme.get('theme')} (Frequency: {theme.get('frequency')})</h4>
                <p style="font-size: 0.95rem; margin-bottom: 0.5rem;">{theme.get('description')}</p>
            """
                        req = theme.get("representative_quote")
                        if req and req.get("text"):
                            html_content += f"""
                <div class="quote-card">
                    "{req.get('text')}"
                    <div class="quote-score">Upvotes: {req.get('score', 0)} | Source: {req.get('source_post', 'Reddit comment')}</div>
                </div>
                """
                        html_content += "</div>"
                    html_content += "</section>"

                # Add Sentiment Distribution & Emotion Map block
                html_content += dist_and_map_html

                if insights:
                    html_content += """
        <section>
            <h2>Actionable Insights</h2>
            <div class="insight-card">
        """
                    team_labels = {
                        "for_product_team": "Product Team Recommendations",
                        "for_marketing_team": "Marketing & Positioning Recommendations",
                        "for_support_team": "Support & Community Recommendations",
                    }
                    for key, actions in insights.items():
                        if isinstance(actions, list) and actions:
                            label = team_labels.get(key, key.replace("_", " ").title())
                            html_content += f"""
                <div class="insight-group">
                    <h3>{label}</h3>
                    <ul>
                """
                            for action in actions:
                                html_content += f"<li>{action}</li>"
                            html_content += """
                    </ul>
                </div>
                """
                    html_content += """
            </div>
        </section>
        """

                html_content += """
    </div>
</body>
</html>
"""
                return HTMLResponse(
                    content=html_content,
                    headers={"Content-Disposition": f"attachment; filename=report_{report_id}.html"}
                )

            # Format as Markdown
            md_content = f"# Intelligence Report: {query}\n\n"
            md_content += f"**Date:** {meta.get('timestamp', 'Unknown')}\n"
            md_content += f"**Mode:** {meta.get('mode', 'Unknown')}\n"
            md_content += f"**Target:** {meta.get('collection', 'Unknown')}\n\n"

            md_content += f"## Executive Summary\n{exec_sum}\n\n"

            md_content += f"## Verdict\n"
            md_content += f"- **Overall Sentiment:** {verdict.get('overall_sentiment', 'Unknown')}\n"
            md_content += f"- **Net Score:** {verdict.get('net_sentiment_score', 0)}\n"
            md_content += f"- **Confidence:** {verdict.get('confidence', 0)}\n\n"

            if insights:
                md_content += "## Actionable Insights\n"
                team_labels = {
                    "for_product_team": "Product Team",
                    "for_marketing_team": "Marketing Team",
                    "for_support_team": "Support Team",
                }
                for key, actions in insights.items():
                    if isinstance(actions, list) and actions:
                        label = team_labels.get(key, key.replace("_", " ").title())
                        md_content += f"### {label}\n"
                        for action in actions:
                            md_content += f"- {action}\n"
                md_content += "\n"

            return PlainTextResponse(
                content=md_content,
                media_type="text/markdown",
                headers={"Content-Disposition": f"attachment; filename=report_{report_id}.md"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate report: {e}")
    raise HTTPException(status_code=404, detail="Report not found")

@app.get("/api/documents")
async def get_documents_endpoint(
    query: str = "",
    subreddit: str = "",
    type: str = "",
    sort_by: str = "comment_score",
    sort_order: str = "desc",
    page: int = 1,
    limit: int = 50
):
    """Get paginated and filtered documents from the database."""
    docs, total = get_documents_paginated(
        query=query,
        subreddit=subreddit,
        doc_type=type,
        sort_by=sort_by,
        sort_order=sort_order,
        page=page,
        limit=limit
    )
    return {
        "documents": docs,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit if total > 0 else 0
    }

@app.get("/api/history")
async def get_history():
    """Get recent analysis history."""
    history = load_history()
    return {"history": list(reversed(history))}

@app.get("/api/collections")
async def get_collections():
    """Get list of available subreddits (Chroma collections)."""
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
