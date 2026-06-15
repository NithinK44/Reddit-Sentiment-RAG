"""
RAG Generator — Quick Mode (Conversational Summary)
----------------------------------------------------
query_rag() now returns a UnifiedAnalysisReport-compatible dict so the
API contract is identical between Quick and Deep modes. The summary lives
in `executive_summary`; the one-line headline is in `verdict.one_line_summary`.
"""

import uuid
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import get_llm, DEFAULT_N_RESULTS, LLM_MODEL, load_prompt_text, USE_GOOGLE_STUDIO, GOOGLE_MODEL


# ============================================================================
QUICK_ANALYSIS_SYSTEM_PROMPT = load_prompt_text("quick_analysis.txt")


# ============================================================================
# DOCUMENT FORMATTER (shared with pipeline.py)
# ============================================================================

def format_docs(docs) -> str:
    """Format retrieved documents into a readable context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = f"--- Document {i} ---"
        post_info = f"Post: {meta.get('post_title', 'Unknown')}"
        date_info = f"Date: {meta.get('post_date', 'Unknown')}"
        score_info = f"Score: {meta.get('comment_score', 0)}"
        flair_info = f"Flair: {meta.get('flair', 'Unknown')}"
        type_info = f"Type: {meta.get('type', 'comment')}"
        formatted.append(
            f"{header}\n{post_info}\n{date_info}\n{score_info}\n{flair_info}\n{type_info}\n\n{doc.page_content}"
        )
    return "\n\n".join(formatted)


# ============================================================================
# MAIN ENTRY POINT — query_rag()
# ============================================================================

# ---------------------------------------------------------------------------
# LangSmith: traceable decorator (no-op if tracing disabled or not installed)
# ---------------------------------------------------------------------------
from config import LANGSMITH_TRACING
if LANGSMITH_TRACING:
    try:
        from langsmith import traceable
        _traceable = traceable
    except ImportError:
        _traceable = lambda **kw: (lambda f: f)
else:
    _traceable = lambda **kw: (lambda f: f)


from rag.retriever import hybrid_retrieve


@_traceable(run_type="chain", name="Reddit Sentiment Quick RAG")
def query_rag(question: str, n_results: int = 8, collection_name: str = "reddit_sentiment", strategy: str = "agentic") -> dict:
    """
    FEAT-005 Quick Mode: Single-LLM-call conversational summary.
    Returns: { summary, sentiment, confidence, meta }
    """
    # 1. Retrieve with smart router
    docs, strategy = hybrid_retrieve(question, n_results=n_results, collection_name=collection_name, forced_strategy=strategy)

    # 2. Format docs for context
    context = format_docs(docs)

    # 3. Collect dates for data_freshness
    dates = [doc.metadata.get("post_date") for doc in docs if doc.metadata.get("post_date")]

    # 4. Single LLM call — conversational prose
    llm = get_llm(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUICK_ANALYSIS_SYSTEM_PROMPT),
        ("human", "Summarise community sentiment about: {question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"context": context, "question": question})

    # 5. Parse the prose body and trailing metadata lines
    sentiment = "Mixed"
    confidence = 0.7
    summary_lines = []

    for line in raw.strip().splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("SENTIMENT:"):
            val = stripped.split(":", 1)[1].strip().capitalize()
            if val in ("Positive", "Negative", "Mixed", "Neutral"):
                sentiment = val
        elif upper.startswith("CONFIDENCE:"):
            try:
                parsed = float(stripped.split(":", 1)[1].strip())
                confidence = max(0.0, min(1.0, parsed))
            except ValueError:
                pass
        else:
            summary_lines.append(line)

    summary = "\n".join(summary_lines).strip()

    # Return a UnifiedAnalysisReport-compatible dict so the API shape is
    # identical between Quick and Deep modes.
    # Get actual model name and retry/fallback info from thread context
    from config import thread_local
    job_id = getattr(thread_local, 'job_id', None)
    model_used = LLM_MODEL
    fallback_used = False
    retries_occurred = 0
    
    if job_id:
        from core.job_store import analysis_jobs
        job = analysis_jobs.get_job(job_id)
        if job:
            if getattr(job, 'models_used', None):
                model_used = ", ".join(job.models_used)
            fallback_used = getattr(job, 'fallback_used', False)
            retries_occurred = getattr(job, 'retries_occurred', 0)
    else:
        if USE_GOOGLE_STUDIO:
            model_used = GOOGLE_MODEL

    # Compute data age in days from latest post
    data_age_days = None
    if dates:
        try:
            from datetime import date as _date
            data_age_days = (_date.today() - _date.fromisoformat(max(dates))).days
        except Exception:
            pass

    return {
        "meta": {
            "report_id": str(uuid.uuid4()),
            "query": question,
            "mode": "quick",
            "timestamp": datetime.now().isoformat(),
            "documents_analyzed": len(docs),
            "retrieval_strategy": strategy,
            "model_used": model_used,
            "fallback_used": fallback_used,
            "retries_occurred": retries_occurred,
            "collection": collection_name,
            "data_freshness": {
                "earliest_post": min(dates) if dates else None,
                "latest_post": max(dates) if dates else None,
                "data_age_days": data_age_days,
            },
        },
        "verdict": {
            "overall_sentiment": sentiment,
            "confidence": confidence,
            "net_sentiment_score": 0,          # not computed in quick mode
            "one_line_summary": summary[:120] if summary else "",
        },
        "executive_summary": summary,
        # Remaining fields are empty — quick mode skips full extraction
        "positive_signals": {"headline": "", "percentage": 0.0, "top_themes": [], "praise_quotes": []},
        "negative_signals": {"headline": "", "percentage": 0.0, "top_themes": [], "criticism_quotes": []},
        "sentiment_distribution": {
            "positive_pct": 0.0, "negative_pct": 0.0, "neutral_pct": 0.0,
            "dominant_emotions": [], "emotion_map": {},
            "sarcasm_detected": False, "controversy_score": 0.0, "controversy_drivers": [],
        },
        "key_entities": [],
        "competitive_signals": {"mentions_competitors": False, "competitors_mentioned": [], "comparison_sentiment": "N/A"},
        "trend_indicators": {"sentiment_trajectory": "Insufficient Data", "urgent_concerns": [], "emerging_positives": []},
        "actionable_insights": {"for_product_team": [], "for_marketing_team": [], "for_support_team": []},
    }
