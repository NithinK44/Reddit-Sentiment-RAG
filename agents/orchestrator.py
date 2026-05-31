import json
import logging
from typing import TypedDict, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import get_llm, LANGSMITH_TRACING, load_prompt_text
from agents.scraper_agent import get_scraper_agent
from agents.processor_agent import get_processor_agent
from agents.rag_agent import run_analysis
from core.llm_utils import parse_llm_json
from core.db import get_analysis_history
from core.job_store import analysis_jobs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangSmith: single root trace wrapper
# ---------------------------------------------------------------------------
if LANGSMITH_TRACING:
    try:
        from langsmith import traceable
        _traceable = traceable
        logger.info("✅ LangSmith @traceable loaded for orchestrator")
    except ImportError:
        logger.warning("langsmith package not installed — tracing disabled")
        _traceable = lambda **kw: (lambda f: f)  # no-op decorator
else:
    _traceable = lambda **kw: (lambda f: f)  # no-op decorator


# ---------------------------------------------------------------------------
# Combined Query Intelligence — replaces Orchestrator + Retrieval Router + Rewriter
# ---------------------------------------------------------------------------
_QUERY_INTELLIGENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", load_prompt_text("query_intelligence.txt")),
    ("human", "{query}"),
])

_QUERY_INTELLIGENCE_CHAIN = _QUERY_INTELLIGENCE_PROMPT | get_llm(temperature=0.1) | StrOutputParser()


def run_query_intelligence(query: str, collection_name: str) -> dict:
    """
    Single LLM call that replaces three separate calls:
      1. Orchestrator router (scraper vs rag)
      2. Retrieval router (SEMANTIC vs HYBRID + query variants)
      3. Query rewriter (conversation history deduplication)

    Returns a dict with keys: route, subreddit, rationale,
    rewritten_query, retrieval_strategy, retrieval_reason, variants
    """
    # Build conversation history string
    history = ""
    try:
        history_entries = get_analysis_history(limit=5)
        if history_entries:
            for h in reversed(history_entries):
                if h["query"].strip().lower() != query.strip().lower():
                    history += (
                        f"User Query: {h['query']}\n"
                        f"System Verdict: {h['sentiment']} "
                        f"(Confidence: {h['confidence']}, Net Score: {h['net_sentiment_score']})\n---\n"
                    )
    except Exception as e:
        logger.warning(f"Failed to load conversation history for query intelligence: {e}")

    try:
        raw = _QUERY_INTELLIGENCE_CHAIN.invoke({
            "query": query,
            "history": history.strip() or "None",
            "current_sub": collection_name,
        })
        result = parse_llm_json(raw)
        logger.info(
            f"🧠 Query Intelligence — route={result.get('route')} "
            f"strategy={result.get('retrieval_strategy')} "
            f"rewritten='{result.get('rewritten_query', query)[:60]}'"
        )
        return result
    except Exception as e:
        logger.warning(f"⚠️ Query Intelligence call failed, using safe defaults: {e}")
        return {
            "route": "rag",
            "subreddit": collection_name,
            "rationale": "Fallback to RAG due to error",
            "rewritten_query": query,
            "retrieval_strategy": "SEMANTIC",
            "retrieval_reason": "Default fallback",
            "variants": [],
        }


@_traceable(run_type="chain", name="Reddit Sentiment RAG Pipeline")
def run_orchestrated_analysis(query: str, collection_name: str, job_id: str = "") -> dict:
    """
    The main entry point for the Multi-Agent system.
    Uses a single 'Query Intelligence' LLM call instead of three separate ones.
    """
    def emit(msg: str):
        if job_id:
            job = analysis_jobs.get_job(job_id)
            if job:
                job.add_event("running", msg)

    emit("Orchestrator: Analyzing query intent, retrieval strategy, and context...")

    intelligence = run_query_intelligence(query, collection_name)

    route = intelligence.get("route", "rag").lower()
    target_sub = intelligence.get("subreddit", collection_name)
    rewritten_query = intelligence.get("rewritten_query", query) or query
    retrieval_strategy = intelligence.get("retrieval_strategy", "SEMANTIC")
    variants = intelligence.get("variants", [])

    emit(f"Orchestrator decision: {intelligence.get('rationale', 'No rationale')}")
    if rewritten_query != query:
        emit(f"Query rewritten: '{rewritten_query}'")

    if route == "scraper":
        emit(f"Orchestrator: Routing to Scraper Agent for r/{target_sub}...")
        scraper = get_scraper_agent()
        try:
            result = scraper.invoke({"messages": [("user", f"Scrape the latest 10 hot posts from r/{target_sub}.")]})
            if not result:
                emit(f"Warning: Scraper returned no result for r/{target_sub}. Falling back to existing data.")
        except Exception as e:
            logger.warning(f"Scraper agent failed for r/{target_sub}: {e}")
            emit(f"Warning: Scraping failed ({e}). Proceeding with existing data.")

        emit(f"Orchestrator: Routing to Processor Agent for r/{target_sub}...")
        processor = get_processor_agent()
        try:
            result = processor.invoke({"messages": [("user", f"Embed the new data for {target_sub}.")]})
            if not result:
                emit(f"Warning: Processor returned no result for r/{target_sub}.")
        except Exception as e:
            logger.warning(f"Processor agent failed for r/{target_sub}: {e}")
            emit(f"Warning: Embedding failed ({e}). Analysis may use stale data.")

    emit("Orchestrator: Routing to RAG Synthesis Agent...")
    return run_analysis(
        rewritten_query,
        collection_name=target_sub,
        strategy=retrieval_strategy,
        variants=variants,
        job_id=job_id,
    )
