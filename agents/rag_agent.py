"""
Multi-Agent Orchestration — FEAT-004 (LangGraph)
-------------------------------------------------
All 4 agents updated:
  1. Retriever Node  — fetches relevant documents (15 docs)
  2. Extractor Agent — domain-agnostic positive/negative fact extraction
  3. Sentiment Agent — emotion map, controversy score, trajectory
  4. Synthesizer Agent — produces full UnifiedAnalysisReport JSON
"""

import json
import uuid
from typing import TypedDict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END

from config import get_llm, LLM_MODEL, DEEP_ANALYSIS_MODEL, UPGRADED_ANALYSIS_MODEL, load_prompt_text, LANGSMITH_TRACING, USE_GOOGLE_STUDIO, GOOGLE_MODEL
from rag.retriever import hybrid_retrieve
from agents.schemas import UnifiedAnalysisReport, UnifiedReportMeta, make_fallback_report
from rag.generator import format_docs
from core.llm_utils import parse_llm_json
from core.job_store import analysis_jobs
from core.db import get_analysis_history

# ---------------------------------------------------------------------------
# LangSmith: traceable decorator (no-op if tracing disabled or not installed)
# ---------------------------------------------------------------------------
if LANGSMITH_TRACING:
    try:
        from langsmith import traceable
        _traceable = traceable
    except ImportError:
        logger.warning("langsmith package not installed — tracing disabled")
        _traceable = lambda **kw: (lambda f: f)
else:
    _traceable = lambda **kw: (lambda f: f)

def _emit_progress(job_id: str, message: str):
    if job_id:
        job = analysis_jobs.get_job(job_id)
        if job:
            job.add_event("running", message)


class SentimentState(TypedDict):
    """State object passed through the multi-agent pipeline."""
    query: str
    retrieved_docs: list
    formatted_context: str
    doc_metadata: list
    extraction: str
    sentiment_analysis: str
    final_report: dict
    retrieval_strategy: str
    doc_count: int
    collection_name: str
    forced_strategy: str
    precomputed_variants: list  # pre-computed by Query Intelligence — skips re-routing LLM call
    job_id: str
    retry_count: int
    validation_errors: str
    raw_synthesis_output: str


# ============================================================================
# NODE 1: RETRIEVE
# ============================================================================

REFLECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", load_prompt_text("reflection.txt")),
    ("human", "Fix the JSON for query: {query}"),
])


def retrieve(state: SentimentState) -> dict:
    """
    Node 1: Retrieve relevant documents.
    Query rewriting, routing, and variant generation are pre-computed by
    the Query Intelligence call in the orchestrator — no LLM call here.
    """
    _emit_progress(state.get("job_id"), "Agent 1: Retrieving relevant documents...")
    query = state["query"]           # already rewritten by orchestrator
    coll_name = state.get("collection_name", "reddit_sentiment")
    forced = state.get("forced_strategy", "SEMANTIC")  # already decided by orchestrator
    precomputed_variants = state.get("precomputed_variants") or []

    docs, strategy = hybrid_retrieve(
        query=query,
        n_results=15,
        min_score=3,
        collection_name=coll_name,
        forced_strategy=forced,
        precomputed_variants=precomputed_variants,
    )

    formatted = format_docs(docs)
    metadata_list = [doc.metadata for doc in docs]

    return {
        "retrieved_docs": docs,
        "formatted_context": formatted,
        "doc_metadata": metadata_list,
        "retrieval_strategy": f"{forced} (Query Intelligence)",
        "doc_count": len(docs),
    }


# ============================================================================
# NODE 2: EXTRACTOR — Domain-Agnostic Positive/Negative Fact Extraction
# ============================================================================

EXTRACTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", load_prompt_text("extractor.txt")),
    ("human", "Extract key facts with positive/negative separation for: {query}"),
])


def extract(state: SentimentState) -> dict:
    _emit_progress(state.get("job_id"), "Agent 2: Extracting factual signals and evidence...")
    llm = get_llm(temperature=0.1, model_name=DEEP_ANALYSIS_MODEL)
    chain = EXTRACTOR_PROMPT | llm | StrOutputParser()
    result = chain.invoke({
        "context": state["formatted_context"],
        "query": state["query"],
    })
    return {"extraction": result}


# ============================================================================
# NODE 3: SENTIMENT ANALYST — Emotion Map, Controversy, Trajectory
# ============================================================================

SENTIMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", load_prompt_text("sentiment.txt")),
    ("human", "Produce full sentiment intelligence for: {query}"),
])


def analyze_sentiment(state: SentimentState) -> dict:
    _emit_progress(state.get("job_id"), "Agent 3: Mapping emotion landscape and controversy...")
    llm = get_llm(temperature=0.2, model_name=UPGRADED_ANALYSIS_MODEL)
    chain = SENTIMENT_PROMPT | llm | StrOutputParser()
    result = chain.invoke({
        "extraction": state["extraction"],
        "context": state["formatted_context"],
        "query": state["query"],
    })
    return {"sentiment_analysis": result}


# ============================================================================
# NODE 4: SYNTHESIZER — Produce Full UnifiedAnalysisReport JSON
# ============================================================================

SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", load_prompt_text("synthesizer.txt")),
    ("human", "Synthesize the full intelligence report for: {query}"),
])


def synthesize(state: SentimentState) -> dict:
    _emit_progress(state.get("job_id"), "Agent 4: Synthesizing and validating report...")
    
    raw_output = state.get("raw_synthesis_output")
    if not raw_output:
        llm = get_llm(temperature=0.1, model_name=UPGRADED_ANALYSIS_MODEL)
        chain = SYNTHESIZER_PROMPT | llm | StrOutputParser()
        raw_output = chain.invoke({
            "extraction": state["extraction"],
            "sentiment_analysis": state["sentiment_analysis"],
            "query": state["query"],
        })

    try:
        report_dict = parse_llm_json(raw_output)
        
        # Custom validation constraints to trigger reflection
        errors = []
        dist = report_dict.get("sentiment_distribution", {})
        pos = dist.get("positive_pct", 0)
        neg = dist.get("negative_pct", 0)
        neu = dist.get("neutral_pct", 0)
        
        if abs(pos + neg + neu - 100) > 0.1:
            errors.append(f"Sentiment distribution percentages must sum to 100 (got positive={pos}, negative={neg}, neutral={neu}, sum={pos+neg+neu})")
        
        pos_signals = report_dict.get("positive_signals", {})
        neg_signals = report_dict.get("negative_signals", {})
        
        pos_themes = pos_signals.get("top_themes", [])
        neg_themes = neg_signals.get("top_themes", [])
        pos_quotes = pos_signals.get("praise_quotes", [])
        neg_quotes = neg_signals.get("criticism_quotes", [])
        
        if len(pos_themes) < 2:
            errors.append(f"At least 2 top positive themes are required (got {len(pos_themes)})")
        if len(pos_quotes) < 3:
            errors.append(f"At least 3 positive praise quotes are required (got {len(pos_quotes)})")
        if len(neg_themes) < 2:
            errors.append(f"At least 2 top negative themes are required (got {len(neg_themes)})")
        if len(neg_quotes) < 3:
            errors.append(f"At least 3 negative criticism quotes are required (got {len(neg_quotes)})")
        
        entities = report_dict.get("key_entities", [])
        if len(entities) < 3:
            errors.append(f"At least 3 key entities are required (got {len(entities)})")
            
        insights = report_dict.get("actionable_insights", {})
        for team in ["for_product_team", "for_marketing_team", "for_support_team"]:
            team_insights = insights.get(team, [])
            if len(team_insights) < 2:
                errors.append(f"At least 2 actionable insights for {team} are required (got {len(team_insights)})")
        
        # Enriched derived fields
        report_dict["verdict"]["net_sentiment_score"] = int(pos - neg)
        dates = [m.get("post_date") for m in state["doc_metadata"] if m.get("post_date")]
        data_freshness = {
            "earliest_post": min(dates) if dates else None,
            "latest_post": max(dates) if dates else None,
        }
        # Get actual model name and retry/fallback info from job context if available
        model_used = LLM_MODEL
        fallback_used = False
        retries_occurred = 0
        
        job_id = state.get("job_id")
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
                from config import FORCE_FALLBACK_FOR_DEEP, FALLBACK_MODEL
                model_used = FALLBACK_MODEL if FORCE_FALLBACK_FOR_DEEP else DEEP_ANALYSIS_MODEL

        report_dict["meta"] = {
            "report_id": str(uuid.uuid4()),
            "query": state["query"],
            "mode": "deep",
            "timestamp": datetime.now().isoformat(),
            "documents_analyzed": state.get("doc_count", 0),
            "retrieval_strategy": state.get("retrieval_strategy", "hybrid"),
            "model_used": model_used,
            "fallback_used": fallback_used,
            "retries_occurred": retries_occurred,
            "collection": state.get("collection_name", "reddit_sentiment"),
            "data_freshness": data_freshness,
        }

        try:
            # Pydantic validation
            UnifiedAnalysisReport(**report_dict)
        except Exception as pyd_err:
            errors.append(f"Pydantic validation: {str(pyd_err)}")
            
        if errors:
            raise ValueError("; ".join(errors))
        
        return {
            "final_report": report_dict,
            "validation_errors": "",
            "raw_synthesis_output": raw_output
        }

    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        logger.warning(f"Validation failed on try {state.get('retry_count', 0) + 1}: {e}")
        return {
            "validation_errors": str(e),
            "raw_synthesis_output": raw_output,
            "final_report": {}
        }


def reflect_and_fix(state: SentimentState) -> dict:
    retry_cnt = state.get("retry_count", 0) + 1
    _emit_progress(state.get("job_id"), f"Reflection Node: Correcting report structure (Attempt {retry_cnt})...")
    
    llm = get_llm(temperature=0.1, model_name=DEEP_ANALYSIS_MODEL)
    chain = REFLECTION_PROMPT | llm | StrOutputParser()
    
    corrected_output = chain.invoke({
        "validation_errors": state["validation_errors"],
        "raw_synthesis_output": state["raw_synthesis_output"],
        "query": state["query"]
    })
    
    return {
        "raw_synthesis_output": corrected_output,
        "retry_count": retry_cnt
    }


def handle_fallback(state: SentimentState) -> dict:
    _emit_progress(state.get("job_id"), "Max validation retries reached. Creating fallback report...")
    dates = [m.get("post_date") for m in state["doc_metadata"] if m.get("post_date")]
    
    model_used = LLM_MODEL
    fallback_used = False
    retries_occurred = 0
    job_id = state.get("job_id")
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

    fallback = make_fallback_report(
        query=state["query"],
        mode="deep",
        raw_text=state.get("sentiment_analysis", ""),
        docs_analyzed=state.get("doc_count", 0),
        strategy=state.get("retrieval_strategy", "hybrid"),
        model=model_used,
        fallback_used=fallback_used,
        retries_occurred=retries_occurred,
        collection=state.get("collection_name", "reddit_sentiment"),
    )
    fallback["meta"]["data_freshness"] = {
        "earliest_post": min(dates) if dates else None,
        "latest_post": max(dates) if dates else None,
    }
    return {"final_report": fallback}


def should_continue_synthesis(state: SentimentState) -> str:
    if not state.get("validation_errors") and state.get("final_report"):
        return "end"
    
    if state.get("retry_count", 0) < 2:
        return "reflect"
    else:
        return "fallback"


# ============================================================================
# GRAPH ASSEMBLY
# ============================================================================

def build_graph():
    graph = StateGraph(SentimentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("extract", extract)
    graph.add_node("analyze_sentiment", analyze_sentiment)
    graph.add_node("synthesize", synthesize)
    graph.add_node("reflect_and_fix", reflect_and_fix)
    graph.add_node("fallback", handle_fallback)
    
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "analyze_sentiment")
    graph.add_edge("analyze_sentiment", "synthesize")
    
    graph.add_conditional_edges(
        "synthesize",
        should_continue_synthesis,
        {
            "end": END,
            "reflect": "reflect_and_fix",
            "fallback": "fallback"
        }
    )
    
    graph.add_edge("reflect_and_fix", "synthesize")
    graph.add_edge("fallback", END)
    
    return graph.compile()


# Compile the graph once at module load
logger.info("⚡ Compiling LangGraph for Sentiment Analysis...")
COMPILED_GRAPH = build_graph()


@_traceable(run_type="chain", name="Sentiment Analysis Graph")
def run_analysis(
    query: str,
    collection_name: str = "reddit_sentiment",
    strategy: str = "SEMANTIC",
    variants: list = None,
    job_id: str = "",
) -> dict:
    """
    Runs the 4-agent LangGraph pipeline.
    strategy and variants are pre-computed by the Query Intelligence call
    in the orchestrator, so no additional LLM routing calls are made here.
    """
    initial_state = {
        "query": query,
        "retrieved_docs": [],
        "formatted_context": "",
        "doc_metadata": [],
        "extraction": "",
        "sentiment_analysis": "",
        "final_report": {},
        "retrieval_strategy": "",
        "doc_count": 0,
        "collection_name": collection_name,
        "forced_strategy": strategy,
        "precomputed_variants": variants or [],
        "job_id": job_id,
        "retry_count": 0,
        "validation_errors": "",
        "raw_synthesis_output": "",
    }
    result = COMPILED_GRAPH.invoke(initial_state)
    return result["final_report"]
