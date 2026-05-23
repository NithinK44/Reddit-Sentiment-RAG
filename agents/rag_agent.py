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

from config import get_llm, LLM_MODEL, DEEP_ANALYSIS_MODEL
from rag.retriever import hybrid_retrieve
from agents.schemas import UnifiedAnalysisReport, UnifiedReportMeta, make_fallback_report
from rag.generator import format_docs
from core.llm_utils import parse_llm_json
from core.job_store import analysis_jobs
from core.db import get_analysis_history

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
    job_id: str
    retry_count: int
    validation_errors: str
    raw_synthesis_output: str


# ============================================================================
# NODE 1: RETRIEVE
# ============================================================================

REWRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a conversational query rewriting assistant.
Given the conversation history (previous queries and summaries of their verdicts) and a follow-up query,
your job is to reformulate the follow-up query into a standalone query that contains all necessary context.
If the follow-up query is already standalone and does not need context, output it exactly as-is.

CONVERSATION HISTORY:
{history}

FOLLOW-UP QUERY:
{query}

Output ONLY the rewritten standalone query. No preamble, no explanation, no markdown formatting.
"""),
    ("human", "Rewrite the query."),
])

REFLECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a JSON correction assistant.
You are given a previously generated JSON report that failed validation.
Your task is to fix the JSON according to the validation error and the original schema.

ORIGINAL SCHEMA & INSTRUCTIONS:
- The JSON must be valid.
- positive_pct + negative_pct + neutral_pct MUST equal 100.
- positive_signals.top_themes: at least 2 entries with real quotes.
- positive_signals.praise_quotes: at least 3 entries.
- negative_signals.top_themes: at least 2 entries with real quotes.
- negative_signals.criticism_quotes: at least 3 entries.
- key_entities: at least 3 entries.
- actionable_insights: at least 2 items per team.

VALIDATION ERRORS ENCOUNTERED:
{validation_errors}

PREVIOUS INCORRECT OUTPUT:
{raw_synthesis_output}

Output ONLY the corrected valid JSON. Do not write any markdown code block wraps, preamble, or explanation. Output only valid JSON.
"""),
    ("human", "Fix the JSON for query: {query}"),
])

def retrieve(state: SentimentState) -> dict:
    _emit_progress(state.get("job_id"), "Agent 1: Resolving query context and retrieving relevant documents...")
    query = state["query"]
    coll_name = state.get("collection_name", "reddit_sentiment")
    forced = state.get("forced_strategy", "agentic")

    # ── Conversational Memory / Query Rewriting ───────────────────────
    try:
        history_entries = get_analysis_history(limit=5)
        if history_entries:
            formatted_history = ""
            for h in reversed(history_entries):
                if h["query"].strip().lower() != query.strip().lower():
                    formatted_history += f"User Query: {h['query']}\nSystem Verdict: {h['sentiment']} (Confidence: {h['confidence']}, Net Score: {h['net_sentiment_score']})\n---\n"
            
            if formatted_history.strip():
                logger.info("Found conversational history. Rewriting query...")
                llm = get_llm(temperature=0.1)
                chain = REWRITER_PROMPT | llm | StrOutputParser()
                rewritten = chain.invoke({
                    "history": formatted_history,
                    "query": query
                }).strip()
                if rewritten and len(rewritten) > 3:
                    logger.info(f"Rewrote query from '{query}' to '{rewritten}'")
                    query = rewritten
    except Exception as history_err:
        logger.warning(f"Failed to apply conversational memory rewrite: {history_err}")

    docs, strategy = hybrid_retrieve(query=query, n_results=15, min_score=3, use_multi_query=True, collection_name=coll_name, forced_strategy=forced)

    formatted = format_docs(docs)
    metadata_list = [doc.metadata for doc in docs]

    return {
        "retrieved_docs": docs,
        "formatted_context": formatted,
        "doc_metadata": metadata_list,
        "retrieval_strategy": f"Agentic Router: {strategy}" if forced == "agentic" else f"Forced: {strategy}",
        "doc_count": len(docs),
    }


# ============================================================================
# NODE 2: EXTRACTOR — Domain-Agnostic Positive/Negative Fact Extraction
# ============================================================================

EXTRACTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a meticulous fact extractor analyzing online community discussions.

You receive retrieved posts and comments about: {query}

Extract ONLY factual information from the comments — do NOT analyze sentiment yet.

POSITIVE SIGNALS — Extract separately:
1. What are users praising or expressing satisfaction about?
2. Which specific features, aspects, decisions, or people are viewed favourably?
3. What outcomes do users celebrate or approve of?
4. Direct quotes expressing positivity (include the comment score in brackets like [score: 847]).

NEGATIVE SIGNALS — Extract separately:
5. What are users criticising, complaining about, or expressing frustration with?
6. Which specific features, aspects, decisions, or people are viewed unfavourably?
7. What recurring pain points or failures do users mention?
8. Direct quotes expressing dissatisfaction, anger, or disappointment (include score).

NEUTRAL / CONTEXTUAL:
9. Key entities mentioned most: products, features, people, events, competitors.
10. Points of strong community agreement (heavily upvoted).
11. Points of controversy or polarisation (divisive topics).
12. Date range of discussions if apparent from context.

Format as structured text with clear POSITIVE / NEGATIVE / NEUTRAL sections.
Be precise. Quote directly from comments. Include Reddit scores where visible.

CONTEXT:
{context}"""),
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
    ("system", """You are an expert sentiment analyst for online community intelligence.

You help companies understand how their users feel about products, services, and brand decisions.

Given the extracted facts about: {query}

Produce a structured sentiment analysis covering ALL of the following:

1. SENTIMENT SPLIT: What % is positive? Negative? Neutral? (must sum to 100)

2. POSITIVE BREAKDOWN:
   - Top 3 emotions driving positivity (e.g., excitement, satisfaction, hope, pride)
   - What specifically triggers positive sentiment?
   - Are there any surprise positives or unexpected praise?

3. NEGATIVE BREAKDOWN:
   - Top 3 emotions driving negativity (e.g., anger, frustration, disappointment, resignation)
   - What specifically triggers negative sentiment?
   - Are there urgent or repeated pain points?

4. EMOTION MAP — Rate 0-100 (relative intensity, not %):
   anger, frustration, hope, satisfaction, disappointment, excitement, sarcasm, resignation

5. CONTROVERSY:
   - CONTROVERSY SCORE (0.0-1.0): How polarised is this community? (1.0 = deeply divided, 0.0 = unanimous)
   - CONTROVERSY DRIVERS: List specific topics, features, decisions, or entities driving this polarization (e.g. why users disagree).

6. SARCASM: Is sarcasm a significant signal? If yes, does it serve positive or negative sentiment?

7. SENTIMENT TRAJECTORY:
   Based on post dates and how topics evolved — is sentiment Improving, Declining, Stable, or Insufficient Data?

8. URGENT CONCERNS: What problems, if any, demand immediate business attention? List specifically.

9. EMERGING POSITIVES: What new positive signals are starting to appear? List specifically.

10. COMPETITIVE CONTEXT:
    Are competitors mentioned? If so, is the comparison favourable or unfavourable?
    List competitor names if present.

11. KEY ENTITIES: List top 5 entities (products, people, features, events) with their net sentiment.

12. ACTIONABLE INSIGHTS per team:
    - For Product Team: What specific changes or features are users asking for?
    - For Marketing Team: What authentic praise can be amplified? Any strong advocacy quotes?
    - For Support Team: What recurring issues need triage?

EXTRACTED FACTS:
{extraction}

RAW CONTEXT:
{context}"""),
    ("human", "Produce full sentiment intelligence for: {query}"),
])


def analyze_sentiment(state: SentimentState) -> dict:
    _emit_progress(state.get("job_id"), "Agent 3: Mapping emotion landscape and controversy...")
    llm = get_llm(temperature=0.2, model_name=DEEP_ANALYSIS_MODEL)
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
    ("system", """You are a report synthesizer. You combine extracted facts and sentiment analysis
into a structured JSON intelligence report.

You MUST output ONLY valid JSON matching this exact schema (no markdown, no extra text):

{{
  "verdict": {{
    "overall_sentiment": "<Positive|Negative|Mixed|Neutral>",
    "confidence": <0.0-1.0>,
    "one_line_summary": "<punchy ≤20-word headline capturing the dominant sentiment>"
  }},
  "executive_summary": "<brief 1-2 sentence summary paragraph covering key positives and negatives>",
  "positive_signals": {{
    "headline": "<one sentence — what users love most>",
    "percentage": <0-100 float>,
    "top_themes": [
      {{
        "theme": "<theme name>",
        "frequency": "<Universal|Common|Rare>",
        "evidence_count": <integer>,
        "description": "<what exactly users praise, 1-2 sentences>",
        "representative_quote": {{"text": "<exact quote>", "score": <int>, "source_post": "<post title or null>"}}
      }}
    ],
    "praise_quotes": [
      {{"text": "<exact quote>", "score": <int>, "context": "<what it responds to>", "source_post": "<title or null>"}}
    ]
  }},
  "negative_signals": {{
    "headline": "<one sentence — what users dislike most>",
    "percentage": <0-100 float>,
    "top_themes": [
      {{
        "theme": "<theme name>",
        "frequency": "<Universal|Common|Rare>",
        "evidence_count": <integer>,
        "description": "<what exactly users criticise, 1-2 sentences>",
        "representative_quote": {{"text": "<exact quote>", "score": <int>, "source_post": "<post title or null>"}}
      }}
    ],
    "criticism_quotes": [
      {{"text": "<exact quote>", "score": <int>, "context": "<what it responds to>", "source_post": "<title or null>"}}
    ]
  }},
  "sentiment_distribution": {{
    "positive_pct": <0-100>,
    "negative_pct": <0-100>,
    "neutral_pct": <0-100>,
    "dominant_emotions": ["<emotion1>", "<emotion2>", "<emotion3>"],
    "emotion_map": {{
      "anger": <0-100>, "frustration": <0-100>, "hope": <0-100>,
      "satisfaction": <0-100>, "disappointment": <0-100>,
      "excitement": <0-100>, "sarcasm": <0-100>, "resignation": <0-100>
    }},
    "sarcasm_detected": <true|false>,
    "controversy_score": <0.0-1.0>,
    "controversy_drivers": ["<driver1>", "<driver2>"]
  }},
  "key_entities": [
    {{"name": "<name>", "type": "<Product|Person|Feature|Event|Other>", "mention_count": <int>, "net_sentiment": "<Positive|Negative|Mixed|Neutral>"}}
  ],
  "competitive_signals": {{
    "mentions_competitors": <true|false>,
    "competitors_mentioned": ["<name>"],
    "comparison_sentiment": "<Favourable|Unfavourable|Neutral|N/A>"
  }},
  "trend_indicators": {{
    "sentiment_trajectory": "<Improving|Declining|Stable|Insufficient Data>",
    "urgent_concerns": ["<concern1>", "<concern2>"],
    "emerging_positives": ["<positive1>", "<positive2>"]
  }},
  "actionable_insights": {{
    "for_product_team": ["<insight1>", "<insight2>"],
    "for_marketing_team": ["<insight1>", "<insight2>"],
    "for_support_team": ["<insight1>", "<insight2>"]
  }}
}}

MINIMUM REQUIREMENTS:
- positive_signals.top_themes: at least 2 entries with real quotes
- positive_signals.praise_quotes: at least 3 entries
- negative_signals.top_themes: at least 2 entries with real quotes
- negative_signals.criticism_quotes: at least 3 entries
- key_entities: at least 3 entries
- actionable_insights: at least 2 items per team
- positive_pct + negative_pct + neutral_pct MUST equal 100

EXTRACTED FACTS:
{extraction}

SENTIMENT ANALYSIS:
{sentiment_analysis}"""),
    ("human", "Synthesize the full intelligence report for: {query}"),
])


def synthesize(state: SentimentState) -> dict:
    _emit_progress(state.get("job_id"), "Agent 4: Synthesizing and validating report...")
    
    raw_output = state.get("raw_synthesis_output")
    if not raw_output:
        llm = get_llm(temperature=0.1, model_name=DEEP_ANALYSIS_MODEL)
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
        report_dict["meta"] = {
            "report_id": str(uuid.uuid4()),
            "query": state["query"],
            "mode": "deep",
            "timestamp": datetime.now().isoformat(),
            "documents_analyzed": state.get("doc_count", 0),
            "retrieval_strategy": state.get("retrieval_strategy", "hybrid"),
            "model_used": LLM_MODEL,
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
    fallback = make_fallback_report(
        query=state["query"],
        mode="deep",
        raw_text=state.get("sentiment_analysis", ""),
        docs_analyzed=state.get("doc_count", 0),
        strategy=state.get("retrieval_strategy", "hybrid"),
        model=LLM_MODEL,
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


def run_analysis(query: str, collection_name: str = "reddit_sentiment", strategy: str = "agentic", job_id: str = "") -> dict:
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
        "job_id": job_id,
        "retry_count": 0,
        "validation_errors": "",
        "raw_synthesis_output": "",
    }
    result = COMPILED_GRAPH.invoke(initial_state)
    return result["final_report"]
