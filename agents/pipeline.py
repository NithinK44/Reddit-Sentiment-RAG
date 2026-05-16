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

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END

from config import get_llm, LLM_MODEL
from rag.retriever import hybrid_retrieve
from agents.schemas import UnifiedAnalysisReport, UnifiedReportMeta, make_fallback_report
from rag.generator import format_docs


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


# ============================================================================
# NODE 1: RETRIEVE
# ============================================================================

def retrieve(state: SentimentState) -> dict:
    query = state["query"]
    docs, strategy = hybrid_retrieve(query=query, n_results=15, min_score=3, use_multi_query=True)

    formatted = format_docs(docs)
    metadata_list = [doc.metadata for doc in docs]

    return {
        "retrieved_docs": docs,
        "formatted_context": formatted,
        "doc_metadata": metadata_list,
        "retrieval_strategy": f"Agentic Router: {strategy}",
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
    llm = get_llm(temperature=0.1)
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

5. CONTROVERSY SCORE (0.0-1.0):
   How polarised is this community? (1.0 = deeply divided, 0.0 = unanimous)

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
    llm = get_llm(temperature=0.2)
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
  "executive_summary": "<3-4 sentence paragraph covering BOTH positives AND negatives with equal depth>",
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
    "controversy_score": <0.0-1.0>
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
    llm = get_llm(temperature=0.1)
    chain = SYNTHESIZER_PROMPT | llm | StrOutputParser()
    raw_output = chain.invoke({
        "extraction": state["extraction"],
        "sentiment_analysis": state["sentiment_analysis"],
        "query": state["query"],
    })

    try:
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        report_dict = json.loads(cleaned)

        # ── Compute derived fields (not delegated to LLM) ──────────────────
        dist = report_dict.get("sentiment_distribution", {})
        pos = dist.get("positive_pct", 0)
        neg = dist.get("negative_pct", 0)
        if "verdict" in report_dict:
            report_dict["verdict"]["net_sentiment_score"] = int(pos - neg)

        # ── Data freshness from doc metadata ───────────────────────────────
        dates = [m.get("post_date") for m in state["doc_metadata"] if m.get("post_date")]
        data_freshness = {
            "earliest_post": min(dates) if dates else None,
            "latest_post": max(dates) if dates else None,
        }

        # ── Build meta ─────────────────────────────────────────────────────
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

        # ── Validate with Pydantic ─────────────────────────────────────────
        report = UnifiedAnalysisReport(**report_dict)
        return {"final_report": report.model_dump()}

    except (json.JSONDecodeError, Exception) as e:
        print(f"[Deep Mode] JSON parsing failed: {e}")
        dates = [m.get("post_date") for m in state["doc_metadata"] if m.get("post_date")]
        fallback = make_fallback_report(
            query=state["query"],
            mode="deep",
            raw_text=state.get("sentiment_analysis", ""),
            docs_analyzed=state.get("doc_count", 0),
            strategy=state.get("retrieval_strategy", "hybrid"),
            model=LLM_MODEL,
        )
        # Override data_freshness with actual dates if available
        fallback["meta"]["data_freshness"] = {
            "earliest_post": min(dates) if dates else None,
            "latest_post": max(dates) if dates else None,
        }
        return {"final_report": fallback}


# ============================================================================
# GRAPH ASSEMBLY
# ============================================================================

def build_graph():
    graph = StateGraph(SentimentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("extract", extract)
    graph.add_node("analyze_sentiment", analyze_sentiment)
    graph.add_node("synthesize", synthesize)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "analyze_sentiment")
    graph.add_edge("analyze_sentiment", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def run_analysis(query: str) -> dict:
    graph = build_graph()
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
    }
    result = graph.invoke(initial_state)
    return result["final_report"]
