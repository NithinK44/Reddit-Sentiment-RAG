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

from config import get_llm, DEFAULT_N_RESULTS, LLM_MODEL


# ============================================================================
# QUICK MODE SYSTEM PROMPT — Conversational, Plain-Text Summary
# ============================================================================

QUICK_ANALYSIS_SYSTEM_PROMPT = """You are a sharp, insightful analyst summarising Reddit community sentiment.

Given retrieved posts and comments about the topic, write a concise conversational paragraph
(4–6 sentences, ~120 words) that a product manager could read in 30 seconds.

RULES:
1. Be direct and opinionated — don't hedge everything.
2. Mention the overall sentiment clearly (positive / negative / mixed / neutral).
3. Call out the single biggest praise and the single biggest criticism if both exist.
4. If one side dominates, say so plainly.
5. Do NOT produce bullet points, headers, or JSON — plain flowing prose only.
6. Finish with a one-sentence "bottom line".
7. Only use evidence from the provided context; never fabricate.

After the prose paragraph, on two separate lines with no extra text, output:
SENTIMENT: <Positive|Negative|Mixed|Neutral>
CONFIDENCE: <0.0-1.0>

CONTEXT (Retrieved Reddit posts and comments):
{context}"""


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

from rag.retriever import hybrid_retrieve


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
    return {
        "meta": {
            "report_id": str(uuid.uuid4()),
            "query": question,
            "mode": "quick",
            "timestamp": datetime.now().isoformat(),
            "documents_analyzed": len(docs),
            "retrieval_strategy": strategy,
            "model_used": LLM_MODEL,
            "data_freshness": {
                "earliest_post": min(dates) if dates else None,
                "latest_post": max(dates) if dates else None,
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
