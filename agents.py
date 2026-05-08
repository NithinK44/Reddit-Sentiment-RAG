"""
Multi-Agent Orchestration — Phase 3 (LangGraph)
-------------------------------------------------
Breaks the analysis workload into specialized agents:
  1. Retriever Node  — fetches relevant documents
  2. Extractor Agent — pulls facts and quotes
  3. Sentiment Agent — evaluates emotional tone
  4. Synthesizer Agent — combines into structured report
"""

import json
from typing import TypedDict, Annotated
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END

from config import get_llm
from retriever import hybrid_retrieve
from schemas import SentimentReport, ReportMeta
from generator import format_docs


# ============================================================================
# STATE SCHEMA
# ============================================================================

class SentimentState(TypedDict):
    """State object passed through the multi-agent pipeline."""
    query: str                      # User's original question
    retrieved_docs: list             # Raw Document objects from retriever
    formatted_context: str           # Formatted string of retrieved docs
    doc_metadata: list               # Metadata for each document
    extraction: str                  # Extractor agent output
    sentiment_analysis: str          # Sentiment agent output
    final_report: dict               # Synthesizer agent structured output
    retrieval_strategy: str          # Which retriever was used
    doc_count: int                   # Number of documents analyzed


# ============================================================================
# AGENT NODE: RETRIEVER
# ============================================================================

def retrieve(state: SentimentState) -> dict:
    """
    Retriever Node: Fetches relevant documents using the hybrid pipeline.
    """
    query = state["query"]
    
    docs = hybrid_retrieve(
        query=query,
        n_results=15,
        min_score=3,
        use_multi_query=True,
    )
    
    # Format docs for downstream agents
    formatted = format_docs(docs)
    
    # Extract metadata
    metadata_list = [doc.metadata for doc in docs]
    
    return {
        "retrieved_docs": docs,
        "formatted_context": formatted,
        "doc_metadata": metadata_list,
        "retrieval_strategy": "hybrid_multi_query",
        "doc_count": len(docs),
    }


# ============================================================================
# AGENT NODE: EXTRACTOR
# ============================================================================

EXTRACTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a meticulous fact extractor analyzing Reddit football fan discussions.

Your job is to extract ONLY factual information from the comments, NOT to analyze sentiment.

From the retrieved documents, extract:
1. **Key Claims & Facts**: What are people saying happened? What decisions are being discussed?
2. **Direct Quotes**: The most impactful quotes that represent different viewpoints (include the comment score)
3. **Points of Agreement**: What do most commenters agree on?
4. **Points of Disagreement**: Where do opinions diverge?
5. **Key Entities**: Which players, managers, owners, or events are mentioned most?

Format your output as a structured extraction. Be precise and quote directly from the comments.

CONTEXT:
{context}"""),
    ("human", "Extract key facts and quotes about: {query}"),
])


def extract(state: SentimentState) -> dict:
    """
    Extractor Agent: Pulls raw facts, quotes, and key entities.
    """
    llm = get_llm(temperature=0.1)
    chain = EXTRACTOR_PROMPT | llm | StrOutputParser()
    
    result = chain.invoke({
        "context": state["formatted_context"],
        "query": state["query"],
    })
    
    return {"extraction": result}


# ============================================================================
# AGENT NODE: SENTIMENT ANALYZER
# ============================================================================

SENTIMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert sentiment analyst specializing in sports fan communities.

You receive both the raw context AND a fact extraction. Your job is to evaluate the EMOTIONAL TONE.

Analyze:
1. **Overall Sentiment**: Is the community Positive, Negative, Mixed, or Neutral?
2. **Confidence**: How confident are you (0.0 to 1.0)?
3. **Dominant Emotions**: List top 3-5 emotions (anger, frustration, hope, resignation, sarcasm, etc.)
4. **Sentiment Distribution**: Estimate percentages (positive/negative/neutral) — must add to 100
5. **Sarcasm Detection**: Is there notable sarcasm or irony?
6. **Majority vs Minority**: What does the majority think? Are there notable contrarian views?
7. **Intensity**: How strongly are emotions expressed?

Consider comment scores as a proxy for community agreement.
Higher-scored comments represent broader consensus.

EXTRACTED FACTS:
{extraction}

RAW CONTEXT:
{context}"""),
    ("human", "Analyze the emotional sentiment about: {query}"),
])


def analyze_sentiment(state: SentimentState) -> dict:
    """
    Sentiment Agent: Evaluates emotional tone, detects sarcasm, and
    breaks down sentiment distribution.
    """
    llm = get_llm(temperature=0.2)
    chain = SENTIMENT_PROMPT | llm | StrOutputParser()
    
    result = chain.invoke({
        "extraction": state["extraction"],
        "context": state["formatted_context"],
        "query": state["query"],
    })
    
    return {"sentiment_analysis": result}


# ============================================================================
# AGENT NODE: SYNTHESIZER
# ============================================================================

SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a report synthesizer. You combine extracted facts and sentiment analysis 
into a structured JSON report.

You MUST output ONLY valid JSON matching this exact schema (no markdown, no extra text):

{{
  "query": "<the original query>",
  "overall_sentiment": "<Positive|Negative|Mixed|Neutral>",
  "confidence": <0.0 to 1.0>,
  "summary": "<2-3 sentence executive summary>",
  "key_themes": [
    {{
      "name": "<theme name>",
      "sentiment": "<Positive|Negative|Mixed|Neutral>",
      "frequency": "<Common|Rare|Universal>",
      "description": "<1-2 sentence description>"
    }}
  ],
  "notable_quotes": [
    {{
      "text": "<exact quote>",
      "context": "<what it's responding to>",
      "sentiment": "<Positive|Negative|Mixed|Neutral>",
      "score": <reddit score as integer>
    }}
  ],
  "sentiment_breakdown": {{
    "positive_pct": <0-100>,
    "negative_pct": <0-100>,
    "neutral_pct": <0-100>,
    "dominant_emotions": ["<emotion1>", "<emotion2>", ...],
    "sarcasm_detected": <true|false>
  }}
}}

Include 3-6 key themes and 3-5 notable quotes.
Ensure percentages add up to 100.

EXTRACTED FACTS:
{extraction}

SENTIMENT ANALYSIS:
{sentiment_analysis}"""),
    ("human", "Synthesize the final report for: {query}"),
])


def synthesize(state: SentimentState) -> dict:
    """
    Synthesizer Agent: Combines extraction + sentiment into a structured
    Pydantic-validated report.
    """
    llm = get_llm(temperature=0.1)
    chain = SYNTHESIZER_PROMPT | llm | StrOutputParser()
    
    raw_output = chain.invoke({
        "extraction": state["extraction"],
        "sentiment_analysis": state["sentiment_analysis"],
        "query": state["query"],
    })
    
    # Parse the JSON output
    try:
        # Clean up any markdown formatting the LLM might add
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            # Remove markdown code fences
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        
        report_dict = json.loads(cleaned)
        
        # Add metadata
        report_dict["meta"] = {
            "timestamp": datetime.now().isoformat(),
            "documents_analyzed": state.get("doc_count", 0),
            "retrieval_strategy": state.get("retrieval_strategy", "hybrid"),
            "model_used": "gemini-2.0-flash",
        }
        
        # Validate with Pydantic
        report = SentimentReport(**report_dict)
        return {"final_report": report.model_dump()}
        
    except (json.JSONDecodeError, Exception) as e:
        # Fallback: return a basic report structure
        print(f"⚠️ JSON parsing failed: {e}")
        print(f"Raw output: {raw_output[:500]}")
        
        fallback_report = {
            "query": state["query"],
            "overall_sentiment": "Mixed",
            "confidence": 0.5,
            "summary": f"Analysis of '{state['query']}': {state.get('sentiment_analysis', 'Analysis was performed but structured output parsing failed.')}",
            "key_themes": [{"name": "General Discussion", "sentiment": "Mixed", "frequency": "Common", "description": "Multiple viewpoints were expressed."}],
            "notable_quotes": [],
            "sentiment_breakdown": {
                "positive_pct": 33.0,
                "negative_pct": 34.0,
                "neutral_pct": 33.0,
                "dominant_emotions": ["mixed"],
                "sarcasm_detected": False,
            },
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "documents_analyzed": state.get("doc_count", 0),
                "retrieval_strategy": state.get("retrieval_strategy", "hybrid"),
                "model_used": "gemini-2.0-flash",
            },
        }
        return {"final_report": fallback_report}


# ============================================================================
# GRAPH COMPILATION
# ============================================================================

def build_graph():
    """
    Build and compile the LangGraph multi-agent pipeline.
    
    Flow: retrieve → extract → analyze_sentiment → synthesize → END
    """
    graph = StateGraph(SentimentState)
    
    # Add nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("extract", extract)
    graph.add_node("analyze_sentiment", analyze_sentiment)
    graph.add_node("synthesize", synthesize)
    
    # Define edges (linear flow)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "analyze_sentiment")
    graph.add_edge("analyze_sentiment", "synthesize")
    graph.add_edge("synthesize", END)
    
    return graph.compile()


# ============================================================================
# PUBLIC API
# ============================================================================

def run_analysis(query: str) -> dict:
    """
    Run the full multi-agent sentiment analysis pipeline.
    
    Args:
        query: The user's question about Reddit sentiment
    
    Returns:
        A SentimentReport as a dictionary
    """
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


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 TESTING MULTI-AGENT PIPELINE")
    print("=" * 60)
    
    test_query = "What is the sentiment about Amorim's management?"
    print(f"\n🔎 Query: '{test_query}'\n")
    
    try:
        report = run_analysis(test_query)
        print(json.dumps(report, indent=2))
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
