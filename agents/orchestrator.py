from typing import TypedDict, Literal
import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import get_llm
from agents.scraper_agent import get_scraper_agent
from agents.processor_agent import get_processor_agent
from agents.rag_agent import run_analysis
from core.llm_utils import parse_llm_json
from core.job_store import analysis_jobs

logger = logging.getLogger(__name__)

class RouteDecision(TypedDict):
    route: Literal["scraper", "rag"]
    subreddit: str
    rationale: str

# Build the routing chain once at module load time (not per-request)
_ROUTE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the Master Orchestrator for a Reddit Sentiment Analysis system.\n"
     "Given a user query, you must decide whether to fetch NEW real-time data (scrape) or answer from historical data (rag).\n\n"
     "Output your decision in strictly valid JSON format with three keys:\n"
     "1. 'route': strictly either 'scraper' or 'rag'.\n"
     "   - Use 'scraper' if the user asks for 'latest', 'today', 'real-time', or something very recent that wouldn't be in the database.\n"
     "   - Use 'rag' for general questions, analysis of existing communities, or historical questions.\n"
     "2. 'subreddit': The name of the subreddit to target (e.g. 'apple', 'wallstreetbets'). Default to '{current_sub}' if not explicitly mentioned.\n"
     "3. 'rationale': A brief string explaining your choice.\n"
     ),
    ("human", "{query}")
])

ORCHESTRATOR_CHAIN = _ROUTE_PROMPT | get_llm(temperature=0.1) | StrOutputParser()

def run_orchestrated_analysis(query: str, collection_name: str, job_id: str = "") -> dict:
    """
    The main entry point for the Multi-Agent system.
    """
    def emit(msg: str):
        if job_id:
            job = analysis_jobs.get_job(job_id)
            if job:
                job.add_event("running", msg)

    emit("Orchestrator: Analyzing intent...")

    try:
        raw_response = ORCHESTRATOR_CHAIN.invoke({"query": query, "current_sub": collection_name})
        decision = parse_llm_json(raw_response)
        route = decision.get("route", "rag").lower()
        target_sub = decision.get("subreddit", collection_name)
        emit(f"Orchestrator decision: {decision.get('rationale', 'No rationale')}")
    except Exception as e:
        logger.error(f"Orchestrator routing failed: {e}")
        emit(f"Orchestrator failed to route, defaulting to RAG: {e}")
        route = "rag"
        target_sub = collection_name

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

    emit(f"Orchestrator: Routing to RAG Synthesis Agent...")
    # Run the existing RAG pipeline to answer the user's question
    return run_analysis(query, collection_name=target_sub, strategy="agentic", job_id=job_id)
