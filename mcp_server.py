"""
MCP Server for Reddit Sentiment RAG
------------------------------------
This script exposes the Reddit Sentiment RAG orchestrator as an MCP server,
allowing Claude Desktop to interact with it directly.
"""
import sys
import logging
from mcp.server.fastmcp import FastMCP

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")

# Initialize FastMCP Server
mcp = FastMCP("Reddit Sentiment Agent")

@mcp.tool()
def analyze_reddit_sentiment(query: str) -> str:
    """
    Analyzes Reddit sentiment based on a natural language query.
    This tool is highly intelligent and autonomous. It will automatically decide if it needs
    to scrape new real-time data from Reddit or if it can answer using historical database context.
    
    Args:
        query: What the user wants to know (e.g., "What is the general sentiment around the new iPhone?", "Any recent news on Tesla stock?")
        
    Returns:
        A Markdown-formatted summary of the intelligence report.
    """
    logger.info(f"Received query: {query}")
    try:
        from agents.orchestrator import run_orchestrated_analysis
        report = run_orchestrated_analysis(query)
        
        # Format the JSON report into a clean Markdown string for Claude
        verdict = report.get("verdict", {})
        meta = report.get("meta", {})
        exec_sum = report.get("executive_summary", "")
        
        md = f"# Reddit Sentiment Analysis: {meta.get('query', query)}\n\n"
        net_val = verdict.get('net_sentiment_score', 0)
        net_txt = 'Positive' if net_val > 0 else 'Negative' if net_val < 0 else 'Neutral'
        md += f"**Verdict:** {verdict.get('overall_sentiment', 'Unknown')} ({net_txt})\n"
        md += f"**Confidence:** {verdict.get('confidence', 0)}\n\n"
        md += f"## Executive Summary\n{exec_sum}\n\n"
        
        if report.get("positive_signals"):
            md += f"## Positive Signals ({report['positive_signals'].get('percentage', 0)}%)\n"
            md += f"{report['positive_signals'].get('headline', '')}\n\n"
            
        if report.get("negative_signals"):
            md += f"## Negative Signals ({report['negative_signals'].get('percentage', 0)}%)\n"
            md += f"{report['negative_signals'].get('headline', '')}\n\n"
            
        return md
    except Exception as e:
        return f"Error executing analysis: {str(e)}"

@mcp.tool()
def get_reddit_database_stats(collection_name: str = "reddit_sentiment") -> str:
    """
    Returns statistics about the currently scraped and embedded Reddit data in the local ChromaDB vector database.
    Use this to check what historical data is already available before querying.
    """
    try:
        from config import get_chroma_collection
        collection = get_chroma_collection(collection_name)
        count = collection.count()
        return f"The vector database '{collection_name}' currently contains {count} embedded Reddit documents/comments."
    except Exception as e:
        return f"Error connecting to database: {str(e)}"

if __name__ == "__main__":
    mcp.run()
