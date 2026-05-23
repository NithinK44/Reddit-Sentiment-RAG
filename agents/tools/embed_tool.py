from langchain_core.tools import tool
from rag.embedder import embed_to_chromadb

@tool
def embed_data_tool(subreddit: str) -> dict:
    """
    Embeds the scraped JSON data for a subreddit into the ChromaDB vector database.
    This must be called AFTER scraping to make the data searchable by the RAG agent.
    
    Args:
        subreddit: The name of the subreddit whose data was just scraped.
        
    Returns:
        Dict containing success status and number of embedded documents.
    """
    return embed_to_chromadb(subreddit=subreddit)
