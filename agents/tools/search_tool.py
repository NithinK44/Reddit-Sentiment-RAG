from langchain_core.tools import tool
from rag.retriever import hybrid_retrieve
from rag.generator import format_docs

@tool
def vector_search_tool(query: str, collection_name: str, n_results: int = 15) -> str:
    """
    Queries the ChromaDB vector database to retrieve relevant Reddit comments and posts.
    Use this to find historical context to answer the user's question.
    
    Args:
        query: The search query string.
        collection_name: The name of the Chroma collection (usually the subreddit name).
        n_results: Number of documents to retrieve.
        
    Returns:
        A formatted string containing the retrieved documents.
    """
    docs, strategy = hybrid_retrieve(
        query=query, 
        n_results=n_results, 
        min_score=3, 
        use_multi_query=True, 
        collection_name=collection_name, 
        forced_strategy="agentic"
    )
    formatted = format_docs(docs)
    return formatted
