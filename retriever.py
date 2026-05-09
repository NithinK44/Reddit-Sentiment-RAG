"""
Advanced Retrieval Strategies — Phase 2
---------------------------------------
Multi-Query and Self-Querying retrievers for casting a wider net
and enabling natural language metadata filters.
"""

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from config import get_llm, get_langchain_vectorstore, DEFAULT_N_RESULTS


# ============================================================================
# MULTI-QUERY RETRIEVER
# ============================================================================

MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant helping to analyze Reddit football fan sentiment.
Your task is to generate 4 different versions of the given user question to retrieve 
relevant documents from a vector database of Reddit comments about Manchester United.

Think about different ways fans might express the same sentiment or discuss the same topic.
Consider variations in:
- Football jargon vs casual language
- Emotional tone (angry rants vs measured analysis)
- Specific player/manager names vs general references
- Sarcastic vs literal expressions

Provide these alternative questions separated by newlines.

Original question: {question}""",
)


def get_multi_query_retriever(n_results: int = DEFAULT_N_RESULTS):
    """
    Create a MultiQueryRetriever that generates query variations
    to cast a wider net in ChromaDB.
    
    Args:
        n_results: Number of results per query variation
    
    Returns:
        Configured MultiQueryRetriever
    """
    llm = get_llm(temperature=0.5)
    vectorstore = get_langchain_vectorstore()
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": n_results})
    
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=MULTI_QUERY_PROMPT,
    )
    
    return retriever


# ============================================================================
# METADATA-FILTERED RETRIEVER
# ============================================================================

def get_filtered_retriever(
    min_score: int = 0,
    flair: str | None = None,
    doc_type: str | None = None,
    n_results: int = DEFAULT_N_RESULTS,
):
    """
    Create a retriever with ChromaDB metadata filters.
    
    Args:
        min_score: Minimum comment score filter
        flair: Post flair filter (e.g., "Discussion", "Match Thread")
        doc_type: Document type filter ("post" or "comment")
        n_results: Number of results to return
    
    Returns:
        Configured retriever with metadata filters
    """
    vectorstore = get_langchain_vectorstore()
    
    # Build the where filter
    filters = {}
    if min_score > 0:
        filters["comment_score"] = {"$gte": min_score}
    if flair:
        filters["flair"] = flair
    if doc_type:
        filters["type"] = doc_type
    
    # ChromaDB needs $and for multiple filters
    where_filter = None
    if len(filters) > 1:
        where_filter = {"$and": [{k: v} for k, v in filters.items()]}
    elif len(filters) == 1:
        where_filter = filters
    
    search_kwargs = {"k": n_results}
    if where_filter:
        search_kwargs["filter"] = where_filter
    
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


# ============================================================================
# HYBRID RETRIEVER (Multi-Query + Score Reranking)
# ============================================================================

def hybrid_retrieve(
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    min_score: int = 5,
    use_multi_query: bool = True,
) -> list[Document]:
    """
    Hybrid retrieval pipeline that combines multi-query retrieval
    with metadata filtering and score-based reranking.
    
    Steps:
    1. Generate query variations (multi-query) OR use single query
    2. Retrieve documents from ChromaDB
    3. Deduplicate results
    4. Rerank by comment score (higher-scored comments surface first)
    5. Filter by minimum score
    
    Args:
        query: User's question
        n_results: Number of results per query
        min_score: Minimum comment score to include
        use_multi_query: Whether to use multi-query generation
    
    Returns:
        List of deduplicated, reranked Document objects
    """
    if use_multi_query:
        try:
            retriever = get_multi_query_retriever(n_results=n_results)
            docs = retriever.invoke(query)
        except Exception as e:
            print(f"⚠️ Multi-query failed, falling back to simple retrieval: {e}")
            vectorstore = get_langchain_vectorstore()
            retriever = vectorstore.as_retriever(search_kwargs={"k": n_results})
            docs = retriever.invoke(query)
    else:
        vectorstore = get_langchain_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": n_results})
        docs = retriever.invoke(query)
    
    # Deduplicate by content
    seen = set()
    unique_docs = []
    for doc in docs:
        content_hash = hash(doc.page_content[:200])
        if content_hash not in seen:
            seen.add(content_hash)
            unique_docs.append(doc)
    
    # Filter by minimum score
    if min_score > 0:
        unique_docs = [
            doc for doc in unique_docs
            if doc.metadata.get("comment_score", 0) >= min_score
        ]
    
    # Rerank by comment score (descending)
    unique_docs.sort(
        key=lambda d: d.metadata.get("comment_score", 0),
        reverse=True,
    )
    
    return unique_docs


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 TESTING ADVANCED RETRIEVAL")
    print("=" * 60)
    
    test_query = "What do fans think about the manager's tactics?"
    print(f"\n🔎 Query: '{test_query}'\n")
    
    print("--- Multi-Query Retrieval ---")
    try:
        docs = hybrid_retrieve(test_query, n_results=5, min_score=10)
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            print(f"\n  [{i}] Score: {meta.get('comment_score', 0)} | "
                  f"Post: {meta.get('post_title', 'N/A')[:50]}")
            print(f"      {doc.page_content[:150]}...")
        print(f"\n  Total unique docs: {len(docs)}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
