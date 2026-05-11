"""
Advanced Retrieval Strategies — Phase 2
---------------------------------------
Multi-Query and Self-Querying retrievers for casting a wider net
and enabling natural language metadata filters.
"""

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from config import get_llm, get_langchain_vectorstore, DEFAULT_N_RESULTS


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
    llm = get_llm(temperature=0.5)
    vectorstore = get_langchain_vectorstore()
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": n_results})
    return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm, prompt=MULTI_QUERY_PROMPT)


def get_filtered_retriever(min_score=0, flair=None, doc_type=None, n_results=DEFAULT_N_RESULTS):
    vectorstore = get_langchain_vectorstore()
    filters = {}
    if min_score > 0:
        filters["comment_score"] = {"$gte": min_score}
    if flair:
        filters["flair"] = flair
    if doc_type:
        filters["type"] = doc_type
    where_filter = None
    if len(filters) > 1:
        where_filter = {"$and": [{k: v} for k, v in filters.items()]}
    elif len(filters) == 1:
        where_filter = filters
    search_kwargs = {"k": n_results}
    if where_filter:
        search_kwargs["filter"] = where_filter
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def hybrid_retrieve(query, n_results=DEFAULT_N_RESULTS, min_score=5, use_multi_query=True):
    if use_multi_query:
        try:
            retriever = get_multi_query_retriever(n_results=n_results)
            docs = retriever.invoke(query)
        except Exception as e:
            print(f"Multi-query failed, falling back: {e}")
            vectorstore = get_langchain_vectorstore()
            docs = vectorstore.as_retriever(search_kwargs={"k": n_results}).invoke(query)
    else:
        vectorstore = get_langchain_vectorstore()
        docs = vectorstore.as_retriever(search_kwargs={"k": n_results}).invoke(query)

    seen = set()
    unique_docs = []
    for doc in docs:
        content_hash = hash(doc.page_content[:200])
        if content_hash not in seen:
            seen.add(content_hash)
            unique_docs.append(doc)
    if min_score > 0:
        unique_docs = [d for d in unique_docs if d.metadata.get("comment_score", 0) >= min_score]
    unique_docs.sort(key=lambda d: d.metadata.get("comment_score", 0), reverse=True)
    return unique_docs
