"""
Advanced Retrieval Strategies — BUGFIX-001 + Phase 2 & 3
---------------------------------------------------------
1. Multi-Query Semantic Search (inline implementation — no langchain_classic/community dep)
2. Keyword Search (BM25)
3. LLM-powered Routing (Agentic Retrieval)
4. Reciprocal Rank Fusion (RRF) for Hybrid results
"""

import json
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import get_llm, get_langchain_vectorstore, DEFAULT_N_RESULTS

# ============================================================================
# PROMPTS
# ============================================================================

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a retrieval router for a Reddit Sentiment RAG system.
Your job is to analyze the user's query and decide the best retrieval strategy.

STRATEGIES:
- 'SEMANTIC': Best for abstract concepts, feelings, vibes, and general opinions.
- 'HYBRID': Best for specific players, names, dates, scores, or unique entities.

Analyze the query:
1. Does it mention a specific person, place, or unique noun? -> HYBRID
2. Is it asking about a general mood or broad theme? -> SEMANTIC

Output ONLY valid JSON with no markdown fencing: {"strategy": "SEMANTIC", "reason": "short explanation"}"""),
    ("human", "{query}"),
])

MULTI_QUERY_PROMPT_TEMPLATE = """You are an AI assistant helping analyze Reddit community sentiment.
Generate 3 alternative phrasings of the following question to maximize document recall
from a vector database. Output each version on its own line, no numbering or prefixes.

Original question: {question}"""


# ============================================================================
# MULTI-QUERY INLINE IMPLEMENTATION
# ============================================================================

def multi_query_retrieve(vectorstore, query: str, k: int = 10) -> list:
    """
    Inline multi-query retrieval — generates 3 query variants via LLM,
    retrieves docs for each, then deduplicates by content hash.
    Replaces the removed MultiQueryRetriever dependency.
    """
    llm = get_llm(temperature=0.4)
    prompt = PromptTemplate(input_variables=["question"], template=MULTI_QUERY_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    try:
        raw_variants = chain.invoke({"question": query})
        variants = [q.strip() for q in raw_variants.strip().splitlines() if q.strip()]
    except Exception as e:
        print(f"⚠️ Multi-query generation failed, using original query: {e}")
        variants = []

    # Always include the original query
    queries = [query] + variants[:3]

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    seen = set()
    all_docs = []

    for q in queries:
        try:
            docs = base_retriever.invoke(q)
            for doc in docs:
                h = hash(doc.page_content[:300])
                if h not in seen:
                    seen.add(h)
                    all_docs.append(doc)
        except Exception as e:
            print(f"⚠️ Retrieval failed for variant '{q[:40]}...': {e}")

    return all_docs


# ============================================================================
# BM25 & FUSION HELPERS
# ============================================================================

def get_bm25_retriever(docs):
    """Build a BM25 index from a list of LangChain Documents."""
    if not docs:
        return None
    tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
    return BM25Okapi(tokenized_corpus)


def reciprocal_rank_fusion(vector_docs: list, keyword_docs: list, k: int = 60) -> list:
    """Combines two ranked lists using Reciprocal Rank Fusion (RRF)."""
    scores: dict = {}

    for rank, doc in enumerate(vector_docs):
        doc_id = doc.page_content[:200]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k)

    for rank, doc in enumerate(keyword_docs):
        doc_id = doc.page_content[:200]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k)

    all_docs_map = {doc.page_content[:200]: doc for doc in vector_docs + keyword_docs}
    sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [all_docs_map[doc_id] for doc_id, _ in sorted_ids]


# ============================================================================
# ROUTER
# ============================================================================

def route_query(query: str) -> str:
    """Ask LLM to decide the retrieval strategy (SEMANTIC or HYBRID)."""
    try:
        llm = get_llm(temperature=0)
        chain = ROUTER_PROMPT | llm
        response = chain.invoke({"query": query})

        # Robustly strip any markdown fencing the LLM may produce
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            inner = lines[1:]
            if inner and inner[-1].strip() == "```":
                inner = inner[:-1]
            content = "\n".join(inner).strip()

        data = json.loads(content)
        strategy = data.get("strategy", "SEMANTIC").upper()
        if strategy not in ("SEMANTIC", "HYBRID"):
            strategy = "SEMANTIC"
        print(f"🔄 Router chose: {strategy} (Reason: {data.get('reason', 'N/A')})")
        return strategy
    except Exception as e:
        print(f"⚠️ Router failed, defaulting to SEMANTIC: {e}")
        return "SEMANTIC"


# ============================================================================
# MAIN RETRIEVAL ENTRY POINT
# ============================================================================

def hybrid_retrieve(query: str, n_results: int = DEFAULT_N_RESULTS,
                    min_score: int = 5, use_multi_query: bool = True,
                    collection_name: str = "reddit_sentiment"):
    """
    Agentic Hybrid Retrieval.
    Returns (List[Document], strategy_label: str)
    """
    strategy = route_query(query)
    vectorstore = get_langchain_vectorstore(collection_name)

    # 1. Semantic / Vector Search
    if use_multi_query:
        vector_docs = multi_query_retrieve(vectorstore, query, k=n_results * 2)
    else:
        vector_docs = vectorstore.as_retriever(search_kwargs={"k": n_results * 2}).invoke(query)

    final_docs = vector_docs
    strategy_label = "Semantic (Vector)"

    # 2. BM25 Keyword Search — only if router chose HYBRID
    if strategy == "HYBRID":
        strategy_label = "Hybrid (BM25 + Vector)"
        try:
            all_docs_data = vectorstore.get()
            langchain_docs = [
                Document(page_content=d, metadata=m)
                for d, m in zip(all_docs_data["documents"], all_docs_data["metadatas"])
            ]
            bm25 = get_bm25_retriever(langchain_docs)
            if bm25:
                tokenized_query = query.lower().split()
                keyword_docs = bm25.get_top_n(tokenized_query, langchain_docs, n=n_results * 2)
                final_docs = reciprocal_rank_fusion(vector_docs, keyword_docs)
        except Exception as e:
            print(f"⚠️ BM25 step failed, falling back to semantic results: {e}")

    # 3. Deduplication
    seen: set = set()
    unique_docs: list = []
    for doc in final_docs:
        h = hash(doc.page_content[:300])
        if h not in seen:
            seen.add(h)
            unique_docs.append(doc)

    # 4. Score filtering
    if min_score > 0:
        unique_docs = [d for d in unique_docs if d.metadata.get("comment_score", 0) >= min_score]

    # 5. Sort by score descending
    unique_docs.sort(key=lambda d: d.metadata.get("comment_score", 0), reverse=True)

    return unique_docs[:n_results], strategy_label
