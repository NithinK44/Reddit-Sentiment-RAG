"""
Advanced Retrieval Strategies — BUGFIX-001 + Phase 2 & 3
---------------------------------------------------------
1. Multi-Query Semantic Search (inline implementation — no langchain_classic/community dep)
2. Keyword Search (BM25)
3. LLM-powered Routing (Agentic Retrieval)
4. Reciprocal Rank Fusion (RRF) for Hybrid results
"""

import json
from concurrent.futures import ThreadPoolExecutor
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


from config import get_llm, get_langchain_vectorstore, DEFAULT_N_RESULTS

# Global cache for BM25 indexes to avoid reloading all docs on every query
# Key: collection_name, Value: (bm25_instance, list_of_langchain_documents)
BM25_CACHE = {}


# ============================================================================
# PROMPTS
# ============================================================================

COMBINED_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a retrieval router and query expander for a Reddit Sentiment RAG system.
Your job is to:
1. Analyze the user's query and decide the best retrieval strategy.
2. Generate 3 alternative phrasings of the query to maximize document recall from a vector database.

STRATEGIES:
- 'SEMANTIC': Best for abstract concepts, feelings, vibes, and general opinions.
- 'HYBRID': Best for specific players, names, dates, scores, or unique entities.

Analyze the query:
1. Does it mention a specific person, place, or unique noun? -> HYBRID
2. Is it asking about a general mood or broad theme? -> SEMANTIC

Output ONLY valid JSON matching this exact schema:
{{
  "strategy": "<SEMANTIC|HYBRID>",
  "reason": "<short explanation>",
  "variants": ["<variant 1>", "<variant 2>", "<variant 3>"]
}}
"""),
    ("human", "{query}"),
])



# ============================================================================
# MULTI-QUERY INLINE IMPLEMENTATION
# ============================================================================

def get_strategy_and_variants(query: str) -> tuple[str, list[str]]:
    """Calls LLM to decide strategy and generate query variants."""
    llm = get_llm(temperature=0.2)
    chain = COMBINED_ROUTER_PROMPT | llm | StrOutputParser()
    try:
        raw_output = chain.invoke({"query": query})
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        
        data = json.loads(cleaned)
        strategy = data.get("strategy", "SEMANTIC").upper()
        variants = data.get("variants", [])
        print(f"🔄 Router chose: {strategy} (Reason: {data.get('reason', 'N/A')})")
        return strategy, variants
    except Exception as e:
        print(f"⚠️ Combined router failed, defaulting to SEMANTIC: {e}")
        return "SEMANTIC", []



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

# route_query was removed in favor of get_strategy_and_variants



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
    if use_multi_query:
        strategy, variants = get_strategy_and_variants(query)
        queries = [query] + variants[:3]
    else:
        strategy, _ = get_strategy_and_variants(query)
        queries = [query]

    vectorstore = get_langchain_vectorstore(collection_name)

    # 1. Semantic / Vector Search
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": n_results * 2})
    
    def fetch_docs(q):
        try:
            return base_retriever.invoke(q)
        except Exception as e:
            print(f"⚠️ Retrieval failed for variant '{q[:40]}...': {e}")
            return []

    print(f"🧵 Parallelizing retrieval for {len(queries)} queries...")
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        results = list(executor.map(fetch_docs, queries))

    seen = set()
    vector_docs = []
    for docs in results:
        for doc in docs:
            h = hash(doc.page_content[:300])
            if h not in seen:
                seen.add(h)
                vector_docs.append(doc)

    final_docs = vector_docs
    strategy_label = "Semantic (Vector)"

    # 2. BM25 Keyword Search — only if router chose HYBRID
    if strategy == "HYBRID":
        strategy_label = "Hybrid (BM25 + Vector)"
        try:
            if collection_name not in BM25_CACHE:
                print(f"📦 Building BM25 index for {collection_name}...")
                all_docs_data = vectorstore.get()
                langchain_docs = [
                    Document(page_content=d, metadata=m)
                    for d, m in zip(all_docs_data["documents"], all_docs_data["metadatas"])
                ]
                bm25 = get_bm25_retriever(langchain_docs)
                BM25_CACHE[collection_name] = (bm25, langchain_docs)
            else:
                print(f"📦 Using cached BM25 index for {collection_name}")
                bm25, langchain_docs = BM25_CACHE[collection_name]

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
