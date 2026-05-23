"""
Advanced Retrieval Strategies — BUGFIX-001 + Phase 2 & 3
---------------------------------------------------------
1. Multi-Query Semantic Search (inline implementation — no langchain_classic/community dep)
2. Keyword Search (BM25)
3. LLM-powered Routing (Agentic Retrieval)
4. Reciprocal Rank Fusion (RRF) for Hybrid results
"""

import json
import logging
import math
import datetime
from concurrent.futures import ThreadPoolExecutor
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from core.db import search_documents_fts

logger = logging.getLogger(__name__)


from config import get_llm, get_langchain_vectorstore, DEFAULT_N_RESULTS

# BM25 is now offloaded to SQLite FTS5. Keeping invalidate_bm25_cache as a noop to maintain backward compatibility.
def invalidate_bm25_cache(collection_name: str = None):
    pass

_cross_encoder_instance = None
def get_cross_encoder():
    """Lazy load the CrossEncoder model to save resources."""
    global _cross_encoder_instance
    if _cross_encoder_instance is None:
        try:
            logger.info("Loading CrossEncoder model 'cross-encoder/ms-marco-MiniLM-L-6-v2'...")
            _cross_encoder_instance = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("CrossEncoder model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder: {e}. Reranking will be bypassed.")
    return _cross_encoder_instance

def apply_temporal_decay(docs: list, decay_rate: float = 0.005) -> list:
    """Apply exponential temporal decay to documents' relevance scores."""
    today = datetime.date.today()
    decayed_docs = []
    for d in docs:
        post_date_str = d.metadata.get("post_date")
        days_old = 0
        if post_date_str:
            try:
                # Format: YYYY-MM-DD
                post_date = datetime.datetime.strptime(post_date_str, "%Y-%m-%d").date()
                days_old = (today - post_date).days
                days_old = max(0, days_old)
            except Exception:
                pass
        decay_factor = math.exp(-decay_rate * days_old)
        d.metadata["temporal_decay_factor"] = decay_factor
        
        base_score = max(1, d.metadata.get("comment_score", 1))
        d.metadata["decayed_score"] = base_score * decay_factor
        decayed_docs.append(d)
    return decayed_docs

def rerank_documents(query: str, docs: list) -> list:
    """Rerank documents using Cross-Encoder model and combine with temporal decay."""
    if not docs:
        return []
    encoder = get_cross_encoder()
    if encoder is None:
        return docs
        
    pairs = [(query, d.page_content) for d in docs]
    try:
        scores = encoder.predict(pairs)
        for d, score in zip(docs, scores):
            d.metadata["rerank_score"] = float(score)
            decay_factor = d.metadata.get("temporal_decay_factor", 1.0)
            # Combine semantic similarity and temporal decay
            d.metadata["final_combined_score"] = float(score) * decay_factor
            
        docs.sort(key=lambda d: d.metadata.get("final_combined_score", d.metadata.get("rerank_score", 0.0)), reverse=True)
    except Exception as e:
        logger.error(f"Error during CrossEncoder reranking: {e}")
    return docs


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
        logger.info(f"🔄 Router chose: {strategy} (Reason: {data.get('reason', 'N/A')})")
        return strategy, variants
    except Exception as e:
        logger.warning(f"⚠️ Combined router failed, defaulting to SEMANTIC: {e}")
        return "SEMANTIC", []



# ============================================================================
# BM25 & FUSION HELPERS
# ============================================================================

# get_bm25_retriever was removed in favor of SQLite FTS5 search


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
                    collection_name: str = "reddit_sentiment",
                    forced_strategy: str = "agentic"):
    """
    Agentic Hybrid Retrieval.
    Returns (List[Document], strategy_label: str)
    """
    # Short-circuit: if strategy is already forced, skip the LLM router call
    if forced_strategy and forced_strategy.upper() in ["SEMANTIC", "HYBRID"]:
        strategy = forced_strategy.upper()
        variants = []
        logger.info(f"🎯 Forced strategy: {strategy} — skipping LLM router call")
        if use_multi_query:
            # Still need query variants; call LLM but only for variants, not strategy
            _, variants = get_strategy_and_variants(query)
        queries = [query] + variants[:3]
    else:
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
            logger.warning(f"⚠️ Retrieval failed for variant '{q[:40]}...': {e}")
            return []

    logger.info(f"🧵 Parallelizing retrieval for {len(queries)} queries...")
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
    strategy_label = "Semantic"
    # 2. SQLite FTS5 Keyword Search — only if router chose HYBRID
    if strategy == "HYBRID":
        strategy_label = "Hybrid (FTS5 + Vector)"
        try:
            logger.info("📦 Executing SQLite FTS5 keyword query...")
            fts_results = search_documents_fts(query, limit=n_results * 4)
            keyword_docs = []
            for r in fts_results:
                meta = {
                    "post_id": r["post_id"],
                    "post_title": r["post_title"],
                    "post_url": r["post_url"],
                    "post_date": r["post_date"],
                    "post_score": r["post_score"],
                    "post_sort": r["post_sort"],
                    "flair": r["flair"],
                    "comment_score": r["comment_score"],
                    "depth": r["depth"],
                    "type": r["type"],
                }
                keyword_docs.append(Document(page_content=r["content"], metadata=meta))
                
            if keyword_docs:
                final_docs = reciprocal_rank_fusion(vector_docs, keyword_docs)
        except Exception as e:
            logger.warning(f"⚠️ FTS5 Keyword search step failed, falling back to semantic results: {e}")

    # 3. Apply temporal decay to all candidate documents
    final_docs = apply_temporal_decay(final_docs)

    # 4. Filter by minimum score
    if min_score > 0:
        final_docs = [d for d in final_docs if d.metadata.get("comment_score", 0) >= min_score]

    # 5. Rerank using CrossEncoder
    final_docs = rerank_documents(query, final_docs)

    # If CrossEncoder was not used or failed (no final_combined_score exists), sort by decayed_score descending
    if final_docs and "final_combined_score" not in final_docs[0].metadata:
        final_docs.sort(key=lambda d: d.metadata.get("decayed_score", 0.0), reverse=True)

    return final_docs[:n_results], strategy_label
