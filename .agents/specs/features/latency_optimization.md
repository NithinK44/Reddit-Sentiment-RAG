# Latency Optimization Spec

## Objective
Optimize the latency of the Reddit Sentiment RAG system by addressing bottlenecks in retrieval, agent pipeline execution, and API endpoints.

## Proposed Phases

### Phase 1: Critical Fixes (Immediate Impact)
1.  **Fix BM25 On-the-Fly Build in `rag/retriever.py`**
    *   **Problem:** Currently, `hybrid_retrieve` fetches *all* documents from ChromaDB to build the BM25 index on every query where strategy is HYBRID. This will not scale.
    *   **Solution:** Cache the BM25 index in memory for the active collection. When a query comes in, check if the index for that collection is already built. If not, build it and store it in a global dictionary or a cache.
2.  **Cache Graph Compilation in `agents/pipeline.py`**
    *   **Problem:** `run_analysis` calls `build_graph()` on every invocation, compiling the LangGraph state graph repeatedly.
    *   **Solution:** Compile the graph once at the module level or use a simple singleton/cache pattern.

### Phase 2: Parallelization & I/O Optimization
1.  **Parallelize Multi-Query Retrieval in `rag/retriever.py`**
    *   **Problem:** `multi_query_retrieve` loops through queries sequentially.
    *   **Solution:** Use `concurrent.futures.ThreadPoolExecutor` or `asyncio` to run the vector store queries in parallel.
2.  **Stream Data in `rag/embedder.py`**
    *   **Problem:** `embed_to_chromadb` loads all documents into memory before batching them for upsert.
    *   **Solution:** Modify `load_reddit_data` or the processing loop to yield batches of documents directly to the `upsert` call, reducing memory footprint.

### Phase 3: Advanced Optimization (Future)
1.  **Combine Router and Multi-Query LLM Calls**
    *   **Problem:** Two sequential LLM calls before retrieval (Route then Multi-Query).
    *   **Solution:** Create a single prompt that asks the LLM to decide the strategy AND generate alternative queries if needed.

## Verification Plan
-   Measure response time for "Quick" and "Deep" analysis before and after changes.
-   Verify that BM25 still returns relevant results after caching.
-   Verify that multi-query still returns relevant results after parallelization.
