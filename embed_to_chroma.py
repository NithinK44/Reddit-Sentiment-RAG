"""
Reddit Data Embedder for ChromaDB
---------------------------------
This script loads Reddit posts and comments from JSON files,
creates embeddings, and stores them in ChromaDB for sentiment analysis retrieval.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Generator

import chromadb
from chromadb.utils import embedding_functions


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("./rag_ready_data")
CHROMA_PERSIST_DIR = Path("./chroma_db")
COLLECTION_NAME = "reddit_sentiment"

# Minimum score threshold - filter out low-quality comments
MIN_COMMENT_SCORE = 1

# Chunk size limits (ChromaDB has limits on document size)
MAX_CHUNK_LENGTH = 8000


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_doc_id(text: str, metadata: dict) -> str:
    """Generate a unique document ID based on content and metadata."""
    unique_str = f"{metadata.get('post_id', '')}_{metadata.get('type', '')}_{metadata.get('depth', 0)}_{text[:200]}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:16]


def flatten_comments(
    comments: list, 
    post_meta: dict, 
    depth: int = 0, 
    parent_context: str = ""
) -> Generator[dict, None, None]:
    """
    Recursively flatten nested comments into individual documents.
    Preserves context by including parent comment snippets.
    """
    for comment in comments:
        body = comment.get("body", "").strip()
        score = comment.get("score", 0)
        
        # Skip low-quality or deleted comments
        if not body or score < MIN_COMMENT_SCORE:
            continue
        if body in ["[deleted]", "[removed]"]:
            continue
        
        # Build context-aware content
        if parent_context:
            contextual_content = f"[Reply to: {parent_context[:200]}...]\n\n{body}"
        else:
            contextual_content = body
        
        # Truncate if too long
        if len(contextual_content) > MAX_CHUNK_LENGTH:
            contextual_content = contextual_content[:MAX_CHUNK_LENGTH] + "..."
        
        yield {
            "content": contextual_content,
            "metadata": {
                "post_id": post_meta.get("post_id", ""),
                "post_title": post_meta.get("title", ""),
                "post_url": post_meta.get("url", ""),
                "post_date": post_meta.get("date", ""),
                "post_score": post_meta.get("score", 0),
                "post_sort": post_meta.get("sort", ""),
                "comment_score": score,
                "depth": depth,
                "type": "comment"
            }
        }
        
        # Process nested replies
        replies = comment.get("replies", [])
        if replies:
            yield from flatten_comments(
                replies, 
                post_meta, 
                depth=depth + 1,
                parent_context=body
            )


def load_reddit_data(data_dir: Path) -> Generator[dict, None, None]:
    """
    Load all JSON files from the data directory and yield flattened documents.
    """
    json_files = list(data_dir.glob("*.json"))
    print(f"📂 Found {len(json_files)} JSON files to process")
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            meta = data.get("meta", {})
            content = data.get("content", {})
            
            # Extract post ID from filename
            post_id = json_file.stem.split("_")[0]
            meta["post_id"] = post_id
            
            # Yield the main post body if it exists
            post_body = content.get("post_body", "").strip()
            if post_body:
                yield {
                    "content": f"[POST] {meta.get('title', '')}\n\n{post_body}",
                    "metadata": {
                        "post_id": post_id,
                        "post_title": meta.get("title", ""),
                        "post_url": meta.get("url", ""),
                        "post_date": meta.get("date", ""),
                        "post_score": meta.get("score", 0),
                        "post_sort": meta.get("sort", ""),
                        "comment_score": meta.get("score", 0),
                        "depth": 0,
                        "type": "post"
                    }
                }
            
            # Yield flattened comments
            comments = content.get("comments", [])
            yield from flatten_comments(comments, meta)
            
            print(f"  ✅ Processed: {json_file.name}")
            
        except Exception as e:
            print(f"  ❌ Error processing {json_file.name}: {e}")


def embed_to_chromadb():
    """
    Main function to embed all Reddit data into ChromaDB.
    """
    print("\n" + "="*60)
    print("🚀 REDDIT DATA EMBEDDER FOR CHROMADB")
    print("="*60 + "\n")
    
    # Initialize ChromaDB client with persistence
    print("📦 Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    
    # Use default embedding function (all-MiniLM-L6-v2)
    # You can switch to OpenAI embeddings if preferred
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create or get collection
    # Using cosine similarity for semantic search
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"📊 Collection '{COLLECTION_NAME}' ready")
    print(f"   Current document count: {collection.count()}\n")
    
    # Load and process data
    print("📥 Loading Reddit data...\n")
    
    documents = []
    metadatas = []
    ids = []
    seen_ids = set()  # Track unique IDs to avoid duplicates
    
    for doc in load_reddit_data(DATA_DIR):
        doc_id = generate_doc_id(doc["content"], doc["metadata"])
        
        # Skip if we've already seen this ID
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        
        documents.append(doc["content"])
        metadatas.append(doc["metadata"])
        ids.append(doc_id)
    
    if not documents:
        print("\n⚠️  No new documents to add")
        return
    
    # Batch insert (ChromaDB recommends batches of ~5000)
    print(f"\n📤 Embedding {len(documents)} documents...")
    
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        collection.upsert(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )
        print(f"   ✅ Batch {i//batch_size + 1}: Added {len(batch_docs)} documents")
    
    print(f"\n✨ DONE! Total documents in collection: {collection.count()}")
    print(f"💾 Data persisted to: {CHROMA_PERSIST_DIR.absolute()}")


def query_sentiment(query: str, n_results: int = 5, min_score: int = 0):
    """
    Query the ChromaDB collection for relevant documents.
    Useful for sentiment analysis and opinion retrieval.
    
    Args:
        query: The search query (e.g., "What do fans think about Garnacho?")
        n_results: Number of results to return
        min_score: Minimum comment score filter
    
    Returns:
        Query results with documents and metadata
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    
    # Build where filter for minimum score
    where_filter = None
    if min_score > 0:
        where_filter = {"comment_score": {"$gte": min_score}}
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    
    return results


def print_query_results(results: dict):
    """Pretty print query results for debugging/testing."""
    print("\n" + "="*60)
    print("🔍 QUERY RESULTS")
    print("="*60)
    
    if not results["documents"] or not results["documents"][0]:
        print("No results found.")
        return
    
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\n--- Result {i+1} (Distance: {dist:.4f}) ---")
        print(f"📌 Post: {meta.get('post_title', 'N/A')}")
        print(f"📅 Date: {meta.get('post_date', 'N/A')}")
        print(f"⬆️  Score: {meta.get('comment_score', 0)}")
        print(f"📝 Content:\n{doc[:500]}{'...' if len(doc) > 500 else ''}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Embed all data
    embed_to_chromadb()
    
    # Test query
    print("\n" + "="*60)
    print("🧪 TESTING RETRIEVAL")
    print("="*60)
    
    test_query = "What is the sentiment about team ownership and management?"
    print(f"\n🔎 Query: '{test_query}'")
    
    results = query_sentiment(test_query, n_results=3, min_score=5)
    print_query_results(results)
