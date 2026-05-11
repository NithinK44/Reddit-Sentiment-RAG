"""
Reddit Data Embedder for ChromaDB
---------------------------------
Loads Reddit posts/comments from JSON, creates embeddings, stores in ChromaDB.
"""

import json
import hashlib
from pathlib import Path
from typing import Generator

import chromadb
from chromadb.utils import embedding_functions

from config import (
    DATA_DIR, CHROMA_PERSIST_DIR, COLLECTION_NAME,
    EMBEDDING_MODEL, MIN_COMMENT_SCORE, MAX_CHUNK_LENGTH,
)


def generate_doc_id(text: str, metadata: dict) -> str:
    unique_str = f"{metadata.get('post_id', '')}_{metadata.get('type', '')}_{metadata.get('depth', 0)}_{text[:200]}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:16]


def flatten_comments(comments, post_meta, depth=0, parent_context=""):
    for comment in comments:
        body = comment.get("body", "").strip()
        score = comment.get("score", 0)
        if not body or score < MIN_COMMENT_SCORE or body in ["[deleted]", "[removed]"]:
            continue
        contextual_content = f"[Reply to: {parent_context[:200]}...]\n\n{body}" if parent_context else body
        if len(contextual_content) > MAX_CHUNK_LENGTH:
            contextual_content = contextual_content[:MAX_CHUNK_LENGTH] + "..."
        yield {
            "content": contextual_content,
            "metadata": {
                "post_id": post_meta.get("post_id", ""), "post_title": post_meta.get("title", ""),
                "post_url": post_meta.get("url", ""), "post_date": post_meta.get("date", ""),
                "post_score": int(post_meta.get("score", 0)), "post_sort": post_meta.get("sort", ""),
                "flair": post_meta.get("flair", "Unknown"), "comment_score": int(score),
                "depth": depth, "type": "comment",
            },
        }
        replies = comment.get("replies", [])
        if replies:
            yield from flatten_comments(replies, post_meta, depth + 1, body)


def load_reddit_data(data_dir: Path):
    json_files = list(data_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files to process")
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("meta", {})
            content = data.get("content", {})
            post_id = json_file.stem.split("_")[0]
            meta["post_id"] = post_id
            post_body = content.get("post_body", "").strip()
            if post_body:
                yield {
                    "content": f"[POST] {meta.get('title', '')}\n\n{post_body}",
                    "metadata": {
                        "post_id": post_id, "post_title": meta.get("title", ""),
                        "post_url": meta.get("url", ""), "post_date": meta.get("date", ""),
                        "post_score": int(meta.get("score", 0)), "post_sort": meta.get("sort", ""),
                        "flair": meta.get("flair", "Unknown"), "comment_score": int(meta.get("score", 0)),
                        "depth": 0, "type": "post",
                    },
                }
            yield from flatten_comments(content.get("comments", []), meta)
            print(f"  Processed: {json_file.name}")
        except Exception as e:
            print(f"  Error processing {json_file.name}: {e}")


_embed_state = {
    "running": False, "total_files": 0, "total_docs": 0,
    "processed_docs": 0, "error": None, "finished": False,
}


def get_embed_status():
    return dict(_embed_state)


def embed_to_chromadb():
    global _embed_state
    if _embed_state["running"]:
        return {"error": "An embed job is already running"}
    _embed_state = {"running": True, "total_files": 0, "total_docs": 0, "processed_docs": 0, "error": None, "finished": False}
    try:
        if not DATA_DIR.exists():
            _embed_state.update(error=f"Data directory not found: {DATA_DIR}", running=False, finished=True)
            return {"error": _embed_state["error"]}
        json_files = list(DATA_DIR.glob("*.json"))
        _embed_state["total_files"] = len(json_files)
        if not json_files:
            _embed_state.update(error="No JSON files found. Scrape data first.", running=False, finished=True)
            return {"error": _embed_state["error"]}

        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn, metadata={"hnsw:space": "cosine"})

        documents, metadatas, ids, seen_ids = [], [], [], set()
        for doc in load_reddit_data(DATA_DIR):
            doc_id = generate_doc_id(doc["content"], doc["metadata"])
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            documents.append(doc["content"])
            metadatas.append(doc["metadata"])
            ids.append(doc_id)
        _embed_state["total_docs"] = len(documents)

        if not documents:
            _embed_state.update(running=False, finished=True)
            return {"success": True, "documents_added": 0, "total": collection.count()}

        batch_size = 100
        for i in range(0, len(documents), batch_size):
            collection.upsert(documents=documents[i:i+batch_size], metadatas=metadatas[i:i+batch_size], ids=ids[i:i+batch_size])
            _embed_state["processed_docs"] = min(i + batch_size, len(documents))

        final_count = collection.count()
        _embed_state.update(running=False, finished=True)
        return {"success": True, "documents_added": len(documents), "total": final_count}
    except Exception as e:
        _embed_state.update(error=str(e), running=False, finished=True)
        return {"error": str(e)}


def count_json_files():
    """Count available JSON files in the data directory."""
    if not DATA_DIR.exists():
        return 0
    return len(list(DATA_DIR.glob("*.json")))
