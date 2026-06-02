"""
Reddit Data Embedder for ChromaDB
---------------------------------
Loads Reddit posts/comments from JSON, creates embeddings, stores in ChromaDB.
"""

import json
import hashlib
import threading
from pathlib import Path
from typing import Generator
import logging

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.db import upsert_documents

logger = logging.getLogger(__name__)
from core.manifest_manager import ManifestManager
from rag.retriever import invalidate_bm25_cache

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


post_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len
)

def extract_docs_from_file(json_file: Path) -> Generator:
    """
    Extracts embeddable documents from a scraped Reddit JSON file.
    Scraper saves: { "meta": { title, url, score, flair, date, sort },
                     "content": { post_body, comments: [...] } }
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # ── Read from the correct nested structure produced by reddit_scraper ──
        meta_block = raw.get("meta", {})
        content_block = raw.get("content", {})

        # Derive a stable post_id from the filename (format: <id>_<title>.json)
        post_id = json_file.stem.split("_")[0]

        post_meta = {
            "post_id": post_id,
            "title": meta_block.get("title", ""),
            "url": meta_block.get("url", ""),
            "date": meta_block.get("date", ""),
            "score": int(meta_block.get("score", 0)),
            "flair": meta_block.get("flair") or "Unknown",
            "sort": meta_block.get("sort", ""),
        }

        # Yield main post body if it has text (semantically chunked if long)
        post_body = content_block.get("post_body", "").strip()
        if post_body:
            full_text = f"Title: {post_meta['title']}\n\n{post_body}"
            if len(full_text) <= 2000:
                yield {
                    "content": full_text,
                    "metadata": {
                        "post_id": post_meta["post_id"],
                        "post_title": post_meta["title"],
                        "post_url": post_meta["url"],
                        "post_date": post_meta["date"],
                        "post_score": post_meta["score"],
                        "post_sort": post_meta["sort"],
                        "flair": post_meta["flair"],
                        "comment_score": post_meta["score"],
                        "depth": 0,
                        "type": "post",
                    },
                }
            else:
                chunks = post_splitter.split_text(post_body)
                for idx, chunk in enumerate(chunks):
                    chunk_content = f"Title: {post_meta['title']} (Part {idx+1})\n\n{chunk}"
                    yield {
                        "content": chunk_content,
                        "metadata": {
                            "post_id": post_meta["post_id"],
                            "post_title": post_meta["title"],
                            "post_url": post_meta["url"],
                            "post_date": post_meta["date"],
                            "post_score": post_meta["score"],
                            "post_sort": post_meta["sort"],
                            "flair": post_meta["flair"],
                            "comment_score": post_meta["score"],
                            "depth": 0,
                            "type": f"post_part_{idx+1}",
                        },
                    }

        yield from flatten_comments(content_block.get("comments", []), post_meta)
    except Exception as e:
        logger.error(f"  Error processing {json_file.name}: {e}")


_embed_state = {
    "running": False, "total_files": 0, "total_docs": 0,
    "processed_docs": 0, "error": None, "finished": False,
}
_embed_lock = threading.Lock()


def get_embed_status():
    """Return the current embed job status with progress calculation."""
    status = dict(_embed_state)
    
    # Calculate progress percentage
    if status["total_docs"] > 0:
        status["progress"] = (status["processed_docs"] / status["total_docs"]) * 100
    elif status["total_files"] > 0 and not status["total_docs"]:
        # If we have files but haven't counted docs yet, give a small progress
        status["progress"] = 5
    else:
        status["progress"] = 0
        
    # Build human-readable message
    if status["running"]:
        if status["total_docs"] > 0:
            status["message"] = f"Embedding documents... ({status['processed_docs']}/{status['total_docs']})"
        else:
            status["message"] = f"Processing {status['total_files']} files..."
    elif status["error"]:
        status["message"] = f"Error: {status['error']}"
    elif status["finished"]:
        status["message"] = f"Finished! Embedded {status['processed_docs']} documents."
    else:
        status["message"] = "Ready to embed."
        
    return status


def _set_embed_state(**kwargs):
    """Thread-safe helper to update _embed_state fields."""
    with _embed_lock:
        _embed_state.update(kwargs)


def embed_to_chromadb(subreddit: str = "reddit_sentiment"):
    global _embed_state
    with _embed_lock:
        if _embed_state["running"]:
            return {"error": "An embed job is already running"}
        _embed_state = {"running": True, "total_files": 0, "total_docs": 0, "processed_docs": 0, "error": None, "finished": False}
    try:
        data_path = DATA_DIR / subreddit
        if not data_path.exists():
            err = f"Data directory not found: {data_path}"
            _set_embed_state(error=err, running=False, finished=True)
            return {"error": err}
        json_files = list(data_path.glob("*.json"))
        _set_embed_state(total_files=len(json_files))
        if not json_files:
            err = "No JSON files found. Scrape data first."
            _set_embed_state(error=err, running=False, finished=True)
            return {"error": err}

        from config import get_embedding_function
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        embedding_fn = get_embedding_function()
        collection = client.get_or_create_collection(name=subreddit, embedding_function=embedding_fn, metadata={"hnsw:space": "cosine"})

        manifest = ManifestManager(data_path)

        seen_ids = set()
        batch_size = 100
        processed_count = 0
        
        # We don't know total_docs upfront without reading all files,
        # so we set it to 0 and update it at the end.
        _embed_state["total_docs"] = 0 
        
        for json_file in json_files:
            if not manifest.is_file_changed(json_file):
                logger.info(f"  Skipped (unchanged): {json_file.name}")
                continue
                
            current_docs = []
            current_metadatas = []
            current_ids = []
            current_sqlite_docs = []
            
            for doc in extract_docs_from_file(json_file):
                doc_id = generate_doc_id(doc["content"], doc["metadata"])
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                
                current_docs.append(doc["content"])
                current_metadatas.append(doc["metadata"])
                current_ids.append(doc_id)
                current_sqlite_docs.append({
                    "doc_id": doc_id,
                    "content": doc["content"],
                    "metadata": doc["metadata"]
                })
                
                if len(current_docs) >= batch_size:
                    collection.upsert(documents=current_docs, metadatas=current_metadatas, ids=current_ids)
                    upsert_documents(current_sqlite_docs)
                    processed_count += len(current_docs)
                    _set_embed_state(processed_docs=processed_count)
                    current_docs, current_metadatas, current_ids = [], [], []
                    current_sqlite_docs = []

            # Upsert remaining for this file
            if current_docs:
                collection.upsert(documents=current_docs, metadatas=current_metadatas, ids=current_ids)
                upsert_documents(current_sqlite_docs)
                processed_count += len(current_docs)
                _set_embed_state(processed_docs=processed_count)

            manifest.update_file(json_file)
            logger.info(f"  Embedded and manifested: {json_file.name}")
            
        _set_embed_state(total_docs=processed_count)

        # Invalidate the BM25 cache so the new docs are indexed on next search
        invalidate_bm25_cache(subreddit)

        if processed_count == 0:
            _set_embed_state(running=False, finished=True)
            return {"success": True, "documents_added": 0, "total": collection.count()}

        final_count = collection.count()
        _set_embed_state(running=False, finished=True)
        return {"success": True, "documents_added": processed_count, "total": final_count}
    except Exception as e:
        _set_embed_state(error=str(e), running=False, finished=True)
        return {"error": str(e)}


def count_json_files(subreddit: str = ""):
    """Count available JSON files in the data directory."""
    if subreddit:
        # Subreddit files live directly in DATA_DIR/<subreddit>/*.json
        data_path = DATA_DIR / subreddit
        pattern = "*.json"
    else:
        # No subreddit specified — count all JSON files in all subdirectories
        data_path = DATA_DIR
        pattern = "**/*.json"
    if not data_path.exists():
        return 0
    return len(list(data_path.glob(pattern)))
