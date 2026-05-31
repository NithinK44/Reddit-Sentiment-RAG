import sqlite3
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Base database path
DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "reddit_sentiment.db"

def get_connection():
    """Get a connection to the SQLite database."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database tables and virtual FTS5 tables."""
    logger.info(f"Initializing SQLite database at {DB_PATH}")
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Create analysis_history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analysis_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT NOT NULL,
        mode TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        sentiment TEXT NOT NULL,
        confidence REAL NOT NULL,
        net_sentiment_score INTEGER NOT NULL,
        report_id TEXT UNIQUE,
        data_freshness TEXT
    )
    """)
    
    # 2. Create documents table (source of truth for metadata & RAG content)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        post_id TEXT NOT NULL,
        post_title TEXT NOT NULL,
        post_url TEXT NOT NULL,
        post_date TEXT NOT NULL,
        post_score INTEGER NOT NULL,
        post_sort TEXT,
        flair TEXT,
        comment_score INTEGER NOT NULL,
        depth INTEGER NOT NULL,
        type TEXT NOT NULL
    )
    """)
    
    # 3. Create FTS5 virtual table for keyword search
    try:
        cursor.execute("SELECT * FROM documents_fts LIMIT 1")
    except sqlite3.OperationalError:
        # FTS5 table doesn't exist, create it
        cursor.execute("""
        CREATE VIRTUAL TABLE documents_fts USING fts5(
            doc_id UNINDEXED,
            content,
            post_title,
            content='documents'
        )
        """)
        
        # Create triggers to sync FTS5 table automatically with documents table
        cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS tbl_ai AFTER INSERT ON documents BEGIN
          INSERT INTO documents_fts(rowid, doc_id, content, post_title) 
          VALUES (new.rowid, new.doc_id, new.content, new.post_title);
        END;
        """)
        
        cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS tbl_ad AFTER DELETE ON documents BEGIN
          INSERT INTO documents_fts(documents_fts, rowid, doc_id, content, post_title) 
          VALUES('delete', old.rowid, old.doc_id, old.content, old.post_title);
        END;
        """)
        
        cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS tbl_au AFTER UPDATE ON documents BEGIN
          INSERT INTO documents_fts(documents_fts, rowid, doc_id, content, post_title) 
          VALUES('delete', old.rowid, old.doc_id, old.content, old.post_title);
          INSERT INTO documents_fts(rowid, doc_id, content, post_title) 
          VALUES (new.rowid, new.doc_id, new.content, new.post_title);
        END;
        """)
        
        # Populate with existing documents
        cursor.execute("""
        INSERT INTO documents_fts(rowid, doc_id, content, post_title)
        SELECT rowid, doc_id, content, post_title FROM documents;
        """)
        
    conn.commit()
    conn.close()
    logger.info("SQLite database tables initialized successfully.")

# ============================================================================
# HISTORY HELPERS
# ============================================================================

def save_analysis_history(query: str, mode: str, timestamp: str, sentiment: str, 
                          confidence: float, net_sentiment_score: int, 
                          report_id: str, data_freshness: dict) -> bool:
    """Save analysis record to sqlite history."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        freshness_str = json.dumps(data_freshness)
        cursor.execute("""
        INSERT OR REPLACE INTO analysis_history 
        (query, mode, timestamp, sentiment, confidence, net_sentiment_score, report_id, data_freshness)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (query, mode, timestamp, sentiment, confidence, net_sentiment_score, report_id, freshness_str))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error saving analysis history: {e}")
        return False
    finally:
        conn.close()

def get_analysis_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent analysis records."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT query, mode, timestamp, sentiment, confidence, net_sentiment_score, report_id, data_freshness 
        FROM analysis_history 
        ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        result = []
        for r in rows:
            freshness = {}
            if r["data_freshness"]:
                try:
                    freshness = json.loads(r["data_freshness"])
                except Exception:
                    pass
            result.append({
                "query": r["query"],
                "mode": r["mode"],
                "timestamp": r["timestamp"],
                "sentiment": r["sentiment"],
                "confidence": r["confidence"],
                "net_sentiment_score": r["net_sentiment_score"],
                "report_id": r["report_id"],
                "data_freshness": freshness
            })
        return result
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return []
    finally:
        conn.close()

# ============================================================================
# DOCUMENT PERSISTENCE HELPERS
# ============================================================================

def upsert_documents(docs_list: List[Dict[str, Any]]):
    """Upsert a list of document dicts into SQLite."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        for doc in docs_list:
            meta = doc["metadata"]
            cursor.execute("""
            INSERT OR REPLACE INTO documents 
            (doc_id, content, post_id, post_title, post_url, post_date, post_score, post_sort, flair, comment_score, depth, type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc["doc_id"],
                doc["content"],
                meta.get("post_id", ""),
                meta.get("post_title", ""),
                meta.get("post_url", ""),
                meta.get("post_date", ""),
                int(meta.get("post_score", 0)),
                meta.get("post_sort", ""),
                meta.get("flair", ""),
                int(meta.get("comment_score", 0)),
                int(meta.get("depth", 0)),
                meta.get("type", "comment")
            ))
        conn.commit()
    except Exception as e:
        logger.error(f"Error upserting documents: {e}")
        conn.rollback()
    finally:
        conn.close()

def search_documents_fts(keyword_query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Execute FTS5 search on documents, ranking by BM25 match score."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        # Clean query for FTS5 (escape special chars, keep alphanumeric + spaces)
        clean_query = re.sub(r'[^\w\s]', '', keyword_query).strip()
        if not clean_query:
            return []
            
        # Format for FTS5 prefix matching on terms
        fts_query = " OR ".join([f"{term}*" for term in clean_query.split() if term])
        if not fts_query:
            return []

        cursor.execute("""
        SELECT d.*, bm25(documents_fts) as rank
        FROM documents d
        JOIN documents_fts fts ON d.doc_id = fts.doc_id
        WHERE documents_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """, (fts_query, limit))
        
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"FTS5 Search failed: {e}")
        return []
    finally:
        conn.close()



# Auto-initialize DB on import
try:
    init_db()
except Exception as init_err:
    logger.error(f"Failed to auto-init SQLite DB on import: {init_err}")

