"""
Centralized Configuration for the Reddit Sentiment RAG System
--------------------------------------------------------------
All shared constants, paths, model names, and initialization helpers
live here so every module imports from one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PATHS & CONSTANTS
# ============================================================================

DATA_DIR = Path("./rag_ready_data")
CHROMA_PERSIST_DIR = Path("./chroma_db")
COLLECTION_NAME = "reddit_sentiment"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-r1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Retrieval defaults
DEFAULT_N_RESULTS = 10
MIN_COMMENT_SCORE = 1
MAX_CHUNK_LENGTH = 8000

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_embedding_function():
    """Get the Sentence Transformer embedding function for ChromaDB."""
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_chroma_collection():
    """Get or create the ChromaDB collection with persistence."""
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"}
    )


def get_llm(temperature: float = 0.3):
    """Get the LangChain LLM instance (OpenRouter DeepSeek)."""
    from langchain_openai import ChatOpenAI
    
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY not found. "
            "Set it in your .env file or as an environment variable. "
        )
    
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
    )


def get_langchain_vectorstore():
    """Get the ChromaDB vectorstore as a LangChain-compatible object."""
    from langchain_chroma import Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
    )
