"""
Centralized Configuration for the Reddit Sentiment RAG System
--------------------------------------------------------------
All shared constants, paths, model names, and initialization helpers
live here so every module imports from one place.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix encoding for Windows console (emojis)
import sys
try:
    if sys.stdout and hasattr(sys.stdout, 'reconfigure') and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    # Fail silently if reconfigure is not supported or causes Errno 22
    pass

# Find the .env file relative to this script
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Load environment variables with override=True to ensure .env values win
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    print(f"[OK] Loaded configuration from {ENV_PATH}")
else:
    print(f"[WARN] .env file not found at {ENV_PATH}")
    load_dotenv() # Fallback to default search

# ============================================================================
# PATHS & CONSTANTS
# ============================================================================

DATA_DIR = BASE_DIR / "data" / "rag_ready_data"
CHROMA_PERSIST_DIR = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "reddit_sentiment"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Provider Toggle (can be overridden in app.py)
USE_GOOGLE_STUDIO = os.getenv("USE_GOOGLE_STUDIO", "False").lower() == "true"

# OpenRouter Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-r1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Google AI Studio Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")

def print_config_status():
    """Print the current configuration status."""
    if USE_GOOGLE_STUDIO:
        print(f"🤖 Provider: Google AI Studio")
        print(f"🤖 Model: {GOOGLE_MODEL}")
        if GOOGLE_API_KEY:
            print(f"🔑 API Key: {'*' * 5}{GOOGLE_API_KEY[-4:]} (from .env)")
        else:
            print("❌ API Key: Google API Key not found")
    else:
        print(f"🤖 Provider: OpenRouter")
        print(f"🤖 Model: {LLM_MODEL}")
        if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_api_key_here":
            print(f"🔑 API Key: {'*' * 5}{OPENROUTER_API_KEY[-4:]} (from .env)")
        else:
            print("❌ API Key: OpenRouter API Key not found")

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
    """Get the LangChain LLM instance (Google Gemini or OpenRouter)."""
    
    if USE_GOOGLE_STUDIO:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found. Set it in your .env file.")
            
        return ChatGoogleGenerativeAI(
            model=GOOGLE_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            max_output_tokens=4000,
        )
    else:
        from langchain_openai import ChatOpenAI
        
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not found. "
                "Set it in your .env file or as an environment variable. "
            )
        
        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=4000,
        )


def get_langchain_vectorstore():
    """Get the ChromaDB vectorstore as a LangChain-compatible object."""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
    )
