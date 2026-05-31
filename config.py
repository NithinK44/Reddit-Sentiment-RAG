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
else:
    load_dotenv() # Fallback to default search

from core.logging_config import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

# ============================================================================
# LANGSMITH TRACING
# ============================================================================

LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "reddit-sentiment-rag")

if LANGSMITH_TRACING and LANGSMITH_API_KEY:
    # LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY are already exported via load_dotenv
    # LangChain picks them up automatically — log confirmation here.
    import logging as _ls_log
    _ls_log.getLogger(__name__).info(
        f"✅ LangSmith tracing enabled → project='{LANGSMITH_PROJECT}' "
        f"key=...{LANGSMITH_API_KEY[-6:]}"
    )
elif LANGSMITH_TRACING and not LANGSMITH_API_KEY:
    import logging as _ls_log
    _ls_log.getLogger(__name__).warning(
        "⚠️ LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is missing — tracing disabled"
    )


if ENV_PATH.exists():
    logger.info(f"Loaded configuration from {ENV_PATH}")
else:
    logger.warning(f".env file not found at {ENV_PATH}")

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
DEEP_ANALYSIS_MODEL = os.getenv("DEEP_ANALYSIS_MODEL", GOOGLE_MODEL)

# Reddit Configuration
REDDIT_SESSION_COOKIE = os.getenv("REDDIT_SESSION_COOKIE", "")

def print_config_status():
    """Print the current configuration status."""
    if USE_GOOGLE_STUDIO:
        logger.info(f"🤖 Provider: Google AI Studio")
        logger.info(f"🤖 Model: {GOOGLE_MODEL}")
        if GOOGLE_API_KEY:
            logger.info(f"🔑 API Key: {'*' * 5}{GOOGLE_API_KEY[-4:]} (from .env)")
        else:
            logger.warning("❌ API Key: Google API Key not found")
    else:
        logger.info(f"🤖 Provider: OpenRouter")
        logger.info(f"🤖 Model: {LLM_MODEL}")
        if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_api_key_here":
            logger.info(f"🔑 API Key: {'*' * 5}{OPENROUTER_API_KEY[-4:]} (from .env)")
        else:
            logger.warning("❌ API Key: OpenRouter API Key not found")

# Retrieval defaults
DEFAULT_N_RESULTS = 10
MIN_COMMENT_SCORE = 1
MAX_CHUNK_LENGTH = 8000

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

_embedding_model_instance = None

def get_embedding_function():
    """Get the Sentence Transformer embedding function for ChromaDB."""
    global _embedding_model_instance
    if _embedding_model_instance is None:
        from chromadb.utils import embedding_functions
        _embedding_model_instance = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _embedding_model_instance


def get_chroma_collection(collection_name=COLLECTION_NAME):
    """Get or create the ChromaDB collection with persistence."""
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"}
    )


def get_llm(temperature: float = 0.3, model_name: str = None):
    """Get the LangChain LLM instance (Google Gemini or OpenRouter)."""
    
    if USE_GOOGLE_STUDIO:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found. Set it in your .env file.")
            
        active_model = model_name or GOOGLE_MODEL
            
        return ChatGoogleGenerativeAI(
            model=active_model,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            max_output_tokens=10000,
        )
    else:
        from langchain_openai import ChatOpenAI
        
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not found. "
                "Set it in your .env file or as an environment variable. "
            )
            
        active_model = model_name or LLM_MODEL
        
        return ChatOpenAI(
            model=active_model,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=10000,
        )


_langchain_embeddings_instance = None
_vectorstores = {}

def get_langchain_vectorstore(collection_name=COLLECTION_NAME):
    """Get the ChromaDB vectorstore as a LangChain-compatible object."""
    global _langchain_embeddings_instance, _vectorstores
    
    if collection_name in _vectorstores:
        return _vectorstores[collection_name]
        
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    
    if _langchain_embeddings_instance is None:
        _langchain_embeddings_instance = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    vs = Chroma(
        collection_name=collection_name,
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=_langchain_embeddings_instance,
    )
    _vectorstores[collection_name] = vs
    return vs


def load_prompt_text(filename: str) -> str:
    """Load prompt template text from the prompts directory."""
    prompt_path = BASE_DIR / "prompts" / filename
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load prompt from {prompt_path}: {e}")
        raise
