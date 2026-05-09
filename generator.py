"""
RAG Generator — Phase 1: The Generative Layer
----------------------------------------------
Builds a LangChain RAG chain that connects the ChromaDB retriever
to a Gemini LLM with sentiment-aware prompts.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import get_llm, get_langchain_vectorstore, DEFAULT_N_RESULTS


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

RAG_SYSTEM_PROMPT = """You are a Reddit Sentiment Analyst specializing in football/soccer fan communities, 
specifically r/ManchesterUnited. You analyze retrieved Reddit comments and posts to provide 
insightful sentiment analysis.

RULES:
1. ONLY use the retrieved context below to form your analysis. Do NOT hallucinate or make up information.
2. Weight comments from 🛡️ [MODERATOR] and 🔴 [OP/CREATOR] higher — these carry authority.
3. Pay attention to comment scores — higher-scored comments represent community consensus.
4. Detect sarcasm, irony, and dark humor (common in football fan communities).
5. Include direct quotes when they powerfully represent the sentiment.
6. Note disagreements and minority opinions separately from the majority view.
7. Be specific about which posts/threads the sentiment comes from.
8. If the context doesn't contain relevant information, say so honestly.

FORMAT YOUR RESPONSE:
- Start with a 2-3 sentence executive summary
- List key themes with sentiment indicators (🟢 Positive, 🔴 Negative, 🟡 Mixed, ⚪ Neutral)
- Include 2-3 notable quotes
- End with a confidence assessment of your analysis

CONTEXT (Retrieved Reddit posts and comments):
{context}
"""

RAG_HUMAN_PROMPT = """Analyze the sentiment for: {question}"""


# ============================================================================
# RAG CHAIN
# ============================================================================

def format_docs(docs) -> str:
    """Format retrieved documents into a readable context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = f"--- Document {i} ---"
        post_info = f"📌 Post: {meta.get('post_title', 'Unknown')}"
        date_info = f"📅 Date: {meta.get('post_date', 'Unknown')}"
        score_info = f"⬆️ Score: {meta.get('comment_score', 0)}"
        flair_info = f"🏷️ Flair: {meta.get('flair', 'Unknown')}"
        type_info = f"📝 Type: {meta.get('type', 'comment')}"
        
        formatted.append(
            f"{header}\n{post_info}\n{date_info}\n{score_info}\n{flair_info}\n{type_info}\n\n{doc.page_content}"
        )
    
    return "\n\n".join(formatted)


def create_rag_chain(n_results: int = DEFAULT_N_RESULTS):
    """
    Build and return the RAG chain.
    
    Returns:
        A LangChain runnable chain that takes a question and returns an analysis.
    """
    llm = get_llm(temperature=0.3)
    vectorstore = get_langchain_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": n_results})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", RAG_HUMAN_PROMPT),
    ])
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def query_rag(question: str, n_results: int = DEFAULT_N_RESULTS) -> str:
    """
    Convenience function to run a single RAG query.
    
    Args:
        question: The user's question about Reddit sentiment
        n_results: Number of documents to retrieve
    
    Returns:
        The LLM's sentiment analysis as a string
    """
    chain = create_rag_chain(n_results=n_results)
    return chain.invoke(question)


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 TESTING RAG CHAIN")
    print("=" * 60)
    
    test_question = "What is the sentiment about team ownership and management?"
    print(f"\n🔎 Query: '{test_question}'\n")
    
    try:
        result = query_rag(test_question, n_results=5)
        print(result)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your OPENROUTER_API_KEY is set in .env")
