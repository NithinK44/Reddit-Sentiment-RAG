from langgraph.prebuilt import create_react_agent
from config import get_llm
from .tools import embed_data_tool

def get_processor_agent():
    """
    Returns a compiled LangGraph agent equipped with embedding tools.
    """
    llm = get_llm(temperature=0.1)
    
    system_prompt = (
        "You are the NLP Processor Agent for a Reddit Sentiment Analysis system.\n"
        "Your job is to take newly scraped data and embed it into the vector database.\n"
        "You MUST use the `embed_data_tool` to do this.\n"
        "Once the tool returns success, summarize the number of documents embedded and exit."
    )
    
    tools = [embed_data_tool]
    
    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
    return agent
