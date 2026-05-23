from langgraph.prebuilt import create_react_agent
from config import get_llm
from .tools import scrape_subreddit_tool

def get_scraper_agent():
    """
    Returns a compiled LangGraph agent equipped with scraping tools.
    """
    llm = get_llm(temperature=0.1)
    
    # The system prompt instructs the agent on its specialized role
    system_prompt = (
        "You are the Data Ingestion Agent (Scraper) for a Reddit Sentiment Analysis system.\n"
        "Your job is to fetch real-time data from Reddit when asked.\n"
        "You MUST use the `scrape_subreddit_tool` to do this.\n"
        "Once the tool returns success, formulate a brief summary of what you scraped and exit."
    )
    
    tools = [scrape_subreddit_tool]
    
    # Create a prebuilt ReAct agent
    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
    return agent
