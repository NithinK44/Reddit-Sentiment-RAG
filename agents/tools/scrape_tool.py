from langchain_core.tools import tool
from scraper.reddit_scraper import scrape_subreddit

@tool
def scrape_subreddit_tool(subreddit: str, post_limit: int = 25, sort_by: str = "top", time_filter: str = "year") -> dict:
    """
    Scrapes posts and comments from a specified subreddit.
    Use this when you need real-time data or when the user asks about something very recent.
    
    Args:
        subreddit: The name of the subreddit to scrape (without 'r/').
        post_limit: Number of posts to scrape (default: 25).
        sort_by: How to sort the posts ('hot', 'new', 'top', 'best').
        time_filter: Time filter if sort_by is 'top' ('day', 'week', 'month', 'year', 'all').
        
    Returns:
        Dict containing success status and number of scraped posts.
    """
    return scrape_subreddit(
        subreddit=subreddit,
        post_limit=post_limit,
        sort_by=sort_by,
        time_filter=time_filter
    )
