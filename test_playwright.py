import os
from dotenv import load_dotenv
load_dotenv()
REDDIT_SESSION_COOKIE = os.getenv("REDDIT_SESSION_COOKIE")

def validate_subreddit(subreddit: str) -> dict:
    url = f"https://www.reddit.com/r/{subreddit}/about.json"
    
    try:
        from playwright.sync_api import sync_playwright
        import json
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            if REDDIT_SESSION_COOKIE:
                context.add_cookies([{
                    "name": "reddit_session",
                    "value": REDDIT_SESSION_COOKIE,
                    "domain": ".reddit.com",
                    "path": "/"
                }])
            
            page = context.new_page()
            response = page.goto(url, wait_until="commit")
            
            if response.status == 404:
                return {"success": False, "error": f"Subreddit 'r/{subreddit}' does not exist."}
            elif response.status == 403:
                return {"success": False, "error": f"Subreddit 'r/{subreddit}' is private or banned."}
            elif response.status != 200:
                return {"success": False, "error": f"Reddit API error: {response.status}"}
            
            try:
                data = response.json()
            except Exception:
                text = page.locator("pre").inner_text() if page.locator("pre").count() > 0 else page.locator("body").inner_text()
                data = json.loads(text)
                
            browser.close()

        if 'kind' not in data or data['kind'] != 't5':
            return {"success": False, "error": f"Subreddit 'r/{subreddit}' does not exist."}
            
        sub_data = data.get('data', {})
        if not sub_data.get('display_name'):
            return {"success": False, "error": f"Subreddit 'r/{subreddit}' does not exist."}
        
        return {
            "success": True,
            "metadata": {
                "name": sub_data.get('display_name'),
                "title": sub_data.get('title'),
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    print(validate_subreddit("playstation"))
