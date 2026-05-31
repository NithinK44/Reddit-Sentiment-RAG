from playwright.sync_api import sync_playwright

def test_comments():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        print("Navigating to thread...")
        page.goto("https://www.reddit.com/r/Python/comments/1tsmzix/", wait_until="domcontentloaded")
        
        try:
            page.wait_for_selector("shreddit-comment", timeout=10000)
            comments = page.locator("shreddit-comment").all()
            print(f"Found {len(comments)} comments on the page.")
            for c in comments[:3]:
                author = c.get_attribute("author")
                score = c.get_attribute("score")
                # text content is inside a div or p
                text = c.locator("div[slot='comment']").inner_text() if c.locator("div[slot='comment']").count() > 0 else "No text"
                print(f"Author: {author}, Score: {score}, Text: {text[:50]}...")
        except Exception as e:
            print("Error finding comments:", e)
            
        browser.close()

if __name__ == "__main__":
    test_comments()
