from playwright.sync_api import sync_playwright

def test_html_scrape():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        print("Navigating to r/python...")
        response = page.goto("https://www.reddit.com/r/python/", wait_until="domcontentloaded")
        print(f"Status: {response.status}")
        
        # Wait for posts to load
        try:
            page.wait_for_selector("shreddit-post", timeout=10000)
            posts = page.locator("shreddit-post").all()
            print(f"Found {len(posts)} posts.")
            for post in posts[:3]:
                title = post.get_attribute("post-title")
                url = post.get_attribute("permalink")
                print(f"- {title} ({url})")
        except Exception as e:
            print("Error finding posts:", e)
            print("Page title:", page.title())
            
        browser.close()

if __name__ == "__main__":
    test_html_scrape()
