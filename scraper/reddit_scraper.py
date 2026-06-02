"""
Reddit Scraper — Playwright Implementation
--------------------------------------------
Scrapes subreddit posts and comments via Playwright headless browser.
Designed to bypass 403 blocks on Reddit's API endpoints.
"""

import time
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from playwright.sync_api import sync_playwright

from config import DATA_DIR, REDDIT_SESSION_COOKIE, SCRAPER_MAX_WORKERS

# ============================================================================
# MAIN SCRAPER STATE
# ============================================================================

_scrape_state = {
    "running": False,
    "total": 0,
    "completed": 0,
    "current_post": "",
    "error": None,
    "finished": False,
    "proxy_ip": None,
    "posts_saved": [],
    "warnings": [],
}
_scrape_lock = threading.Lock()

def _set_scrape_state(**kwargs):
    with _scrape_lock:
        _scrape_state.update(kwargs)

def _increment_completed_posts(filename: str):
    with _scrape_lock:
        _scrape_state["completed"] += 1
        # Avoid sharing list references to avoid race conditions or modifications
        _scrape_state["posts_saved"] = list(_scrape_state["posts_saved"]) + [filename]

def _add_warning(warning_msg: str):
    with _scrape_lock:
        # Append is atomic in Python, but using lock for consistency
        _scrape_state["warnings"].append(warning_msg)

def get_scrape_status() -> dict:
    with _scrape_lock:
        status = dict(_scrape_state)
    
    if status["total"] > 0:
        status["progress"] = (status["completed"] / status["total"]) * 100
    else:
        status["progress"] = 0
        
    if status["running"]:
        status["message"] = f"Scraping r/{_scrape_state.get('subreddit', 'Reddit')}... ({status['completed']}/{status['total']})"
    elif status["error"]:
        status["message"] = f"Error: {status['error']}"
    elif status["finished"]:
        if status.get("warnings"):
            status["message"] = f"Finished! Scraped {status['completed']} posts (with {len(status['warnings'])} warnings)."
        else:
            status["message"] = f"Finished! Scraped {status['completed']} posts."
    else:
        status["message"] = "Ready to scrape."
        
    return status

# ============================================================================
# PLAYWRIGHT SCRAPING LOGIC
# ============================================================================

def setup_browser(p):
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--dns-prefetch-disable",
            "--disable-features=VizDisplayCompositor",
            "--disable-extensions",
            "--mute-audio",
            # Hide automation flags from Reddit's bot detection
            "--disable-blink-features=AutomationControlled",
        ]
    )
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        viewport={"width": 1280, "height": 800},
        locale="en-US",
        timezone_id="America/New_York",
    )
    # Hide navigator.webdriver property which Reddit checks
    context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    # Intercept and block heavy resources (images, media, fonts, stylesheets) and tracking domains to optimize loading speed
    def handle_route(route):
        url = route.request.url.lower()
        if route.request.resource_type in ["image", "media", "font", "stylesheet"]:
            return route.abort()
        
        blacklisted_keywords = [
            "google-analytics", "googletagmanager", "googleadservices", "doubleclick",
            "amazon-adsystem", "adnxs", "ads-twitter", "facebook.net", "facebook.com/tr",
            "scorecardresearch", "quantserve", "hotjar", "optimizely", "crashlytics",
            "sentry.io", "mixpanel", "amplitude", "branch.io"
        ]
        if any(kw in url for kw in blacklisted_keywords):
            return route.abort()
        
        return route.continue_()

    context.route("**/*", handle_route)

    if REDDIT_SESSION_COOKIE:
        context.add_cookies([{
            "name": "reddit_session",
            "value": REDDIT_SESSION_COOKIE,
            "domain": ".reddit.com",
            "path": "/"
        }])
    return browser, context


def _goto_with_challenge_wait(page, url, timeout=30000):
    """Navigate and wait for Reddit's JS challenge to auto-resolve."""
    for attempt in range(3):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            # If Reddit served a JS challenge, it will redirect; wait for it
            # Check if we're still on a challenge page
            for _ in range(10):
                current_url = page.url
                if "js_challenge" in current_url or "challenge" in current_url:
                    time.sleep(1.5)
                else:
                    break
            return True
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(3)
    return False

def validate_subreddit(subreddit: str) -> dict:
    url = f"https://www.reddit.com/r/{subreddit}/about/"
    import concurrent.futures
    
    def _do_validate():
        try:
            with sync_playwright() as p:
                browser, context = setup_browser(p)
                page = context.new_page()
                
                _goto_with_challenge_wait(page, url)
                
                # Wait for actual Reddit content — shreddit-subreddit-header or r/ title
                try:
                    page.wait_for_selector("shreddit-subreddit-header, [data-testid='subreddit-title'], h1", timeout=10000)
                except:
                    pass
                
                display_name = ""
                description = ""
                subscribers = 0
                
                header_loc = page.locator("shreddit-subreddit-header")
                header_found = header_loc.count() > 0
                
                if header_found:
                    header = header_loc.first
                    display_name = header.get_attribute("display-name") or ""
                    description = header.get_attribute("description") or ""
                    
                    sub_count_str = (
                        header.get_attribute("weekly-active-users") or 
                        header.get_attribute("subscribers") or 
                        header.get_attribute("sub-count") or 
                        header.get_attribute("members") or 
                        ""
                    )
                    if sub_count_str:
                        try:
                            cleaned_sub = re.sub(r'[^\d]', '', sub_count_str)
                            if cleaned_sub:
                                subscribers = int(cleaned_sub)
                        except:
                            pass
                
                if not display_name:
                    h1_loc = page.locator("h1")
                    if h1_loc.count() > 0:
                        display_name = h1_loc.first.inner_text()
                    else:
                        display_name = subreddit
                
                if not description:
                    try:
                        meta_desc = page.locator("meta[name='description']").first
                        if meta_desc.count() > 0:
                            description = meta_desc.get_attribute("content") or ""
                    except:
                        pass
                
                page_title = page.title()
                browser.close()
                
                if "404" in page_title or "page not found" in page_title.lower() or "reddit - dive into anything" in page_title.lower() or not header_found:
                    if "private" in page_title.lower() or "banned" in page_title.lower():
                        return {"success": False, "error": f"Subreddit 'r/{subreddit}' is private or banned."}
                    return {"success": False, "error": f"Subreddit 'r/{subreddit}' does not exist."}
                     
                return {
                    "success": True,
                    "metadata": {
                        "name": subreddit,
                        "title": display_name or page_title,
                        "subscribers": subscribers,
                        "description": description,
                    }
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_do_validate).result()

def scrape_post_worker(post_tasks, subreddit, sort_by, depth_limits, output_dir):
    """
    Scrapes a chunk of Reddit posts sequentially using a single Playwright instance.
    post_tasks: List of tuples (idx, p_data)
    """
    if not post_tasks:
        return 0

    scraped_count = 0
    with sync_playwright() as p:
        try:
            browser, context = setup_browser(p)
        except Exception as e:
            warning_msg = f"Failed to initialize browser context for thread: {e}"
            _add_warning(warning_msg)
            return 0

        for idx, p_data in post_tasks:
            thread_url = f"https://www.reddit.com{p_data['url']}"
            safe_title = re.sub(r'[\x00-\x1f<>:"/\\|?*]', '', p_data['title'])[:50].strip()
            filename = f"{idx}_{safe_title}.json"
            file_path = output_dir / filename

            _set_scrape_state(current_post=safe_title)

            if file_path.exists():
                scraped_count += 1
                _increment_completed_posts(filename)
                continue

            thread_page = context.new_page()
            try:
                # Navigate to thread and wait for challenge + comments to load
                _goto_with_challenge_wait(thread_page, thread_url)
                try:
                    thread_page.wait_for_selector("shreddit-comment, shreddit-post", timeout=8000)
                except:
                    pass

                post_body = ""
                comment_count = 1  # Default to 1 to attempt comment scraping if attribute is missing
                try:
                    post_loc = thread_page.locator("shreddit-post")
                    if post_loc.count() > 0:
                        post_el = post_loc.first
                        body_loc = post_el.locator("div[slot='text-body']")
                        if body_loc.count() > 0:
                            post_body = body_loc.first.inner_text()
                        
                        cc_str = post_el.get_attribute("comment-count")
                        if cc_str is not None:
                            try:
                                comment_count = int(cc_str)
                            except:
                                comment_count = 1
                except:
                    pass

                comments = []
                # Only wait/scrape comments if the post actually contains comments
                if comment_count > 0:
                    try:
                        thread_page.wait_for_selector("shreddit-comment", timeout=5000)
                        comment_count_on_page = thread_page.locator("shreddit-comment").count()
                        limit = min(depth_limits[0], comment_count_on_page)
                        for i in range(limit):
                            try:
                                c_el = thread_page.locator("shreddit-comment").nth(i)
                                c_score = c_el.get_attribute("score") or "0"
                                c_text_loc = c_el.locator("div[slot='comment']")
                                c_text = c_text_loc.first.inner_text() if c_text_loc.count() > 0 else ""
                                if c_text:
                                    comments.append({
                                        "body": c_text,
                                        "score": c_score,
                                        "replies": []
                                    })
                            except:
                                pass
                    except:
                        pass

                doc_object = {
                    "meta": {
                        "title": p_data['title'],
                        "url": thread_url,
                        "score": p_data['score'],
                        "flair": "",
                        "date": str(datetime.now(timezone.utc).strftime('%Y-%m-%d')),
                        "sort": sort_by.upper(),
                    },
                    "content": {
                        "post_body": post_body,
                        "comments": comments,
                    },
                }

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(doc_object, f, indent=4, ensure_ascii=False)

                scraped_count += 1
                _increment_completed_posts(filename)
            except Exception as e:
                warning_msg = f"Error on thread {p_data['url']}: {e}"
                _add_warning(warning_msg)
            finally:
                try:
                    thread_page.close()
                except:
                    pass

        try:
            browser.close()
        except:
            pass

    return scraped_count


def scrape_subreddit(
    subreddit: str,
    post_limit: int = 25,
    sort_by: str = "top",
    time_filter: str = "year",
    depth_limits: Optional[dict] = None,
) -> dict:
    global _scrape_state

    if _scrape_state["running"]:
        return {"error": "A scrape job is already running"}

    if depth_limits is None:
        depth_limits = {0: 25, 1: 15, 2: 10}

    post_limit = max(1, min(250, post_limit))

    with _scrape_lock:
        _scrape_state.update({
            "running": True,
            "subreddit": subreddit,
            "total": post_limit,
            "completed": 0,
            "current_post": "",
            "error": None,
            "warnings": [],
            "finished": False,
            "proxy_ip": "local-playwright",
            "posts_saved": [],
        })

    try:
        output_dir = DATA_DIR / subreddit
        output_dir.mkdir(parents=True, exist_ok=True)

        if sort_by == 'top':
            start_url = f"https://www.reddit.com/r/{subreddit}/top/?t={time_filter}"
        else:
            start_url = f"https://www.reddit.com/r/{subreddit}/{sort_by}/"

        import concurrent.futures
        
        def _do_scrape():
            posts_data = []
            with sync_playwright() as p:
                browser, context = setup_browser(p)
                page = context.new_page()
                
                _set_scrape_state(message=f"Loading r/{subreddit}...")
                
                # Navigate and wait for JS challenge to resolve
                _goto_with_challenge_wait(page, start_url)
                
                # Wait for actual post elements
                try:
                    page.wait_for_selector("shreddit-post", timeout=15000)
                except:
                    pass
                
                # Scroll down to load posts
                for _ in range(5):
                    try:
                        page.wait_for_selector("shreddit-post", timeout=5000)
                    except:
                        break
                    
                    elements = page.locator("shreddit-post").all()
                    for el in elements:
                        try:
                            title = el.get_attribute("post-title")
                            url = el.get_attribute("permalink")
                            score = el.get_attribute("score") or "0"
                            if title and url and url not in [p_item['url'] for p_item in posts_data]:
                                posts_data.append({"title": title, "url": url, "score": score})
                        except:
                            pass
                    
                    if len(posts_data) >= post_limit:
                        break
                        
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    time.sleep(2)
                    
                posts_data = posts_data[:post_limit]
                page.close()
                browser.close()

            num_posts = len(posts_data)
            if num_posts == 0:
                return 0

            _set_scrape_state(total=num_posts)

            # Parallelize scraping of posts using thread workers
            num_workers = min(SCRAPER_MAX_WORKERS, num_posts)
            chunks = [[] for _ in range(num_workers)]
            for idx, p_data in enumerate(posts_data):
                chunks[idx % num_workers].append((idx, p_data))

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        scrape_post_worker,
                        chunk,
                        subreddit,
                        sort_by,
                        depth_limits,
                        output_dir
                    )
                    for chunk in chunks
                ]
                posts_collected = sum(f.result() for f in futures)
                
            return posts_collected

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            posts_collected = executor.submit(_do_scrape).result()

        _set_scrape_state(finished=True, running=False)
        return {
            "success": True,
            "posts_scraped": posts_collected,
            "output_dir": str(output_dir),
        }

    except Exception as e:
        _set_scrape_state(error=str(e), running=False, finished=True)
        return {"error": str(e)}
