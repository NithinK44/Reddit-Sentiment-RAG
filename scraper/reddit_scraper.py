"""
Reddit Scraper — Extracted from rough.ipynb
--------------------------------------------
Scrapes subreddit posts and comments via Reddit's public JSON API.
Supports optional Cloudflare WARP proxy for IP rotation.
Designed to be called from the FastAPI backend with progress callbacks.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import json
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from config import DATA_DIR


# ============================================================================
# WARP PROXY (OPTIONAL)
# ============================================================================

WARP_CLI_PATH = r"C:\Program Files\Cloudflare\Cloudflare WARP\warp-cli.exe"


def find_warp_port() -> Optional[int]:
    """Scans for the Cloudflare WARP Proxy port."""
    potential_ports = [40000, 1080, 8080, 9091]
    for port in potential_ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result == 0:
                return port
        except Exception:
            pass
    return None


def get_proxy_config() -> dict:
    """
    Detect if WARP proxy is available. Returns proxy dict or empty dict.
    """
    port = find_warp_port()
    if port:
        proxy_url = f"socks5://127.0.0.1:{port}"
        # Verify the proxy actually works
        try:
            resp = requests.get(
                "https://api.ipify.org?format=json",
                proxies={'http': proxy_url, 'https': proxy_url},
                timeout=5,
            )
            if resp.status_code == 200:
                return {
                    'http': proxy_url,
                    'https': proxy_url,
                    '_ip': resp.json().get('ip', 'unknown'),
                    '_port': port,
                }
        except Exception:
            pass
    return {}


# ============================================================================
# COMMENT TREE PROCESSING
# ============================================================================

def process_comment_tree(
    comment_data: dict,
    thread_author: str,
    depth_limits: dict,
    current_depth: int = 0,
) -> list:
    """
    Recursively builds a comment tree.
    Injects [OP] and [MOD] tags for authority.
    Captures 'score' for quality filtering.
    """
    limit = depth_limits.get(current_depth, 0)
    if limit == 0:
        return []

    processed_comments = []
    children = comment_data.get('children', [])[:limit + 2]

    count = 0
    for child in children:
        if count >= limit:
            break

        data = child.get('data', {})
        if child.get('kind') == 'more':
            continue

        body = data.get('body')
        author = data.get('author')
        distinguished = data.get('distinguished')
        score = data.get('score', 0)

        if body and body not in ["[deleted]", "[removed]"]:
            # Authority injection
            if distinguished == 'moderator':
                body = f"🛡️ [MODERATOR]: {body}"
            elif author == thread_author:
                body = f"🔴 [OP/CREATOR]: {body}"

            comment_obj = {
                "body": body,
                "score": score,
                "replies": [],
            }

            replies_raw = data.get('replies')
            if isinstance(replies_raw, dict):
                reply_tree = replies_raw.get('data', {})
                comment_obj['replies'] = process_comment_tree(
                    reply_tree, thread_author, depth_limits, current_depth + 1
                )

            processed_comments.append(comment_obj)
            count += 1

    return processed_comments


# ============================================================================
# MAIN SCRAPER
# ============================================================================

# Global state for tracking scrape progress
_scrape_state = {
    "running": False,
    "total": 0,
    "completed": 0,
    "current_post": "",
    "error": None,
    "finished": False,
    "proxy_ip": None,
    "posts_saved": [],
}


def get_scrape_status() -> dict:
    """Return the current scrape job status."""
    return dict(_scrape_state)


def scrape_subreddit(
    subreddit: str,
    post_limit: int = 25,
    sort_by: str = "top",
    time_filter: str = "year",
    depth_limits: Optional[dict] = None,
) -> dict:
    """
    Scrape a subreddit and save posts as JSON files.

    Args:
        subreddit: Name of the subreddit (without r/)
        post_limit: Number of posts to scrape (max 200)
        sort_by: Sort method — 'top', 'hot', 'new'
        time_filter: Time filter for 'top' — 'day', 'week', 'month', 'year', 'all'
        depth_limits: Comment depth limits dict, e.g. {0: 25, 1: 15, 2: 10}

    Returns:
        Dict with scrape results summary
    """
    global _scrape_state

    if _scrape_state["running"]:
        return {"error": "A scrape job is already running"}

    if depth_limits is None:
        depth_limits = {0: 25, 1: 15, 2: 10}

    # Clamp post_limit
    post_limit = max(1, min(200, post_limit))

    # Reset state
    _scrape_state = {
        "running": True,
        "total": post_limit,
        "completed": 0,
        "current_post": "",
        "error": None,
        "finished": False,
        "proxy_ip": None,
        "posts_saved": [],
    }

    try:
        # Ensure output directory exists
        output_dir = DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup session
        session = requests.Session()

        # Try WARP proxy
        proxy_config = get_proxy_config()
        if proxy_config:
            session.proxies.update({
                'http': proxy_config['http'],
                'https': proxy_config['https'],
            })
            _scrape_state["proxy_ip"] = proxy_config.get('_ip')

        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36'
        }

        posts_collected = 0
        after = None

        while posts_collected < post_limit:
            # Build listing URL
            if sort_by == 'top':
                list_url = f"https://www.reddit.com/r/{subreddit}/top.json?t={time_filter}&limit=100"
            else:
                list_url = f"https://www.reddit.com/r/{subreddit}/{sort_by}.json?limit=100"

            if after:
                list_url += f"&after={after}"

            try:
                res = session.get(list_url, headers=headers, timeout=15).json()
                if 'data' not in res:
                    break
                posts = res['data']['children']
                after = res['data']['after']
            except Exception as e:
                _scrape_state["error"] = f"Listing error: {e}"
                break

            if not posts:
                break

            for post in posts:
                if posts_collected >= post_limit:
                    break

                p_data = post['data']
                thread_id = p_data['id']
                thread_author = p_data.get('author')

                safe_title = re.sub(r'[<>:"/\\|?*]', '', p_data['title'])[:50].strip()
                filename = f"{thread_id}_{safe_title}.json"
                file_path = output_dir / filename

                _scrape_state["current_post"] = safe_title

                if file_path.exists():
                    posts_collected += 1
                    _scrape_state["completed"] = posts_collected
                    _scrape_state["posts_saved"].append(filename)
                    continue

                thread_url = (
                    f"https://www.reddit.com/r/{subreddit}/comments/{thread_id}/.json?sort=top"
                )

                try:
                    thread_res = session.get(thread_url, headers=headers, timeout=15).json()
                    if not isinstance(thread_res, list) or len(thread_res) < 2:
                        continue

                    sort_display = sort_by.upper()
                    if sort_by == 'top':
                        sort_display += f" OF {time_filter.upper()}"

                    structured_comments = process_comment_tree(
                        thread_res[1]['data'],
                        thread_author=thread_author,
                        depth_limits=depth_limits,
                        current_depth=0,
                    )

                    doc_object = {
                        "meta": {
                            "title": p_data['title'],
                            "url": f"https://reddit.com{p_data['permalink']}",
                            "score": p_data.get('score', 0),
                            "flair": p_data.get('link_flair_text'),
                            "date": datetime.fromtimestamp(
                                p_data.get('created_utc', 0), timezone.utc
                            ).strftime('%Y-%m-%d'),
                            "sort": sort_display,
                        },
                        "content": {
                            "post_body": p_data.get('selftext', ''),
                            "comments": structured_comments,
                        },
                    }

                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(doc_object, f, indent=4, ensure_ascii=False)

                    posts_collected += 1
                    _scrape_state["completed"] = posts_collected
                    _scrape_state["posts_saved"].append(filename)

                    # Rate limit: 2 seconds between requests
                    time.sleep(2)

                except Exception as e:
                    _scrape_state["error"] = f"Error on thread {thread_id}: {e}"

            if not after:
                break

        _scrape_state["finished"] = True
        _scrape_state["running"] = False
        return {
            "success": True,
            "posts_scraped": posts_collected,
            "output_dir": str(output_dir),
        }

    except Exception as e:
        _scrape_state["error"] = str(e)
        _scrape_state["running"] = False
        _scrape_state["finished"] = True
        return {"error": str(e)}
