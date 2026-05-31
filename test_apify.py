from apify_client import ApifyClient
from config import APIFY_API_TOKEN

def main():
    if not APIFY_API_TOKEN:
        print("No API token.")
        return
        
    client = ApifyClient(APIFY_API_TOKEN)
    actors = [
        "apify/reddit-scraper",
        "m1n0/reddit-scraper",
        "zuzka/reddit-scraper",
        "curious_coder/reddit-scraper",
        "radubot/reddit-scraper",
        "trudax/reddit-scraper",
        "trudax/reddit-scraper-lite"
    ]
    
    for actor_id in actors:
        print(f"\n--- Testing {actor_id} ---")
        try:
            run_input = {"startUrls": [{"url": "https://www.reddit.com/r/python/top/?t=day"}]}
            run = client.actor(actor_id).call(run_input=run_input)
            print(f"Success! Run ID: {run.get('id') if isinstance(run, dict) else getattr(run, 'id', 'Unknown')}")
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == "__main__":
    main()
