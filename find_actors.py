import urllib.request
import re

url = "https://apify.com/store?search=reddit"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req).read().decode('utf-8')
    actors = set(re.findall(r'href="/([^/]+/[^/]*reddit[^/]*)"', html))
    print("Found actors:", actors)
except Exception as e:
    print("Error:", e)
