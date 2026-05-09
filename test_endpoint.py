import asyncio
import traceback
from app import analyze_sentiment, AnalyzeRequest

async def main():
    try:
        req = AnalyzeRequest(query="test", mode="deep")
        res = analyze_sentiment(req)
        print("Success:", res)
    except Exception as e:
        print("Failed!")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
