from tavily import TavilyClient
import os
from dotenv import load_dotenv


load_dotenv()


client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def tavily_search(query):
    response = client.search(query,max_results=3)

    results = []

    for i,res in enumerate(response["results"], 1):
        title = res.get("title", "No title")
        url = res.get("url", "")
        snippet = res.get("content","").strip()

        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ",1)[0] + "..."

        results.append(f"{i}. **{title}**\n {url}\n {snippet}")

    return "\n\n".join(results)