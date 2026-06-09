import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

def get_attraction(city: str, weather: str) -> str:
    """
    Based on city and weather, use Tavily Search API to search and return optimized attraction recommendations.
    """
    # 1. Read API key from environment variable
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable not configured."

    # 2. Initialize Tavily client
    tavily = TavilyClient(api_key=api_key)

    # 3. Construct a precise query
    query = f"'{city}' most worthwhile tourist attractions and reasons in '{weather}' weather"

    try:
        # 4. Call API, include_answer=True will return a comprehensive answer
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        # 5. Tavily's returned results are already very clean and can be used directly
        # response['answer'] is a summary answer based on all search results
        if response.get("answer"):
            return response["answer"]

        # If there's no comprehensive answer, format raw results
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
             return "Sorry, no relevant tourist attraction recommendations found."

        return "Based on search, found the following information for you:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"Error: Problem occurred when executing Tavily search - {e}"