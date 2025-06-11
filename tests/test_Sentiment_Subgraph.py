# tests/test_sentiment_agent_standalone.py
import os
import sys
import logging
import pytest
import time
import json
import pprint
from typing import Dict, Any, List
from datetime import date
from dotenv import load_dotenv

# --- Add project root to sys.path ---
# TODO: Remove sys.path manipulation block below.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# Import the agent graph runnable
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
from agents.sentiment_agent_subgraph import sentiment_agent_runnable
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
import config # For API Key check

from config import logger


# --- Integration Test Function ---

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv('RUN_INTEGRATION_TESTS'), reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests")
@pytest.mark.parametrize("company_name", [
    "Apple",
    # "Microsoft",
    # "Nvidia", # Example volatile stock
    # Add more diverse companies
])
def test_standalone_sentiment_agent_live(company_name):
    """
    Performs an integration test of the standalone ReAct Sentiment Agent.
    """
    logger.info(f"\n--- Running LIVE Standalone ReAct Sentiment Test for: {company_name} ---")

    # Ensure API key is available
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=dotenv_path)
    if not config.GOOGLE_API_KEY:
        pytest.skip("GOOGLE_API_KEY not found, skipping integration test.")
    if not config.NEWS_API_KEY:
         pytest.skip("NEWS_API_KEY not found, skipping integration test.")
    current_date_str = date.today().isoformat()
    days_ago = config.NEWS_API_DEFAULT_DAYS_AGO # Get default days from config
    max_articles_review = 10


    SENTIMENT_WORKER_PROMPT = """
    You are a specialized financial market sentiment analyst. Your goal is to determine the current market sentiment (Positive, Negative, or Neutral) towards **{company_name}** and its stock, based on recent information, and provide a concise summary explaining the sentiment.

    **Your Process:**
    1.  **Fetch Recent News:** Use the `get_news_articles` tool to retrieve relevant news articles published within the last {days_ago} days (default 7, today is {current_date}) for "{company_name}".
    2.  **Initial Assessment:** Review the titles and descriptions of the top {max_articles_review} news articles (e.g., top 10). Identify the main themes (e.g., earnings, product, legal, macro). Is there a clear overall sentiment?
    3.  **Identify Key Articles for Deeper Dive :** Based on your initial assessment, are there specific articles whose headlines or descriptions seem particularly significant (e.g., major earnings surprise, significant lawsuit update, major analyst action) but the provided snippet is too short to fully understand the impact or tone? If yes, identify their URLs.
    4.  **Fetch Full Content ):** If you identified key articles in the previous step, use the `get_page_content` tool for those specific URLs.
    5.  **Supplement with Search (If News is Sparse/Inconclusive):** If the initial news search yielded very few results OR the sentiment remains very unclear after reviewing news , use the `google_search` tool or `search_duck_duck_go` tool with a focused query like "{company_name} stock sentiment recent analyst rating" to look for strong sentiment indicators missed by the news API.
    6.  **Synthesize Findings:** Based on *all* gathered information (initial news snippets, full content from any investigated articles, supplemental search results), determine the overall sentiment (Positive, Negative, or Neutral). Justify your conclusion by referencing the key findings.
    7.  **Final Answer:** Generate your final answer ONLY as a JSON object containing two keys:
        *   `sentiment`: Your final assessment - one of "Positive", "Negative", or "Neutral".
        *   `sentiment_summary`: A detailed which captures the details report explaining the reasoning, citing key news or events discovered. If no relevant info was found, state sentiment as "Neutral" and explain the lack of data.

    **Constraint:** Today's date is {current_date}. Focus on information relevant to market sentiment from the last {days_ago} days. Base your analysis *only* on the information gathered through the tools. Be efficient - only use `get_page_content` if truly necessary for high-impact articles where the snippet is insufficient.

    **Available Tools:**
    - `get_news_articles`: Fetches structured recent news articles for a query. Use this first.
    - `google_search`: Performs a web search.
    - `search_duck_duck_go`: Performs a web search. Use sparingly if news is insufficient.
    - `get_page_content`: Fetches text content from a specific URL. 
    Begin! Analyze sentiment for {company_name}.
    """

    # Prepare initial input message including the company name
    initial_input = {
        "messages": [(
            "human",
            f"{SENTIMENT_WORKER_PROMPT.format(
                company_name=company_name,
                current_date=current_date_str,
                days_ago=days_ago,
                max_articles_review=max_articles_review
            )}"
            # "human",
            # f"Please analyze the current market sentiment for {company_name}."
        )]
    }

    final_state = None
    start_time = time.time()
    try:
        # Invoke the ReAct agent graph
        logger.info("Streaming ReAct agent execution...")
        output_chunks = []
        # Increase recursion limit slightly if needed for complex sentiment gathering
        stream_config = {"recursion_limit": 20}
        for chunk in sentiment_agent_runnable.stream(initial_input, config=stream_config, stream_mode="values"):
             output_chunks.append(chunk)
             if chunk.get("messages"):
                  last_msg = chunk["messages"][-1]
                  logger.info(f"[{last_msg.type}] Content: {str(last_msg.content)[:500]}...")
             time.sleep(0.1)

        final_state = output_chunks[-1] if output_chunks else None

    except Exception as e:
         pytest.fail(f"Standalone Sentiment Agent invocation failed unexpectedly: {e}", pytrace=True)

    end_time = time.time()
    logger.info(f"Agent execution finished in {end_time - start_time:.2f} seconds.")

    # --- Assertions ---
    assert final_state is not None, "Agent did not return a final state."
    assert "messages" in final_state, "Final state missing 'messages' key."
    final_message = final_state["messages"][-1]
    assert final_message.type == "ai", f"Expected last message to be from AI, but got {final_message.type}"
    assert isinstance(final_message.content, str), "Expected last AI message content to be a string."

    logger.info(f"Final AI Message Content:\n{final_message.content}")

    # Attempt to parse the JSON from the final message
    parsed_json = None
    error_parsing = None
    try:
        response_text = final_message.content.strip()
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match: json_str = match.group(1)
        else:
            json_start = response_text.find('{'); json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != -1: json_str = response_text[json_start:json_end]
            else: raise ValueError("JSON object not found")
        parsed_json = json.loads(json_str)
    except Exception as e:
        error_parsing = f"Failed to parse JSON from final AI response: {e}"

    # Assert JSON parsing and structure
    assert error_parsing is None, f"{error_parsing}\nFull Response: {final_message.content}"
    assert parsed_json is not None, "JSON parsing yielded None."
    assert "sentiment" in parsed_json, "Final JSON missing 'sentiment' key."
    assert "sentiment_summary" in parsed_json, "Final JSON missing 'sentiment_summary' key."
    assert parsed_json["sentiment"] in ["Positive", "Negative", "Neutral"], \
        f"Sentiment value '{parsed_json['sentiment']}' is not one of Positive/Negative/Neutral."
    assert isinstance(parsed_json["sentiment_summary"], str) and len(parsed_json["sentiment_summary"]) > 10, \
        "Sentiment summary is missing, not a string, or too short."

    logger.info(f"Successfully parsed sentiment: {parsed_json['sentiment']}")
    print("\n" + "="*30)
    print(f"FINAL PARSED RESULT for {company_name}:")
    pprint.pprint(parsed_json, indent=2)
    print("="*30 + "\n")

    logger.info("Pausing briefly...")
    time.sleep(5) # Pause between companies


# --- Main Block for Direct Execution ---
if __name__ == "__main__":
     import pprint # Make sure pprint is imported
     import re # Make sure re is imported

     print("--- Running Standalone ReAct Sentiment Agent Test Script ---")
     load_dotenv(os.path.join(project_root, '.env'))

     test_companies = ["Apple"]#, "Microsoft", "Nvidia", "Infosys"]

     # Configure genai client globally for this run
     if not config.GOOGLE_API_KEY: print("Missing GOOGLE_API_KEY"); exit()
     if not config.NEWS_API_KEY: print("Missing NEWS_API_KEY"); exit() # Check News key too
    #  try: genai.configure(api_key=config.GOOGLE_API_KEY)
    #  except Exception as e: print(f"Client config failed: {e}"); exit()

     for company in test_companies:
         try:
             print(f"\n>>> Testing Sentiment for: {company} <<<\n")
             test_standalone_sentiment_agent_live(company)
             print("\n>>> Test Completed <<<\n")
         except Exception as e:
              # Catch failures from pytest.fail or other issues
              print(f"\n!!! Test run failed for {company}: {e} !!!\n")

     print("--- Standalone ReAct Sentiment Test Script Finished ---")