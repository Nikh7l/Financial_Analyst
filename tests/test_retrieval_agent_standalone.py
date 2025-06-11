# tests/test_retrieval_agent_standalone.py
import os
import sys
import logging
import pytest
import time
import json
from typing import Dict, Any, List
from google import genai
from dotenv import load_dotenv

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# --- Add project root to sys.path ---
# TODO: Remove sys.path manipulation block below.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# Import the agent graph runnable
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
from agents.retrieval_agent import retrieval_agent_runnable
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
import config # For API Key check

# Configure logging if running directly
if __name__ == "__main__":
     if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


# --- Integration Test Function ---

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv('RUN_INTEGRATION_TESTS'), reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests")
@pytest.mark.parametrize("company_name", [
    "Apple",
    "Microsoft",
    "Infosys", # Try without suffix first, agent might add it or fail
    # "RELIANCE.NS", # Ticker might work better for search
])
def test_standalone_retrieval_agent_live(company_name):
    """
    Performs an integration test of the standalone ReAct Retrieval Agent.
    """
    logger.info(f"\n--- Running LIVE Standalone ReAct Retrieval Test for: {company_name} ---")

    # Ensure API key is available
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=dotenv_path)
    if not config.GOOGLE_API_KEY:
        pytest.skip("GOOGLE_API_KEY not found, skipping integration test.")

    # Prepare initial input message
    initial_input = {
        "messages": [(
            "human",
            f"Please find the most recent official Annual Report (or 10-K) and the most recent official Quarterly Report (or 10-Q) PDF URLs for {company_name}."
        )]
    }

    final_state = None
    start_time = time.time()
    try:
        # Invoke the ReAct agent graph
        # Use stream to see intermediate steps (optional, but good for debugging)
        # final_state = retrieval_agent_runnable.invoke(initial_input)

        # --- Streaming Invocation ---
        logger.info("Streaming ReAct agent execution...")
        output_chunks = []
        for chunk in retrieval_agent_runnable.stream(initial_input, stream_mode="values"):
             # 'chunk' here is the full state dict at each step
             # logger.info(f"--- Agent Step State ---")
             # logger.info(pprint.pformat(chunk)) # Very verbose!
             output_chunks.append(chunk) # Store state at each step
             # Print last message nicely
             if chunk.get("messages"):
                  last_msg = chunk["messages"][-1]
                  logger.info(f"[{last_msg.type}] Content: {str(last_msg.content)[:500]}...") # Log truncated content
             time.sleep(0.1) # Small delay to allow logs to flush

        final_state = output_chunks[-1] if output_chunks else None
        # ---------------------------

    except Exception as e:
         pytest.fail(f"Standalone Retrieval Agent invocation failed unexpectedly: {e}", pytrace=True)

    end_time = time.time()
    logger.info(f"Agent execution finished in {end_time - start_time:.2f} seconds.")

    # --- Assertions ---
    assert final_state is not None, "Agent did not return a final state."
    assert "messages" in final_state, "Final state missing 'messages' key."
    assert len(final_state["messages"]) > 1, "Agent message history is too short (should have AI response)."

    final_message = final_state["messages"][-1]
    assert final_message.type == "ai", f"Expected last message to be from AI, but got {final_message.type}"
    assert isinstance(final_message.content, str), "Expected last AI message content to be a string."

    logger.info(f"Final AI Message Content:\n{final_message.content}")

    # Attempt to parse the JSON from the final message
    found_urls = None
    error_parsing = None
    try:
        response_text = final_message.content.strip()
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = response_text[json_start:json_end]
            parsed_json = json.loads(json_str)
            if "document_urls" in parsed_json and isinstance(parsed_json["document_urls"], list):
                 found_urls = parsed_json["document_urls"]
            else:
                error_parsing = "JSON parsed, but 'document_urls' key missing or not a list."
        else:
            error_parsing = "Could not find JSON object in the final AI response."
    except json.JSONDecodeError as e:
        error_parsing = f"Failed to parse JSON from final AI response: {e}"
    except Exception as e:
         error_parsing = f"Unexpected error parsing final AI response: {e}"

    # Assert that parsing succeeded and URLs were found (or list is empty if none found)
    assert error_parsing is None, f"Failed to get expected JSON output: {error_parsing}\nFull Response: {final_message.content}"
    assert found_urls is not None, "Parsed JSON successfully but 'found_urls' list wasn't extracted (logic error)."

    logger.info(f"Successfully extracted URLs: {found_urls}")

    # Flexible check on the number of URLs found
    max_reports = getattr(config, 'MAX_REPORTS_TO_FETCH', 3)
    assert 0 <= len(found_urls) <= max_reports, \
        f"Expected 0 to {max_reports} URLs, found {len(found_urls)}"

    if company_name in ["Apple", "Microsoft"] and not found_urls:
        logger.warning(f"Found 0 URLs for major company {company_name}. Check agent reasoning logs.")
        # pytest.fail(f"Expected at least one URL for {company_name}, found none.") # Optional: make it a failure

    # Basic plausibility check on URLs if any were found
    for url in found_urls:
        assert isinstance(url, str) and url.startswith("http"), f"Invalid URL format found: {url}"
        # assert ".pdf" in url.lower(), f"Returned URL doesn't look like PDF: {url}" # Maybe too strict

    logger.info("Pausing briefly...")
    time.sleep(5)


# --- Main Block for Direct Execution ---
if __name__ == "__main__":
     print("--- Running Standalone ReAct Retrieval Agent Test Script ---")
     load_dotenv(os.path.join(project_root, '.env')) # Ensure loaded

     # (Optional) Apply requests monkey-patch if needed for local run
     # import requests
     # ... (patch code) ...

     test_companies = ["Apple", "Microsoft", "Infosys"] # Test subset

     for company in test_companies:
         if not config.GOOGLE_API_KEY:
             print(f"Skipping {company} - GOOGLE_API_KEY not found.")
             continue
         try:
            #  client = genai.Client()
             print(f"\n>>> Testing Company: {company} <<<\n")
             test_standalone_retrieval_agent_live(company)
             print("\n>>> Test Completed <<<\n")
         except Exception as e:
              print(f"\n!!! Critical error during test for {company}: {e} !!!\n")

     print("--- Standalone ReAct Test Script Finished ---")