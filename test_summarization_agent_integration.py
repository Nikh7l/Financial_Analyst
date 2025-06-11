# # tests/test_summarization_agent_integration.py
# import os
# import sys
# import logging
# import pytest
# import time

# from dotenv import load_dotenv

# # --- Add project root to sys.path ---
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
# # ------------------------------------

# # Import necessary components
# import config
# from state import AgentState
# from agents.summarization_agent import DocumentSummarizationAgent
# import tools # Need the real tool functions

# # Import Gemini library
# from google import genai

# # Import potential exceptions from tools
# import requests.exceptions
# import httpx
# from duckduckgo_search.exceptions import DuckDuckGoSearchException # If search was used

# logger = logging.getLogger(__name__)

# # --- Fixtures ---
# @pytest.fixture(scope="module")
# def configured_client():
#     """Fixture to set up the Gemini client once per test module."""
#     logger.info("--- Setting up Gemini Client for Summarization Integration tests ---")
#     dotenv_path = os.path.join(project_root, '.env')
#     load_dotenv(dotenv_path=dotenv_path)
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         pytest.skip("GOOGLE_API_KEY not found, skipping integration tests.")
#     try:
#         client = genai.Client()
#         logger.info("Gemini Client created for summarization integration tests.")
#         return client
#     except Exception as e:
#         pytest.fail(f"Failed to create Gemini Client: {e}")

# # --- Known PDF URLs for Testing ---
# # Use PDFs known to be accessible and likely parsable. Replace with stable URLs if needed.
# # Example: Google's 2023 10-K (Check if URL is still valid)
# GOOD_PDF_URL_GOOGLE_10K = "https://www.abc.xyz/assets/70/a3/43ba8a804b49ac2fa2595c3c6704/2024-annual-report.pdf" # Check/Replace with actual URL
# # Example: A known smaller, simple PDF
# SIMPLE_PDF_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
# # Example: A URL likely to fail validation or download
# BAD_URL_NON_PDF = "https://google.com"
# BAD_URL_404 = "https://example.com/non_existent_document.pdf"


# # --- Integration Test Function ---
# @pytest.mark.integration
# @pytest.mark.skipif(not os.getenv('RUN_INTEGRATION_TESTS'), reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests")
# @pytest.mark.parametrize("input_urls, expected_success_keys, expected_error_keys", [
#     # Test with one good financial report URL
#     pytest.param([GOOD_PDF_URL_GOOGLE_10K], [GOOD_PDF_URL_GOOGLE_10K], [], id="good_10k"),
#     # Test with one simple valid PDF
#     pytest.param([SIMPLE_PDF_URL], [SIMPLE_PDF_URL], [], id="simple_pdf"),
#     # Test with a mix of good and bad URLs
#     pytest.param([GOOD_PDF_URL_GOOGLE_10K, BAD_URL_NON_PDF, BAD_URL_404, SIMPLE_PDF_URL],
#                  [GOOD_PDF_URL_GOOGLE_10K, SIMPLE_PDF_URL], [BAD_URL_NON_PDF, BAD_URL_404], id="mixed_urls"),
#     # Test with only bad URLs
#     pytest.param([BAD_URL_NON_PDF, BAD_URL_404], [], [BAD_URL_NON_PDF, BAD_URL_404], id="only_bad_urls"),
#     # Test with empty list
#     pytest.param([], [], [], id="empty_list"),
# ])
# def test_summarization_agent_live_run(configured_client, input_urls, expected_success_keys, expected_error_keys):
#     """
#     Performs an integration test of DocumentSummarizationAgent using live network calls
#     and the real summarization tool.
#     """
#     logger.info(f"\n--- Running LIVE Summarization Test for URLs: {input_urls} ---")

#     # Instantiate agent with the REAL summarization tool
#     agent_tools_list = [tools.summarize_pdf_document_finance]
#     try:
#         agent = DocumentSummarizationAgent(client=configured_client, model_name=config.GEMINI_MODEL_NAME, tools=agent_tools_list)
#     except ValueError as e:
#          pytest.fail(f"Agent instantiation failed: {e}")

#     # Prepare initial state
#     initial_state = AgentState(
#         query="Summarize docs", company_name="IntegrationTest", query_type='company',
#         document_urls=input_urls, # Use parametrized input URLs
#         # Initialize other fields
#         sector_name=None, workflow_plan=None, document_summaries=None, sentiment_summary=None,
#         stock_prediction=None, stock_data=None, competitors=None, sector_analysis_summary=None,
#         candidate_companies=None, selected_companies=None, companies_to_analyze=[],
#         current_company_in_loop=None, final_output=None, error_message=None
#     )

#     result_updates = None
#     try:
#         # Run the agent - makes REAL network and LLM calls
#         result_updates = agent.run(initial_state)
#         logger.info(f"Agent run returned updates: {result_updates}")

#     except Exception as e:
#          # Catch any unexpected errors during the agent's run method itself
#          pytest.fail(f"Agent run failed unexpectedly: {e}", pytrace=True)

#     # --- Assertions ---
#     assert result_updates is not None, "Agent run did not return a result."
#     assert "error_message" not in result_updates or result_updates.get("error_message") is None, \
#         f"Agent run reported a top-level error: {result_updates.get('error_message')}"
#     assert "document_summaries" in result_updates, "Result missing 'document_summaries' key."

#     summaries_dict = result_updates["document_summaries"]
#     assert isinstance(summaries_dict, dict), "'document_summaries' is not a dictionary."
#     assert len(summaries_dict) == len(input_urls), \
#         f"Expected {len(input_urls)} entries in summaries dict, found {len(summaries_dict)}"

#     # Check specific results
#     for url in expected_success_keys:
#         assert url in summaries_dict, f"Expected successful summary for {url} not found in results."
#         summary_text = summaries_dict[url]
#         assert isinstance(summary_text, str), f"Summary for {url} is not a string."
#         assert not summary_text.startswith("Error:"), f"Expected success for {url}, but got error: {summary_text}"
#         assert len(summary_text) > 10, f"Summary for {url} seems too short: {summary_text}" # Basic sanity check
#         logger.info(f"  OK: Summary found for {url} (length: {len(summary_text)})")

#     for url in expected_error_keys:
#          assert url in summaries_dict, f"Expected error message for {url} not found in results."
#          error_text = summaries_dict[url]
#          assert isinstance(error_text, str), f"Error message for {url} is not a string."
#          assert error_text.startswith("Error:"), f"Expected error for {url}, but got: {error_text}"
#          logger.info(f"  OK: Error message found for {url}: {error_text}")

#     logger.info("Pausing briefly before next integration test case...")
#     time.sleep(5) # Longer pause as this involves LLM calls


# tests/test_summarization_node_integration.py
import os
import sys
import logging
import pytest
import time
import pprint
from typing import Dict, Any, List

from dotenv import load_dotenv

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# Import necessary components
import config
from state import AgentState
# Import the node function directly
from agents.summarization_agent import summarize_documents_node
# Import the tool function's response model if needed for type hints
from tools import FinancialSummary

# Import Gemini library
from google import genai

# Configure logging if running directly
if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


# --- Helper to Initialize Client ---
# Separate function to ensure client is configured for each test run if needed
def get_configured_client():
    logger.info("--- Setting up Gemini Client for Summarization Integration test ---")
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found. Cannot run test.")
        return None
    try:
        # Ensure genai is configured before creating client if necessary
        # genai.configure(api_key=api_key)
        client = genai.Client()
        logger.info("Gemini Client created for summarization integration test.")
        return client
    except Exception as e:
        logger.error(f"Failed to create/configure Gemini Client: {e}")
        return None

# --- Real PDF URLs (Replace with current, valid URLs) ---
# These MUST be direct links to publicly accessible PDFs
# Use URLs found by the retrieval agent previously if they were good.
# Example structure:
URLS_TO_TEST = {
    "Apple": [
        "https://app.stocklight.com/stocks/us/nasdaq-aapl/apple/annual-reports/nasdaq-aapl-2024-10K-241416806.pdf", # 2024 10k (FY End Sep'24)
        "https://www.apple.com/newsroom/pdfs/fy2024-q4/FY24_Q4_Consolidated_Financial_Statements.pdf?os=jva" # Q4 FY24 Fin Statements (Maybe smaller?)
        # Add a known invalid/non-pdf url
        # "https://apple.com/investor"
    ],
    "Microsoft": [
        # Use the GCS link if it resolves to a PDF - CHECK THIS MANUALLY FIRST
        "https://microsoft.gcs-web.com/static-files/1c864583-06f7-40cc-a94d-d11400c83cc8", # 2024 10K? Check manually
        # Add a known quarterly report link if the above is annual
        "https://microsoft.gcs-web.com/static-files/19c77d4d-97c9-442a-abb4-a5c30b592507" # Q1 FY24 10Q? Check manually
    ],
    "Infosys": [
        "https://www.infosys.com/investors/reports-filings/quarterly-results/2024-2025/q4/documents/q4-and-12m-fy25-financial-results-auditorsreports.pdf", 
        "https://www.infosys.com/investors/reports-filings/quarterly-results/2024-2025/q4/documents/standalone/sa-fy25-annual-finstatement.pdf", 
        "https://www.infosys.com/investors/reports-filings/quarterly-results/2024-2025/q4/documents/standalone/sa-fy25-annual-auditorsreport.pdf"
         ],
    "ErrorCase": [
        "https://example.com/non_existent_document.pdf", # 404
        "https://google.com" # Not a PDF
    ]
}


# --- Main Integration Test Execution (Direct Run) ---
if __name__ == "__main__":
    print("--- Running Summarization Node Integration Test Script ---")

    client = get_configured_client()
    if not client:
        print("!!! Aborting test: Could not configure Gemini client.")
        sys.exit(1)

    all_results = {}

    for company, urls in URLS_TO_TEST.items():
        print(f"\n>>> Testing Summarization for: {company} <<<")
        print(f"Input URLs: {urls}")

        # Prepare initial state for this company
        test_state = AgentState(
            query=f"Summarize {company} reports",
            company_name=company,
            query_type='company',
            document_urls=urls,
            # Fill other keys with None or defaults
            sector_name=None, workflow_plan=None, document_summaries=None,
            sentiment_summary=None, stock_prediction=None, stock_data=None,
            competitors=None, sector_analysis_summary=None, candidate_companies=None,
            selected_companies=None, company_analysis_results=None, final_output=None,
            error_message=None, messages=[]
        )

        node_output = None
        start_time = time.time()
        try:
            # Call the node function directly, passing the client
            node_output = summarize_documents_node(state=test_state, client=client)

        except Exception as e:
            logger.error(f"Summarization node call failed unexpectedly for {company}: {e}", exc_info=True)
            node_output = {"document_summaries": {}, "error_message": f"Node call failed: {e}"} # Simulate error dict structure

        end_time = time.time()
        logger.info(f"Node execution finished in {end_time - start_time:.2f} seconds.")

        # --- Print Results ---
        print("\n" + "="*30)
        print(f"RESULTS FOR: {company}")
        print("="*30)
        print("Node Output Dictionary:")
        pprint.pprint(node_output, indent=2)

        if node_output and "document_summaries" in node_output:
            print("\nIndividual Summaries/Errors:")
            for url, summary_or_error in node_output["document_summaries"].items():
                print(f"\n--- URL: {url} ---")
                print(summary_or_error)
                print("-"*(len(url) + 12))
        print("="*30 + "\n")

        all_results[company] = node_output
        # Optional delay
        # print("Pausing...")
        # time.sleep(10) # Pause if hitting rate limits

    print("\n--- Integration Test Script Finished ---")
    # Optional: print summary of all results again
    print("Overall results:")
    pprint.pprint(all_results, indent=2)