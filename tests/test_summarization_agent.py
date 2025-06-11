# tests/test_summarization_agent.py
import os
import sys
import logging
import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# --- Add project root to sys.path ---
# TODO: Remove sys.path manipulation block below.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# Import necessary components
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
import config
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
from state import AgentState
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
from agents.summarization_agent import DocumentSummarizationAgent
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
import tools # Need the real tool function for agent init
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
from tools import FinancialSummary # Import the response model

# Import Gemini library
from google import genai

logger = logging.getLogger(__name__)

# --- Fixtures ---
@pytest.fixture(scope="module")
def configured_client():
    """Fixture to set up the Gemini client once per test module."""
    logger.info("--- Setting up Gemini Client for SummarizationAgent tests ---")
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found, skipping tests needing Gemini client.")
    try:
        client = genai.Client()
        logger.info("Gemini Client created for summarization tests.")
        return client
    except Exception as e:
        pytest.fail(f"Failed to create Gemini Client: {e}")

# --- Mock Tool Side Effect ---
def mock_summarize_side_effect(doc_url: str, client: genai.Client, **kwargs) -> FinancialSummary:
    """Simulates the summarization tool's behavior."""
    logger.debug(f"Mock Summarize Tool called for URL: {doc_url}")
    if "good-url.pdf" in doc_url:
        return FinancialSummary(success=True, summary=f"Summary for {doc_url}")
    elif "tool-error-url.pdf" in doc_url:
        return FinancialSummary(success=False, error="Simulated tool processing error")
    elif "network-error-url.pdf" in doc_url:
         # Simulate an exception the decorator might catch, though the tool aims to catch these
         raise ConnectionError("Simulated network failure during summarization tool call")
    elif "empty-summary-url.pdf" in doc_url:
         # Simulate success but no summary text returned by LLM
         return FinancialSummary(success=True, summary=None)
    else:
        # Default failure case for unexpected URLs
        return FinancialSummary(success=False, error="Unknown URL for mock summarizer")

# --- Test Cases ---

@pytest.mark.parametrize("input_urls, expected_summaries_keys, expected_success_count, expected_error_count, expect_agent_error", [
    # Happy path: two good URLs
    (["http://good-url.pdf", "http://another-good-url.pdf"], ["http://good-url.pdf", "http://another-good-url.pdf"], 2, 0, False),
    # Empty input list
    ([], [], 0, 0, False),
    # List with one good, one tool error
    (["http://good-url.pdf", "http://tool-error-url.pdf"], ["http://good-url.pdf", "http://tool-error-url.pdf"], 1, 1, False),
    # List with one good, one network error (exception)
    (["http://network-error-url.pdf", "http://good-url.pdf"], ["http://network-error-url.pdf", "http://good-url.pdf"], 1, 1, False),
    # List with tool error only
    (["http://tool-error-url.pdf"], ["http://tool-error-url.pdf"], 0, 1, False),
    # List with good url but empty summary result
    (["http://empty-summary-url.pdf"], ["http://empty-summary-url.pdf"], 0, 1, False),
    # Non-list input for document_urls
    ("not_a_list", [], 0, 0, True),
    # None input for document_urls
    (None, [], 0, 0, False), # Agent should handle None gracefully
])
def test_summarization_agent_run_logic(mocker, configured_client, input_urls, expected_summaries_keys, expected_success_count, expected_error_count, expect_agent_error):
    """Tests the DocumentSummarizationAgent's run method logic with mocked tool."""
    logger.info(f"\n--- Testing Summarization Agent run with input URLs: {input_urls} ---")

    # Instantiate agent with the real tool function initially
    agent_tools_list = [tools.summarize_pdf_document_finance]
    try:
        agent = DocumentSummarizationAgent(client=configured_client, model_name="test", tools=agent_tools_list)
    except ValueError as e:
         pytest.fail(f"Agent instantiation failed: {e}")

    # Patch the 'summarization_tool' attribute on the instance
    mock_summarizer = mocker.patch.object(agent, 'summarization_tool', side_effect=mock_summarize_side_effect)

    # Prepare initial state
    initial_state = AgentState(
        query="Summarize docs", company_name="TestCo", query_type='company',
        document_urls=input_urls, # Use parametrized input
        # Initialize other fields
        sector_name=None, workflow_plan=None, document_summaries=None, sentiment_summary=None,
        stock_prediction=None, stock_data=None, competitors=None, sector_analysis_summary=None,
        candidate_companies=None, selected_companies=None, companies_to_analyze=[],
        current_company_in_loop=None, final_output=None, error_message=None
    )

    # Run the agent
    result_updates = agent.run(initial_state)
    logger.info(f"Agent run resulted in updates: {result_updates}")

    # Assertions
    if expect_agent_error:
         # Agent itself should report an error (e.g., bad input type)
         assert "error_message" in result_updates and result_updates["error_message"] is not None
         assert "document_summaries" not in result_updates or not result_updates.get("document_summaries") # Should not produce summaries dict on input error
         assert mock_summarizer.call_count == 0 # Tool should not have been called
    else:
        # Agent run should succeed, even if individual summaries failed
        assert "error_message" not in result_updates or result_updates["error_message"] is None
        assert "document_summaries" in result_updates
        summaries_dict = result_updates["document_summaries"]
        assert isinstance(summaries_dict, dict)
        assert len(summaries_dict) == len(expected_summaries_keys) # Check if all inputs produced an output (summary or error)

        # Verify call count matches expected number of URLs (if input was valid list)
        expected_calls = len(input_urls) if isinstance(input_urls, list) else 0
        assert mock_summarizer.call_count == expected_calls

        # Check content of summaries dict
        actual_success = 0
        actual_error = 0
        for url, result_text in summaries_dict.items():
             assert url in expected_summaries_keys # Ensure keys match input URLs
             if result_text.startswith("Error:"):
                 actual_error += 1
                 # Check specific error messages if needed based on mock_summarize_side_effect
                 if "tool-error-url" in url: assert "Simulated tool processing error" in result_text
                 elif "network-error-url" in url: assert "Simulated network failure" in result_text or "Unexpected exception" in result_text
                 elif "empty-summary-url" in url: assert "Unexpected response from summarization tool" in result_text or "empty summary" in result_text # Adjust based on actual error message
             else:
                 actual_success += 1
                 assert f"Summary for {url}" == result_text # Check success message format

        assert actual_success == expected_success_count
        assert actual_error == expected_error_count