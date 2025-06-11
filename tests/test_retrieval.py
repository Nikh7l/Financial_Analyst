# tests/test_retrieval_agent.py
import os
import sys
import logging
import pytest
from unittest.mock import MagicMock, patch
from datetime import date
from typing import List, Dict, Any

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
from agents.retrieval_agent import DocumentRetrievalAgent
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
import tools # Import module to allow patching its functions
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
from tools import SearchResponse, URLValidityResult, PageSummary # Import response models

# Import Gemini library (needed for client setup)
from google import genai

logger = logging.getLogger(__name__)

# --- Fixtures ---

agent_tools_list = [
    tools.search_duck_duck_go,
    tools.check_pdf_url_validity
]

@pytest.fixture(scope="module")
def configured_client():
    """Fixture to set up the Gemini client once per test module."""
    logger.info("--- Setting up Gemini Client for RetrievalAgent tests ---")
    # Load .env from project root
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found, skipping tests needing Gemini client.")
    try:
        client = genai.Client()
        logger.info("Gemini Client created for retrieval tests.")
        return client
    except Exception as e:
        pytest.fail(f"Failed to create Gemini Client for retrieval tests: {e}")

# --- Test Helper Data & Mocks ---

# Sample Search Results (PageSummary format for DDG)
MOCK_SEARCH_RESULTS_GOOD = [
    PageSummary(page_title="Apple Inc. 2023 Annual Report", page_summary="Official 2023 10-K filing...", page_url="http://investor.apple.com/reports/2023/10k_report.pdf"),
    PageSummary(page_title="AAPL Q3 2024 Earnings Report PDF", page_summary="Apple Q3 2024 results...", page_url="http://investor.apple.com/reports/2024/q3_earnings.pdf"),
    PageSummary(page_title="Apple Q2 2024 10-Q Filing", page_summary="Quarterly report for Q2...", page_url="http://sec.gov/apple/q2_2024_10q.pdf"),
    PageSummary(page_title="Some Random Apple News (Not PDF)", page_summary="Apple announces new phone...", page_url="http://news.com/apple_news.html"),
    PageSummary(page_title="Old Report 2022", page_summary="2022 filing...", page_url="http://archive.com/apple_2022_10k.pdf"),
    PageSummary(page_title="Investor Presentation Q3 2024", page_summary="Slides from conf call...", page_url="http://investor.apple.com/slides/q3_2024_slides.pdf"), # Less ideal than report
]

MOCK_SEARCH_RESULTS_EMPTY = []
MOCK_SEARCH_RESULTS_NON_RELEVANT = [
     PageSummary(page_title="Apple Recipes", page_summary="How to make pie...", page_url="http://food.com/apple_pie.html"),
     PageSummary(page_title="History of Apple Computers", page_summary="Wozniak and Jobs...", page_url="http://history.com/apple.pdf"), # PDF but not financial
]

# --- Test Functions ---

@pytest.mark.parametrize("mock_today, expected_annual_yr, expected_q, expected_q_yr", [
    (date(2024, 10, 25), 2023, 3, 2024), # Oct -> Q3
    (date(2024, 8, 15), 2023, 2, 2024),  # Aug -> Q2
    (date(2024, 5, 5), 2023, 1, 2024),   # May -> Q1
    (date(2024, 2, 20), 2023, 4, 2023),  # Feb -> Prev Q4
    (date(2025, 1, 10), 2024, 4, 2024),  # Jan -> Prev Q4
])
def test_get_target_periods(mocker, configured_client, mock_today, expected_annual_yr, expected_q, expected_q_yr):
    """Tests the internal _get_target_periods logic."""
    # TODO: Update mocker.patch path to reflect 'src.' structure (e.g., 'src.core.tools.X').
    mocker.patch('agents.retrieval_agent.date').today.return_value = mock_today
    # Instantiate with dummy tools dict for this internal test
    agent = DocumentRetrievalAgent(client=configured_client, model_name="test", tools=agent_tools_list)
    annual_yr, latest_q, latest_q_year = agent._get_target_periods()
    assert annual_yr == expected_annual_yr
    assert latest_q == expected_q
    assert latest_q_year == expected_q_yr

# Basic test for query generation structure
def test_generate_search_queries(configured_client):
    """Tests the structure of generated search queries."""
    # Instantiate with dummy tools dict
    agent = DocumentRetrievalAgent(client=configured_client, model_name="test", tools=agent_tools_list)
    queries = agent._generate_search_queries("TestCorp", 2023, 3, 2024)
    assert isinstance(queries, list)
    assert len(queries) > 4
    assert '"TestCorp" 2023 annual report filetype:pdf' in queries
    assert '"TestCorp" Q3 2024 earnings report filetype:pdf' in queries
    # Update this assertion to match actual generated query
    assert '"TestCorp" investor relations reports 2023' in queries # Check for year-specific one
    assert '"TestCorp" SEC filings' in queries # Keep this general check

# --- Test the main 'run' method with mocked tools ---

# Define mock side effect functions for tools
def mock_search_side_effect(query: str, **kwargs) -> SearchResponse:
    logger.debug(f"Mock Search Tool called with query: '{query}'")
    # Simulate different responses based on query content for variety
    if "investor relations reports" in query:
        return SearchResponse(success=True, page_summaries=MOCK_SEARCH_RESULTS_GOOD[::2]) # Return subset
    elif "filing" in query:
        return SearchResponse(success=True, page_summaries=MOCK_SEARCH_RESULTS_GOOD[1::2]) # Return other subset
    elif "no_results_company" in query:
         return SearchResponse(success=True, page_summaries=[])
    elif "search_error_company" in query:
         return SearchResponse(success=False, error="Simulated search API error")
    else: # Default good results
        return SearchResponse(success=True, page_summaries=MOCK_SEARCH_RESULTS_GOOD)

def mock_url_validation_side_effect(url: str, **kwargs) -> URLValidityResult:
    logger.debug(f"Mock URL Validator called for: '{url}'")
    if "investor.apple.com" in url or "sec.gov" in url: # Assume official domains are valid PDFs
        if "slides.pdf" in url: # Treat presentation as less ideal but valid PDF
            return URLValidityResult(success=True, is_valid=True)
        elif url.endswith('.pdf'):
            return URLValidityResult(success=True, is_valid=True)
        else: # Simulate finding non-PDF on official site
             return URLValidityResult(success=True, is_valid=False, error="Content-Type mismatch")
    elif "archive.com" in url: # Simulate valid PDF on other domain
         return URLValidityResult(success=True, is_valid=True)
    elif "news.com" in url or not url.endswith('.pdf'): # Assume non-PDFs are invalid
        return URLValidityResult(success=False, is_valid=False, error="Not a PDF")
    else: # Default invalid
        return URLValidityResult(success=False, is_valid=False, error="Validation check failed")


@pytest.mark.parametrize("company_name, mock_today, expected_urls_contain, expected_num_urls, expect_error", [
    # Test cases remain the same
    ("Apple", date(2024, 10, 25), ["2023/10k_report.pdf", "2024/q3_earnings.pdf"], 3, False),
    ("Apple", date(2024, 2, 20), ["2023/10k_report.pdf", "q2_2024_10q.pdf"], 3, False),
    ("no_results_company", date(2024, 10, 25), [], 0, False),
    ("NonRelevant Inc.", date(2024, 10, 25), [], 0, False),
    ("search_error_company", date(2024, 10, 25), [], 0, True),
    ("ValidSearchInvalidPDF Inc.", date(2024, 10, 25), [], 0, False),
    (None, date(2024, 10, 25), [], 0, True),
])
def test_run_retrieval_agent(mocker, configured_client, company_name, mock_today, expected_urls_contain, expected_num_urls, expect_error):
    """Tests the full run method of DocumentRetrievalAgent with mocked date and tools."""
    logger.info(f"\n--- Testing run for Company: '{company_name}', Date: {mock_today} ---")

    # Mock date (patching the source where date is used)
    mocker.patch('agents.retrieval_agent.date').today.return_value = mock_today

    # Instantiate the agent FIRST using the *real* tool functions initially
    # (This is necessary because __init__ uses tool_mapping based on real function names)
    agent_tools_list = [
        tools.search_duck_duck_go,
        tools.check_pdf_url_validity
    ]
    try:
        agent = DocumentRetrievalAgent(client=configured_client, model_name="test", tools=agent_tools_list)
    except ValueError as e:
        # Handle case where tool might genuinely be missing in tools module if code changed
        pytest.fail(f"Agent instantiation failed, potentially missing tool: {e}")


    # --- NOW, patch the *instance attributes* holding the tool methods ---
    # TODO: Update mocker.patch path to reflect 'src.' structure (e.g., 'src.core.tools.X').
    mock_search = mocker.patch.object(agent, 'search_tool', side_effect=mock_search_side_effect)
    # TODO: Update mocker.patch path to reflect 'src.' structure (e.g., 'src.core.tools.X').
    mock_validate = mocker.patch.object(agent, 'url_validator', side_effect=mock_url_validation_side_effect)
    # ---------------------------------------------------------------------

    # Adjust mock search side_effect dynamically based on company_name for specific tests
    # (This needs to re-assign the side_effect on the *mock object* created by patch.object)
    if company_name == "no_results_company":
         mock_search.side_effect = lambda query, **kwargs: SearchResponse(success=True, page_summaries=[]) # Ensure specific mock applies
    elif company_name == "NonRelevant Inc.":
         mock_search.side_effect = lambda query, **kwargs: SearchResponse(success=True, page_summaries=MOCK_SEARCH_RESULTS_NON_RELEVANT)
    elif company_name == "search_error_company":
         mock_search.side_effect = lambda query, **kwargs: SearchResponse(success=False, error="Simulated search API error")
    elif company_name == "ValidSearchInvalidPDF Inc.":
         # Return good search results, but make validator fail them
         mock_search.side_effect = lambda query, **kwargs: SearchResponse(success=True, page_summaries=MOCK_SEARCH_RESULTS_GOOD)
         mock_validate.side_effect = lambda url, **kwargs: URLValidityResult(success=False, is_valid=False, error="Mocked validation failure")
    # else: # It will use the default mock_search_side_effect defined outside the test function


    # Prepare initial state (same as before)
    initial_state = AgentState(
        query=f"Analyze {company_name}" if company_name else "No company query",
        company_name=company_name,
        query_type='company' if company_name else None, sector_name=None, workflow_plan=None,
        document_urls=None, document_summaries=None, sentiment_summary=None,
        stock_prediction=None, stock_data=None, competitors=None,
        sector_analysis_summary=None, candidate_companies=None, selected_companies=None,
        companies_to_analyze=[], current_company_in_loop=None, final_output=None, error_message=None
    )

    # Run the agent
    result_updates = agent.run(initial_state)
    logger.info(f"Agent run resulted in updates: {result_updates}")

    # Assertions (remain the same, but should now pass)
    if expect_error:
        # Case 1: Error expected because company_name was None
        if company_name is None:
             assert "error_message" in result_updates
             assert result_updates.get("error_message") == "Company name not found in state."
             # Tools should NOT have been called
             assert mock_search.call_count == 0
             assert mock_validate.call_count == 0
        # Case 2: Error expected because all searches failed
        elif company_name == "search_error_company":
             assert "error_message" in result_updates
             assert result_updates.get("error_message") == "All search attempts failed."
             # Search tool WAS called multiple times
             assert mock_search.call_count > 0 # Check it was called
             # Validate tool might not be called if no results pass initial filter
             assert mock_validate.call_count >= 0
        else:
             # Fallback for other potential future error cases
             assert "error_message" in result_updates
             assert result_updates.get("error_message") is not None

        # Ensure document_urls is not present or empty in error cases
        assert "document_urls" not in result_updates or not result_updates.get("document_urls")

    else: # If no error was expected
        assert "error_message" not in result_updates or result_updates.get("error_message") is None
        assert "document_urls" in result_updates
        retrieved_urls = result_updates["document_urls"]
        assert isinstance(retrieved_urls, list)
        assert len(retrieved_urls) == expected_num_urls
        if expected_num_urls > 0:
            for expected_part in expected_urls_contain:
                assert any(expected_part in url for url in retrieved_urls), f"Expected URL part '{expected_part}' not found in {retrieved_urls}"

        # Verify mocks were called appropriately for non-error cases
        if company_name: # Only check calls if agent run logic was expected to execute fully
            assert mock_search.call_count > 0
            assert mock_validate.call_count >= 0