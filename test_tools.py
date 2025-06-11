# tests/test_tools.py
import os
import sys
import logging
import pytest
from unittest.mock import patch, MagicMock
import requests # Import for exceptions
import httpx # Import for exceptions
from dotenv import load_dotenv
from datetime import timedelta
from datetime import date as real_date
import pandas as pd 
# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# Import necessary components from your project
import config
from state import AgentState
import tools # Import the module itself
# Import specific functions and models
from tools import (
    check_pdf_url_validity, get_web_page_content, search_duck_duck_go,
    google_search, summarize_pdf_document_finance,get_news_articles, #get_stock_data, # Assuming placeholder exists
    URLValidityResult, SearchResponse, GoogleSearchResponse, FullPageContent,
    FinancialSummary, StockData, PageSummary, GoogleSearchResult, BaseToolResponse,NewsApiResponse, NewsArticle, NewsSource
)

# Import Gemini library (needed for client setup)
from google import genai
from google.genai import types # Still needed for Tool creation in BaseAgent

# Get the logger configured in config.py
logger = logging.getLogger(__name__)

# --- Fixtures ---

@pytest.fixture(scope="module")
def configured_client():
    """Fixture to set up the Gemini client once per test module."""
    logger.info("--- Setting up Gemini Client for Tool tests ---")
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found, skipping tests needing Gemini client.")
    try:
        client = genai.Client()
        logger.info("Gemini Client created for tool tests.")
        return client
    except Exception as e:
        pytest.fail(f"Failed to create Gemini Client for tool tests: {e}")

# --- Mocks for External Services (Revised) ---

# Mock response for requests.head used in check_pdf_url_validity
def mock_requests_head(*args, **kwargs):
    url = args[0]
    mock_response = MagicMock(spec=requests.Response) # Use spec for better mocking
    mock_response.headers = {}
    mock_response.status_code = 500 # Default to error

    if "valid-pdf.com" in url:
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/pdf'}
        mock_response.raise_for_status = MagicMock()
    elif "valid-non-pdf.com" in url:
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.raise_for_status = MagicMock()
    elif "notfound.com" in url:
        mock_response.status_code = 404
        mock_response.raise_for_status = MagicMock(side_effect=requests.exceptions.HTTPError("404 Client Error: Not Found for url"))
    elif "timeout.com" in url:
         raise requests.exceptions.Timeout("Request timed out")
    elif "generic-error.com" in url:
        raise requests.exceptions.RequestException("Generic request error")
    elif "invalid-url-format" in url: # Simulate error for invalid URL format
        raise requests.exceptions.MissingSchema("Invalid URL: No schema supplied")
    else: # Default success (maybe shouldn't happen in strict tests)
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'text/plain'}
        mock_response.raise_for_status = MagicMock()

    return mock_response

# Mock response for requests.get used in get_web_page_content and google_search (API call)
def mock_requests_get(*args, **kwargs):
    url = args[0]
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 500

    # --- Mock for get_web_page_content ---
    if "example-content.com" in url or "example.com/1" in url or "example.com/2" in url: # Handle body fetch for google search
        mock_response.status_code = 200
        mock_response.content = b"<html><head><title>Test Page</title></head><body><p>Some example content.</p> <script>ignore me</script></body></html>"
        mock_response.text = mock_response.content.decode('utf-8')
        mock_response.raise_for_status = MagicMock()
    elif "content-timeout.com" in url:
        raise requests.exceptions.Timeout("Content request timed out")
    elif "content-notfound.com" in url:
        mock_response.status_code = 404
        mock_response.raise_for_status = MagicMock(side_effect=requests.exceptions.HTTPError("404 Client Error: Content Not Found"))
    elif "invalid-url" == url: # Match exact invalid url string
        raise requests.exceptions.MissingSchema("Invalid URL: No schema")
        

    # --- Mock for google_search API call ---
    elif "customsearch.googleapis.com" in url:
        query = kwargs.get('params', {}).get('q', '')
        if "good search" in query:
             mock_response.status_code = 200
             mock_response.json = MagicMock(return_value={
                 "items": [
                     {"title": "Result 1", "link": "http://example.com/1", "snippet": "Snippet 1..."},
                     {"title": "Result 2", "link": "http://example.com/2", "snippet": "Snippet 2..."},
                 ]
             })
             mock_response.raise_for_status = MagicMock()
        elif "search error" in query:
             mock_response.status_code = 503
             mock_response.text = "Service Unavailable"
             mock_response.raise_for_status = MagicMock(side_effect=requests.exceptions.HTTPError("503 Server Error: Service Unavailable"))
        elif "no results search" in query:
             mock_response.status_code = 200
             mock_response.json = MagicMock(return_value={"items": []})
             mock_response.raise_for_status = MagicMock()
        else: # Default success for API call
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value={"items": [{"title": "Default", "link": "http://default.com", "snippet": "Default..."}]})
            mock_response.raise_for_status = MagicMock()
    # Add a fallback for unexpected requests.get calls
    else:
        logger.warning(f"Unhandled mock requests.get for URL: {url}")
        mock_response.status_code = 418 # I'm a teapot
        mock_response.raise_for_status = MagicMock(side_effect=requests.exceptions.HTTPError("Mock not implemented for this URL"))


    return mock_response

# Mock for DDGS().text (Keep as before)
def mock_ddgs_text(*args, **kwargs):
    query = args[0]
    if "good ddg search" in query:
        return [{"title": "DDG Result 1", "body": "Body 1", "href": "http://ddg.example.com/1"},
                {"title": "DDG Result 2", "body": "Body 2", "href": "http://ddg.example.com/2"}]
    elif "ddg error" in query:
        raise Exception("Simulated DDG library error")
    else: return []

# Mock response for httpx.get used in summarize_pdf_document_finance
def mock_httpx_get(*args, **kwargs):
    url = args[0]
    # Use spec=httpx.Response for better type hinting during mock setup
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.headers = {}
    mock_response.status_code = 500
    mock_response.request = MagicMock(url=url) # Mock request needed by HTTPStatusError

    if "good-pdf.com" in url:
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/pdf;charset=utf-8'} # More specific
        mock_response.content = b'%PDF-1.0...' # Minimal PDF bytes
        mock_response.raise_for_status = MagicMock()
    elif "bad-content-type.com" in url:
         mock_response.status_code = 200
         mock_response.headers = {'content-type': 'text/html'}
         mock_response.content = b"<html></html>"
         mock_response.raise_for_status = MagicMock()
    elif "pdf-download-error.com" in url:
         mock_response.status_code = 503
         mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("Service Unavailable", request=mock_response.request, response=mock_response))
    elif "pdf-timeout.com" in url:
         raise httpx.TimeoutException("Timeout downloading PDF", request=mock_response.request)
    else:
        mock_response.status_code = 404
        mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("Not Found", request=mock_response.request, response=mock_response))

    return mock_response


# --- Test Functions (Revised Assertions and Structure) ---

@pytest.mark.parametrize("url, expected_valid, expected_success, expected_error_substr", [
    ("http://valid-pdf.com/doc.pdf", True, True, None),
    ("http://valid-non-pdf.com/page.html", False, False, "Content-Type is not application/pdf"), # Expect success=False now
    ("http://notfound.com/missing.pdf", False, False, "404 Client Error"),
    ("http://timeout.com/slow.pdf", False, False, "Timeout"),
    ("http://generic-error.com/file", False, False, "Generic request error"),
    ("invalid-url-format", False, False, "Invalid URL"),
])
@patch('tools.requests.head', side_effect=mock_requests_head)
def test_check_pdf_url_validity(mock_head, url, expected_valid, expected_success, expected_error_substr):
    """Tests check_pdf_url_validity with refined logic."""
    logger.info(f"Testing PDF validity for: {url}")
    result = tools.check_pdf_url_validity(url=url)

    assert isinstance(result, URLValidityResult)
    assert result.success is expected_success
    assert result.is_valid is expected_valid # is_valid can only be True if success is True

    if expected_success:
        assert result.error is None
    else:
        assert result.is_valid is False # Should always be False if success is False
        assert result.error is not None
        if expected_error_substr:
            assert expected_error_substr in result.error


@pytest.mark.parametrize("url, use_bs4, expected_success, expected_content_substr, expected_error_substr", [
    ("http://example-content.com", True, True, "Some example content", None),
    ("http://example-content.com", False, True, "Some example content", None),
    ("http://content-timeout.com", True, False, None, "Timeout"),
    ("http://content-notfound.com", True, False, None, "404 Client Error"),
    ("invalid-url", True, False, None, "Invalid URL"),
])
@patch('tools.requests.get', side_effect=mock_requests_get)
def test_get_web_page_content(mock_get, url, use_bs4, expected_success, expected_content_substr, expected_error_substr):
    """Tests get_web_page_content with revised exception handling."""
    logger.info(f"Testing get_web_page_content for: {url} (bs4: {use_bs4})")
    result = tools.get_web_page_content(url=url, use_bs4=use_bs4, max_chars=50) # Limit chars for test

    assert isinstance(result, FullPageContent)
    assert result.success is expected_success
    assert result.page_url == url # Should always include the url

    if expected_success:
        assert result.error is None
        assert result.content is not None
        if expected_content_substr:
            assert expected_content_substr in result.content
            # Check that script tag was removed by bs4
            if use_bs4:
                 assert "<script>" not in result.content
    else:
        assert result.content is None
        assert result.error is not None
        if expected_error_substr:
             assert expected_error_substr in result.error


@pytest.mark.parametrize("query, expected_success, expected_num_results, expected_error_substr", [
    ("good ddg search", True, 2, None),
    ("empty ddg search", True, 0, None),
    ("ddg error", False, 0, "Simulated DDG library error"),
])
@patch('tools.DDGS')
def test_search_duck_duck_go(mock_ddgs_constructor, query, expected_success, expected_num_results, expected_error_substr):
    """Tests search_duck_duck_go (logic unchanged, relies on decorator)."""
    logger.info(f"Testing DDG search for: {query}")
    mock_ddgs_constructor.return_value.text.side_effect = mock_ddgs_text
    result = tools.search_duck_duck_go(query=query)

    assert isinstance(result, SearchResponse)
    assert result.success is expected_success
    if expected_success:
        assert result.error is None
        assert len(result.page_summaries) == expected_num_results
        if expected_num_results > 0: assert isinstance(result.page_summaries[0], PageSummary)
    else:
        assert result.error is not None
        if expected_error_substr: assert expected_error_substr in result.error


@pytest.mark.parametrize("query, fetch_body, expected_success, expected_num_results, expected_error_substr", [
    ("good search", False, True, 2, None),
    ("no results search", False, True, 0, None),
    ("search error", False, False, 0, "503 Server Error"), # Matching mock exception
    ("good search", True, True, 2, None), # Fetch body case fixed
])
@patch('tools.requests.get', side_effect=mock_requests_get) # Mocks both API call and potential body fetch
def test_google_search(mock_get, query, fetch_body, expected_success, expected_num_results, expected_error_substr):
    """Tests google_search tool with revised mocks and logic."""
    if not config.GOOGLE_SEARCH_API_KEY or not config.GOOGLE_CSE_ID:
        pytest.skip("Google API Key/CSE ID not configured, skipping google_search test.")

    logger.info(f"Testing Google search for: {query} (fetch_body: {fetch_body})")
    result = tools.google_search(query=query, fetch_body=fetch_body)

    assert isinstance(result, GoogleSearchResponse)
    assert result.success is expected_success

    if expected_success:
        assert result.error is None
        assert len(result.results) == expected_num_results
        if expected_num_results > 0:
             assert isinstance(result.results[0], GoogleSearchResult)
             if fetch_body:
                  # Body fetch should now succeed because get_web_page_content handles its own errors
                  assert result.results[0].body is not None
                  assert "example content" in result.results[0].body
             else:
                  assert result.results[0].body is None # Ensure body is None if not fetched
    else:
        assert result.error is not None
        if expected_error_substr:
             assert expected_error_substr in result.error


# Test for PDF Summarization (Revised Mocking)
@pytest.mark.parametrize("url, mock_llm_response_text, mock_llm_exception, expected_success, expected_summary_substr, expected_error_substr", [
    ("http://good-pdf.com/report.pdf", "Summary from LLM.", None, True, "Summary from LLM", None),
    ("http://bad-content-type.com/doc", None, None, False, None, "Downloaded content is not PDF"),
    ("http://pdf-download-error.com/doc.pdf", None, None, False, None, "HTTP error downloading PDF"),
    ("http://pdf-timeout.com/doc.pdf", None, None, False, None, "Timeout after"),
    ("http://good-pdf.com/report.pdf", None, ValueError("LLM API Format Error"), False, None, "LLM API Format Error"), # Simulate LLM error
])
@patch('tools.httpx.get', side_effect=mock_httpx_get)
def test_summarize_pdf_document_finance(mock_get, mocker, configured_client, url, mock_llm_response_text, mock_llm_exception, expected_success, expected_summary_substr, expected_error_substr):
    """Tests summarize_pdf_document_finance with revised mocks."""
    logger.info(f"Testing PDF summary for: {url}")

    # Mock the generate_content call on the client's model attribute
    mock_generate_content = mocker.patch.object(configured_client.models, 'generate_content')

    if mock_llm_exception:
        mock_generate_content.side_effect = mock_llm_exception
    else:
        # Create a simple mock object with a .text attribute
        mock_llm_response = MagicMock()
        mock_llm_response.text = mock_llm_response_text
        # Add other attributes if the tool code uses them (e.g., candidates)
        # mock_llm_response.candidates = ...
        mock_generate_content.return_value = mock_llm_response

    # Run the tool
    result = tools.summarize_pdf_document_finance(doc_url=url, client=configured_client)

    assert isinstance(result, FinancialSummary)
    assert result.success is expected_success

    # Assert based on expected outcome
    if expected_success:
        assert result.error is None
        assert result.summary is not None
        if expected_summary_substr:
            assert expected_summary_substr in result.summary
        # Verify LLM call was made only in success case (and not bad content type)
        if "good-pdf.com" in url:
             mock_generate_content.assert_called_once()
    else:
        assert result.summary is None
        assert result.error is not None
        if expected_error_substr:
             assert expected_error_substr in result.error
        # Verify LLM call was NOT made if download failed or content type was wrong
        if "bad-content-type.com" in url or "pdf-download-error.com" in url or "pdf-timeout.com" in url:
             mock_generate_content.assert_not_called()


# --- Test for Placeholder Tool ---
# def test_get_stock_data_placeholder():
#     """Tests the placeholder get_stock_data tool."""
#     logger.info("Testing placeholder get_stock_data tool")
#     symbol = "AAPL"
#     # Ensure the placeholder function actually exists in tools.py
#     assert hasattr(tools, 'get_stock_data'), "Placeholder function get_stock_data not found in tools.py"
#     result = tools.get_stock_data(symbol=symbol)

#     assert isinstance(result, StockData)
#     assert result.success is False
#     assert result.symbol == symbol
#     assert result.data is None
#     assert "Tool not implemented" in result.error # Check substring


# --- NEW Mock for requests.get used by get_news_articles ---
def mock_requests_get_news(*args, **kwargs):
    """Mocks requests.get specifically for the NewsAPI calls."""
    url = args[0]
    params = kwargs.get("params", {})
    query = params.get("q", "")
    headers = kwargs.get("headers", {})
    api_key = headers.get("X-Api-Key")

    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 500 # Default error
    mock_response.request = MagicMock(url=url, headers=headers) # Mock request for errors

    logger.debug(f"Mock NewsAPI GET: URL={url}, Params={params}, Headers={headers}")

    # Simulate missing/invalid API key before checking query
    if not api_key or api_key == "invalid_key":
         mock_response.status_code = 200 # API itself returns 200 on auth errors
         mock_response.json = MagicMock(return_value={
             "status": "error",
             "code": "apiKeyInvalid",
             "message": "Your API key is invalid or missing.",
         })
         mock_response.raise_for_status = MagicMock()
         return mock_response

    # Simulate based on query
    if "good-news-query" in query:
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "status": "ok",
            "totalResults": 2,
            "articles": [
                {
                    "source": {"id": "bbc-news", "name": "BBC News"},
                    "author": "BBC Correspondent",
                    "title": "Good News Title 1",
                    "description": "Description for good news 1.",
                    "url": "https://www.bbc.com/news/good1",
                    "urlToImage": "https://ichef.bbci.co.uk/images/ic/1024x576/p09x5q9w.jpg",
                    "publishedAt": "2024-05-01T10:00:00Z",
                    "content": "Content snippet 1..."
                },
                {
                    "source": {"id": None, "name": "Some Blog"},
                    "author": "Blogger",
                    "title": "Good News Title 2",
                    "description": "Description for good news 2.",
                    "url": "https://someblog.com/good2",
                    "urlToImage": None,
                    "publishedAt": "2024-05-01T09:00:00Z",
                    "content": "Content snippet 2..."
                }
            ]
        })
        mock_response.raise_for_status = MagicMock()
    elif "no-news-query" in query:
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "status": "ok",
            "totalResults": 0,
            "articles": []
        })
        mock_response.raise_for_status = MagicMock()
    elif "rate-limit-query" in query:
         mock_response.status_code = 200 # API returns 200
         mock_response.json = MagicMock(return_value={
             "status": "error",
             "code": "rateLimited",
             "message": "You have been rate limited.",
         })
         mock_response.raise_for_status = MagicMock()
    elif "api-error-query" in query: # Generic API error status
          mock_response.status_code = 200
          mock_response.json = MagicMock(return_value={
             "status": "error",
             "code": "sourcesTooMany",
             "message": "You have requested too many sources.",
         })
          mock_response.raise_for_status = MagicMock()
    elif "network-error-query" in query: # Simulate 5xx network error
        mock_response.status_code = 503
        mock_response.reason = "Service Unavailable"
        mock_response.text = "Service Unavailable" # <--- ADD THIS LINE
        mock_response.raise_for_status = MagicMock(side_effect=requests.exceptions.HTTPError(response=mock_response))
        mock_response.raise_for_status = MagicMock(side_effect=requests.exceptions.HTTPError(response=mock_response))
    elif "timeout-query" in query:
        raise requests.exceptions.Timeout("Request timed out connecting to NewsAPI")
    else: # Default to no results found for unexpected queries
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"status": "ok", "totalResults": 0, "articles": []})
        mock_response.raise_for_status = MagicMock()

    return mock_response


# --- Test Functions (Keep existing tests) ---
# ... (test_check_pdf_url_validity, test_get_web_page_content, etc.) ...

# --- NEW Test Function for get_news_articles ---

@pytest.mark.parametrize("query, from_date, to_date, patch_config_key, expected_success, expected_num_articles, expected_error_substr, expected_from_in_params", [
    # Happy path, default dates
    ("good-news-query", None, None, True, True, 2, None, True),
    # Happy path, specified dates
    ("good-news-query", "2024-04-01", "2024-04-30", True, True, 2, None, "2024-04-01"),
    # No results found
    ("no-news-query", None, None, True, True, 0, None, True),
    # API Rate Limit Error
    ("rate-limit-query", None, None, True, False, 0, "Rate limit exceeded", True),
    # Other API Error Status
    ("api-error-query", None, None, True, False, 0, "NewsAPI Error (sourcesTooMany)", True),
    # Network Error (e.g., 503)
    ("network-error-query", None, None, True, False, 0, "NewsAPI request failed: 503 Server Error", True),
    # Network Timeout
    ("timeout-query", None, None, True, False, 0, "Timeout after", True),
    # Missing API Key in Config
    ("good-news-query", None, None, False, False, 0, "NewsAPI key not configured", False), # Expect immediate failure
])
# Patch 'requests.get' specifically where it's called in tools.py
@patch('tools.requests.get', side_effect=mock_requests_get_news)
def test_get_news_articles(mock_get, mocker, query, from_date, to_date, patch_config_key, expected_success, expected_num_articles, expected_error_substr, expected_from_in_params):
    """Tests the get_news_articles tool with various scenarios."""
    logger.info(f"\n--- Testing Get News: Query='{query}', From='{from_date}', To='{to_date}' ---")

    # Mock date.today() for consistent default date calculation
    fixed_today = real_date(2024, 5, 10)
    mock_date_class = mocker.patch('tools.date', autospec=True)
    # Configure the 'today' method *on the mock class*
    mock_date_class.fromisoformat.side_effect = lambda d_str: real_date.fromisoformat(d_str)

    # Mock config API Key if needed for the specific test case
    if not patch_config_key:
        mocker.patch('tools.config.NEWS_API_KEY', None)
        logger.info("Patched config.NEWS_API_KEY to None for this test.")
    else:
        # Ensure config has a valid key for other tests (can be dummy if call is mocked)
        mocker.patch('tools.config.NEWS_API_KEY', "dummy_key_for_mock")

    # Run the tool function
    result = tools.get_news_articles(
        query=query,
        from_date_str=from_date,
        to_date_str=to_date
    )

    # Assertions
    assert isinstance(result, NewsApiResponse)
    assert result.success is expected_success

    if expected_success:
        assert result.error is None
        assert len(result.articles) == expected_num_articles
        if expected_num_articles > 0:
            assert isinstance(result.articles[0], NewsArticle)
            assert isinstance(result.articles[0].source, NewsSource)
            assert result.articles[0].url is not None # Check URL is present
            assert result.articles[0].title is not None
        # Check if the mock was called (only if API key was supposed to be present)
        if patch_config_key:
            mock_get.assert_called_once()
            call_args, call_kwargs = mock_get.call_args
            actual_params = call_kwargs.get("params", {})
            if expected_from_in_params is True:
                expected_default_from = (fixed_today - timedelta(days=config.NEWS_API_DEFAULT_DAYS_AGO)).isoformat()
                assert actual_params.get("from") == expected_default_from
            elif expected_from_in_params:
                assert actual_params.get("from") == expected_from_in_params

    else:
            assert result.articles == []
            assert result.error is not None
            if expected_error_substr:
                # Refined check for network/timeout errors
                if "503" in expected_error_substr:
                    assert "NewsAPI request failed" in result.error and ("503" in result.error or "Service Unavailable" in result.error) # <-- Check for reason phrase too
                elif "Timeout" in expected_error_substr:
                     assert ("NewsAPI request failed" in result.error or "Timeout after" in result.error) and "Timeout" in result.error # Allow both forms, ensure Timeout keyword
                else: # General check
                    assert expected_error_substr in result.error

# --- Mock Data for yfinance ---
MOCK_INFO_GOOD = {
    "currency": "USD", "symbol": "MOCKTICKER",
    "currentPrice": 150.50, "previousClose": 149.00, "dayHigh": 151.00, "dayLow": 148.50,
    "open": 149.50, "volume": 50000000, "marketCap": 2500000000000,
    "trailingPE": 25.5, "forwardPE": 22.0, "fiftyTwoWeekHigh": 160.00, "fiftyTwoWeekLow": 120.00,
    "averageVolume": 60000000
}
MOCK_INFO_PARTIAL = { # Missing currentPrice, PE ratios
    "currency": "USD", "symbol": "PARTIALTICKER",
    "previousClose": 95.00, "dayHigh": 96.00, "dayLow": 94.00,
    "open": 94.50, "volume": 10000000, "marketCap": 500000000000,
    "fiftyTwoWeekHigh": 100.00, "fiftyTwoWeekLow": 80.00,
    "averageVolume": 12000000
}
MOCK_INFO_MINIMAL_NEED_HISTORY = { # Only contains currency/symbol
    "currency": "INR", "symbol": "NEEDHIST.NS", "marketCap": 100000000000,
}
MOCK_INFO_INVALID = {} # yfinance often returns empty dict for invalid ticker

MOCK_HISTORY_GOOD = pd.DataFrame({'Close': [148.0, 149.0]}, index=pd.to_datetime(['2024-05-08', '2024-05-09']))
MOCK_HISTORY_EMPTY = pd.DataFrame()

# --- Mock yfinance.Ticker ---
def mock_yfinance_ticker_side_effect(ticker_symbol):
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.ticker = ticker_symbol # Store ticker for history mock

    if ticker_symbol == "MOCKTICKER":
        mock_ticker_instance.info = MOCK_INFO_GOOD
        mock_ticker_instance.history.return_value = MOCK_HISTORY_GOOD # Not needed if info has price
    elif ticker_symbol == "PARTIALTICKER":
        mock_ticker_instance.info = MOCK_INFO_PARTIAL
        mock_ticker_instance.history.return_value = MOCK_HISTORY_GOOD # Not needed if info has price
    elif ticker_symbol == "NEEDHIST.NS":
        mock_ticker_instance.info = MOCK_INFO_MINIMAL_NEED_HISTORY
        # Define history side effect based on stored ticker
        mock_ticker_instance.history.side_effect = lambda period: MOCK_HISTORY_GOOD if mock_ticker_instance.ticker == "NEEDHIST.NS" else MOCK_HISTORY_EMPTY
    elif ticker_symbol == "NOHISTORY":
         mock_ticker_instance.info = MOCK_INFO_MINIMAL_NEED_HISTORY
         mock_ticker_instance.history.return_value = MOCK_HISTORY_EMPTY # Simulate history failing too
    elif ticker_symbol == "YFINERROR":
         mock_ticker_instance.info = {} # Empty info
         mock_ticker_instance.history.side_effect = Exception("Simulated yfinance error") # Simulate history error
    else: # Default invalid ticker
        mock_ticker_instance.info = MOCK_INFO_INVALID
        mock_ticker_instance.history.return_value = MOCK_HISTORY_EMPTY

    return mock_ticker_instance

# --- Add this new test function ---
@pytest.mark.parametrize("ticker, expected_success, expected_data_keys, expected_error_substr", [
    # Happy path US
    ("MOCKTICKER", True, ["currentPrice", "marketCap", "trailingPE"], None),
    # Partial data, price available
    ("PARTIALTICKER", True, ["previousClose", "marketCap"], None),
    # Minimal info, price found in history, INR
    ("NEEDHIST.NS", True, ["currency", "marketCap", "currentPrice"], None), # currentPrice added from history
    # Minimal info, no price in history
    ("NOHISTORY", False, None, "Could not retrieve valid price data"),
    # Invalid Ticker (empty info, empty history)
    ("INVALIDTICKER", False, None, "Could not retrieve valid price data"),
    # Error during history fetch
    ("YFINERROR", False, None, "Simulated yfinance error"), # Error comes from yfinance call
])
# Patch the Ticker class within the tools module where yfinance (yf) is imported
@patch('tools.yf.Ticker', side_effect=mock_yfinance_ticker_side_effect)
def test_get_stock_data_unit(mock_ticker_constructor, ticker, expected_success, expected_data_keys, expected_error_substr):
    """Unit tests the get_stock_data tool by mocking yfinance."""
    logger.info(f"\n--- Unit Testing Get Stock Data: Ticker='{ticker}' ---")

    # Run the tool function
    result = tools.get_stock_data(ticker_symbol=ticker)

    # Assertions
    assert isinstance(result, StockData)
    assert result.success is expected_success
    assert result.symbol == ticker

    if expected_success:
        assert result.error is None
        assert result.data is not None
        assert isinstance(result.data, dict)
        assert len(result.data) > 0 # Should have some data
        if expected_data_keys:
            for key in expected_data_keys:
                assert key in result.data, f"Expected key '{key}' not found in data for {ticker}"
        # Check price specifically if history was used
        if ticker == "NEEDHIST.NS":
             assert result.data.get("currentPrice") == 149.0 # From MOCK_HISTORY_GOOD
    else:
        assert result.data is None or not result.data # Data should be None or empty dict on failure
        assert result.error is not None
        if expected_error_substr:
            assert expected_error_substr in result.error

    # Verify Ticker was instantiated
    mock_ticker_constructor.assert_called_with(ticker)
