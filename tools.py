# tools.py
import requests

old_request = requests.Session.request

def new_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return old_request(self, *args, **kwargs)

requests.Session.request = new_request

import os
import base64
from functools import wraps
from typing import Any, Callable, List, Optional ,Dict
import yfinance as yf
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from pydantic import BaseModel, Field , HttpUrl 
from strip_tags import strip_tags # Keep if preferred for simple cleaning
from google import genai
from google.genai import types
from datetime import date, timedelta
from tenacity import retry , stop_after_attempt,wait_exponential,retry_if_exception_type
# Import configuration settings
import config 

# Get logger (configured in config.py)
from config import logger

# --- Pydantic Models for Tool Outputs ---
# (Keep most of your models, maybe add success/error fields)
import warnings
warnings.filterwarnings("ignore")

# --- Define Retryable Exceptions for Search Tools ---
RETRYABLE_REQUEST_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError, # Can happen with unstable connections
    httpx.ConnectError, # If using httpx elsewhere
    httpx.TimeoutException,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.NetworkError
)

RETRYABLE_HTTP_STATUS_CODES = (
    429, # Too Many Requests
    500, # Internal Server Error
    502, # Bad Gateway
    503, # Service Unavailable
    504  # Gateway Timeout
)

def should_retry_requests_http_error(exception):
    """Predicate to check if a requests.exceptions.HTTPError should be retried."""
    return isinstance(exception, requests.exceptions.HTTPError) and \
           exception.response is not None and \
           exception.response.status_code in RETRYABLE_HTTP_STATUS_CODES

def should_retry_ddg_search_exception(exception):
    """Predicate to check if DuckDuckGoSearchException should be retried (for rate limits)."""
    if isinstance(exception, DuckDuckGoSearchException):
        # The library often includes the status code or "Ratelimit" in the message
        # This is a basic check; you might need to refine it based on exact error messages
        msg = str(exception).lower()
        if "ratelimit" in msg or "202" in msg or "429" in msg: # 202 is sometimes used by DDG for rate limits
            logger.warning(f"DDG rate limit or retryable error encountered: {exception}. Retrying...")
            return True
    return False

class BaseToolResponse(BaseModel):
    """Base model for tool responses including success status and errors."""
    success: bool = True
    error: Optional[str] = None

class URLValidityResult(BaseToolResponse):
    is_valid: Optional[bool] = Field(None, description="Boolean indicating if the URL is a valid and accessible PDF document.")

class PageSummary(BaseModel):
    page_title: str
    page_summary: str # Keep original field names for compatibility if needed
    page_url: str

class SearchResponse(BaseToolResponse):
    page_summaries: List[PageSummary] = []

class GoogleSearchResult(BaseModel):
    title: str
    link: str
    snippet: str
    body: Optional[str] = None # Make body optional as fetching can fail

class GoogleSearchResponse(BaseToolResponse):
    results: List[GoogleSearchResult] = []

class FullPageContent(BaseToolResponse):
    page_url: str
    content: Optional[str] = None

class FinancialSummary(BaseToolResponse):
    summary: Optional[str] = Field(None, description="Summary of the financial document")

class NewsSource(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None

class NewsArticle(BaseModel):
    source: Optional[NewsSource] = None
    # author: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    url: Optional[HttpUrl] = None # Use HttpUrl for validation
    # urlToImage: Optional[HttpUrl] = None
    # publishedAt: Optional[str] = None # Keep as string for now (ISO 8601)
    content: Optional[str] = None # Truncated content

class NewsApiResponse(BaseToolResponse):
    totalResults: Optional[int] = None
    articles: List[NewsArticle] = []

class StockData(BaseToolResponse):
    symbol: str
    # Make data optional as fetching might fail partially
    data: Optional[Dict[str, Any]] = Field(None, description="Dictionary containing stock data like price, P/E, market cap etc.")

# --- Error Handling Decorator ---
def catch_tool_exceptions(func: Callable) -> Callable:
    """Decorator to catch exceptions during tool execution and return error info."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            # Call the original function
            result = func(*args, **kwargs)
            # Ensure the result is a Pydantic model inheriting from BaseToolResponse
            if isinstance(result, BaseToolResponse):
                return result # Success case handled by the tool's own return model
            else:
                 # If the tool didn't return a BaseToolResponse on success, wrap it or log warning
                 logger.warning(f"Tool '{func.__name__}' did not return a BaseToolResponse subclass.")
                 # Attempt to return the raw result - agent must handle this
                 return result
        except Exception as e:
            logger.warning(f"Exception in tool '{func.__name__}': {e}", exc_info=True) # Log full traceback
            # Attempt to return the base model type of the function's annotation
            # This is heuristic - might not always work perfectly.
            return_type = func.__annotations__.get('return')
            if isinstance(return_type, type) and issubclass(return_type, BaseToolResponse):
                # Return an instance of the expected type with error info
                return return_type(success=False, error=str(e))
            else:
                # Fallback if return type isn't a BaseToolResponse subclass
                return BaseToolResponse(success=False, error=str(e))
    return wrapper

# --- Tool Implementations ---

@catch_tool_exceptions
def check_pdf_url_validity(url: str) -> URLValidityResult:
    """
    Checks if a given URL is accessible and likely points to a PDF.

    Args:
        url (str): The URL to check.

    Returns:
        URLValidityResult: Result indicating validity and any error message.
    """
    try:
        # headers = {'User-Agent': f'FinancialAgent/1.0 ({os.name})'} # More informative user agent
        # Use default SSL verification (recommended)
        response = requests.head(url, timeout=config.URL_TIMEOUT, allow_redirects=True, verify=False)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' in content_type:
            logger.debug(f"URL '{url}' appears to be a valid PDF.")
            return URLValidityResult(is_valid=True,success=True)
        else:
            logger.warning(f"URL '{url}' does not have PDF Content-Type: {content_type}")
            return URLValidityResult(is_valid=False,success=False, error=f"Content-Type is not application/pdf: {content_type}")

    except requests.exceptions.Timeout:
         logger.warning(f"Timeout checking URL validity for: {url}")
         return URLValidityResult(is_valid=False,success=False, error=f"Timeout after {config.URL_TIMEOUT}s")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Request exception checking URL '{url}': {e}")
        return URLValidityResult(is_valid=False,success=False, error=str(e))
    except Exception as e: # Catch unexpected errors
        logger.error(f"Unexpected error checking URL '{url}': {e}", exc_info=True)
        return URLValidityResult(is_valid=False,success=False, error=f"Unknown error: {e}")
        

@catch_tool_exceptions
def get_web_page_content(url: str, use_bs4: bool = True, max_chars: int = 4000) -> FullPageContent:
    """
    Fetches and extracts text content from a web page URL.

    Args:
        url (str): The URL of the web page.
        use_bs4 (bool): If True, uses BeautifulSoup for potentially cleaner extraction.
                        If False, uses simple strip_tags. Defaults to True.
        max_chars (int): Maximum characters to return (approximate).

    Returns:
        FullPageContent: Object containing the URL and extracted content.
    """
    try:
        headers = {'User-Agent': f'FinancialAgent/1.0 ({os.name})'}
        response = requests.get(url, timeout=config.URL_TIMEOUT, headers=headers) # Use default verification
        response.raise_for_status()

        content = None
        if use_bs4:
            soup = BeautifulSoup(response.content, "html.parser")
            # Attempt to get main text, removing script/style tags
            for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
                element.decompose()
            text = soup.get_text(separator=" ", strip=True)
            content = ' '.join(text.split())[:max_chars] # Basic whitespace normalization and trimming
        else:
            # Simpler method using strip_tags
            html = response.text
            text = strip_tags(html)
            content = "\n".join([line for line in text.split("\n") if line.strip()])[:max_chars]

        logger.debug(f"Successfully fetched content (first 100 chars): {content[:100]}... from {url}")
        return FullPageContent(page_url=url, content=content)

    except requests.exceptions.Timeout:
         logger.warning(f"Timeout fetching content from: {url}")
         return FullPageContent(page_url=url, success=False, error=f"Timeout after {config.URL_TIMEOUT}s")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Request exception fetching content from '{url}': {e}")
        return FullPageContent(page_url=url, success=False, error=str(e))
    except Exception as e:
        logger.error(f"Unexpected error fetching content from '{url}': {e}", exc_info=True)
        raise # Let decorator handle

@catch_tool_exceptions
def summarize_pdf_document_finance(doc_url: str, client: genai.Client) -> FinancialSummary:
    """
    Retrieves a PDF from a URL, encodes it, and uses the provided Gemini Client
    to summarize it focusing on financial insights.

    Args:
        doc_url (str): URL of the PDF document.
        client (genai.Client): An initialized google.genai Client instance.

    Returns:
        FinancialSummary: Object containing the financial summary.
    """
    try:
        logger.info(f"Attempting to download and summarize PDF: {doc_url}")
        # Download PDF content
        response = httpx.get(doc_url, timeout=config.PDF_DOWNLOAD_TIMEOUT, follow_redirects=True,verify=False) # Use httpx, default verification
        response.raise_for_status()

        # Check content type again just to be sure
        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' not in content_type:
             logger.warning(f"Content-Type mismatch for PDF download: {content_type} from {doc_url}")
             return FinancialSummary(success=False, error=f"Downloaded content is not PDF: {content_type}")

        # Encode the PDF data
        pdf_data = base64.b64encode(response.content).decode("utf-8")
        logger.debug(f"PDF downloaded and encoded successfully from {doc_url} (Size: {len(response.content)} bytes)")

        # Define the prompt (Consider moving to prompts.py if it gets complex)
        prompt = """
        You are an expert financial analyst. Go through the following PDF document, which is likely a financial report (e.g., quarterly/annual earnings). Focus on extracting key financial information relevant for stock analysis and investment decisions. Include:
        - Company name and report date (if available).
        - Report type (e.g., earnings report, annual report).
        - Period covered (e.g., Q3 2023, FY 2023).
        - Key financial figures (Revenue, Net Income, EPS) and notable changes (YoY, QoQ).
        - Major trends in financial performance.
        - Management commentary on results and future outlook.
        - Key risks and opportunities highlighted.
        - Any other crucial insights for investors.
        Make a detailed report with multiple sections , it should be multiple paragraphs , include all relevant information.
        This information is critical for investors and analysts. Please be thorough and precise in your summary.
        """

        # Prepare content for the Gemini model
        contents = [
            types.Part.from_text(text=prompt),
            types.Part(inline_data=types.Blob(mime_type='application/pdf', data=pdf_data))
        ]

        # Prepare generation config (optional, can add temperature etc. from config.py)
        gen_config = types.GenerateContentConfig(
            temperature=config.TEMPERATURE # Example using config
            # Add other parameters like max_output_tokens if needed
        )

        # Call the Gemini model via the provided client
        logger.info(f"Sending PDF summary request to model '{config.GEMINI_MODEL_NAME}'...")
        gemini_response = client.models.generate_content(
            model=config.GEMINI_MODEL_NAME,
            contents=contents,
            config=gen_config
        )

        summary_text = gemini_response.text # Access text directly with new SDK
        logger.info(f"Successfully generated summary for {doc_url}")
        return FinancialSummary(summary=summary_text)

    except httpx.TimeoutException:
        logger.warning(f"Timeout downloading PDF from: {doc_url}")
        return FinancialSummary(success=False, error=f"Timeout after {config.PDF_DOWNLOAD_TIMEOUT}s downloading PDF")
    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP status error downloading PDF from '{doc_url}': {e}")
        # Format the error as expected by the test
        return FinancialSummary(success=False, error=f"HTTP error downloading PDF: {e.response.status_code} {e.response.reason_phrase}")
    except httpx.RequestError as e:
        logger.warning(f"HTTP request error downloading PDF from '{doc_url}': {e}")
        return FinancialSummary(success=True, error=f"HTTP error downloading PDF: {e}")
    except Exception as e:
        logger.error(f"Unexpected error summarizing PDF '{doc_url}': {e}", exc_info=True)
        raise # Let decorator handle formatting this error


@catch_tool_exceptions
def search_duck_duck_go(query: str, num_results: int = config.SEARCH_RESULTS_LIMIT) -> SearchResponse:
    """
    Performs a search using DuckDuckGo and returns summaries.

    Args:
        query (str): The search query.
        num_results (int): The desired number of results.

    Returns:
        SearchResponse: Object containing a list of page summaries.
    """
    logger.info(f"Performing DuckDuckGo search for: '{query}' (max {num_results} results)")
    # DDGS handles context management internally now
    results = DDGS(verify = False).text(query, max_results=num_results) # Use default verification

    summaries = [
        PageSummary(page_title=r["title"], page_summary=r["body"], page_url=r["href"])
        for r in results
    ]
    logger.debug(f"DDG Search returned {len(summaries)} results.")
    return SearchResponse(page_summaries=summaries)


@catch_tool_exceptions
def google_search(query: str, num_results: int = config.SEARCH_RESULTS_LIMIT, fetch_body: bool = False) -> GoogleSearchResponse:
    """
    Performs a Google search using the Custom Search API. Optionally fetches page content.

    Args:
        query (str): The search query.
        num_results (int): The desired number of results.
        fetch_body (bool): Whether to attempt fetching the body content of each result URL. Defaults to False.

    Returns:
        GoogleSearchResponse: Object containing a list of search results.
    """
    logger.info(f"Performing Google Custom Search for: '{query}' (max {num_results} results, fetch body: {fetch_body})")
    api_key = config.GOOGLE_SEARCH_API_KEY
    search_engine_id = config.GOOGLE_CSE_ID

    if not api_key or not search_engine_id:
        logger.error("Google Search API Key or CSE ID is not configured.")
        return GoogleSearchResponse(success=False, error="Google API Key/CSE ID not configured.")

    url = "https://customsearch.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": search_engine_id, "q": query, "num": num_results}

    try:
        # Use default verification for the API call itself
        response = requests.get(url, params=params, timeout=config.URL_TIMEOUT,verify=False)
        response.raise_for_status()

        results_data = response.json().get("items", [])
        enriched_results = []

        for item in results_data:
            result = GoogleSearchResult(
                title=item.get("title", ""),
                link=item.get("link", ""),
                snippet=item.get("snippet", ""),
                body=None # Initialize body as None
            )
            if fetch_body and result.link:
                logger.debug(f"Fetching body content for Google result: {result.link}")
                content_result = get_web_page_content(url=result.link)
                if content_result.success:
                    result.body = content_result.content
                else:
                    logger.warning(f"Failed to fetch body for {result.link}: {content_result.error}")
                # Optional: Add a small delay if fetching many bodies to be polite
                # import time
                # time.sleep(0.5)
            enriched_results.append(result)

        logger.debug(f"Google Search returned {len(enriched_results)} results.")
        return GoogleSearchResponse(results=enriched_results)

    except requests.exceptions.Timeout:
         logger.warning(f"Timeout during Google Search API call for: '{query}'")
         return GoogleSearchResponse(success=False, error=f"Timeout after {config.URL_TIMEOUT}s during API call")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in Google Search API request for '{query}': {e}", exc_info=True)
        # Check if response body contains more error details
        error_detail = str(e)
        try:
            if e.response is not None:
                 error_detail = f"{e} - Response: {e.response.text[:500]}"
        except Exception:
            pass # Ignore errors trying to get more details
        return GoogleSearchResponse(success=False, error=f"Google API request failed: {error_detail}")
    except Exception as e:
        logger.error(f"Unexpected error during Google Search for '{query}': {e}", exc_info=True)
        raise # Let decorator handle




@catch_tool_exceptions
def get_news_articles(
    query: str,
    sort_by: Optional[str] = 'popularity', # Options: relevancy, popularity, publishedAt
) -> NewsApiResponse:
    """
    Fetches recent news articles related to a query using the NewsAPI.org /v2/everything endpoint.

    Args:
        query (str): Keywords or phrases to search for (e.g., company name, ticker).
        sort_by (str): Order of results ('relevancy', 'popularity', 'publishedAt'). Defaults to 'popularity'.

    Returns:
        NewsApiResponse: Object containing the list of articles or an error.
    """
    api_key = config.NEWS_API_KEY
    if not api_key:
        logger.error("NewsAPI key not found in configuration.")
        return NewsApiResponse(success=False, error="NewsAPI key not configured.")

    logger.info(f"Fetching news for query: '{query}', Language: english, SortBy: {sort_by}")

    # --- Calculate Date Range ---
    today = date.today()
    # Use provided dates or calculate defaults
    try:
        to_date = today
    except ValueError:
        logger.warning(f"Invalid to_date_str using today.")
        to_date = today

    try:        
        from_date = today - timedelta(days=config.NEWS_API_DEFAULT_DAYS_AGO)
    except ValueError:
         logger.warning(f"Invalid from_date_str , using default {config.NEWS_API_DEFAULT_DAYS_AGO} days ago.")
         from_date = today - timedelta(days=config.NEWS_API_DEFAULT_DAYS_AGO)

    # Format dates for API
    from_param = from_date.isoformat()
    to_param = to_date.isoformat()
    logger.debug(f"Using date range: {from_param} to {to_param}")
    # --- End Date Range ---

    # --- API Call ---
    endpoint = "https://newsapi.org/v2/everything" # Correct endpoint
    headers = {
        "X-Api-Key": api_key, # Use header authentication
        # "User-Agent": f"FinancialAgent/1.0 ({os.name})"
    }
    params = {
        "q": query,
        "from": from_param,
        "to": to_param,
        "language": "en",
        "sortBy": sort_by,
        "pageSize": min(config.NEWS_API_PAGE_SIZE, 100), # Respect API max limit
        # Optional: Add 'sources' or 'domains' if needed later
    }

    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=config.URL_TIMEOUT,verify=False) # Use default SSL verification
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        data = response.json()

        # Check API status
        if data.get("status") == "error":
            error_code = data.get("code", "UnknownCode")
            error_message = data.get("message", "Unknown API error")
            logger.error(f"NewsAPI returned an error: Code='{error_code}', Message='{error_message}'")
            # Return specific error codes users might understand
            if error_code == "rateLimited":
                 return NewsApiResponse(success=False, error="Rate limit exceeded for NewsAPI.")
            elif error_code == "apiKeyInvalid" or error_code == "apiKeyMissing":
                 return NewsApiResponse(success=False, error="Invalid or missing NewsAPI key.")
            else:
                return NewsApiResponse(success=False, error=f"NewsAPI Error ({error_code}): {error_message}")

        # Process successful response
        if data.get("status") == "ok":
            raw_articles = data.get("articles", [])
            parsed_articles = []
            for article_data in raw_articles:
                # Handle nested source field
                source_data = article_data.get('source', {})
                source = NewsSource(id=source_data.get('id'), name=source_data.get('name'))
                try:
                    # Create NewsArticle object, Pydantic handles missing Optional fields
                    parsed_article = NewsArticle(
                        source=source,
                        # author=article_data.get('author'),
                        title=article_data.get('title'),
                        description=article_data.get('description'),
                        url=article_data.get('url'),
                        # urlToImage=article_data.get('urlToImage'),
                        # publishedAt=article_data.get('publishedAt'),
                        content=article_data.get('content')
                    )
                    parsed_articles.append(parsed_article)
                except Exception as p_err: # Catch validation errors if HttpUrl fails etc.
                     logger.warning(f"Could not parse article due to validation error: {p_err}. Data: {article_data}")

            total_results = data.get("totalResults", 0)
            logger.info(f"Successfully fetched {len(parsed_articles)} news articles (Total available: {total_results}).")
            # log the first few articles for debugging
            logger.debug(f"First few articles: {parsed_articles[:3]}")
            return NewsApiResponse(
                success=True,
                totalResults=total_results,
                articles=parsed_articles
            )
        else:
             # Should not happen if status isn't 'ok' or 'error', but handle defensively
             logger.error(f"NewsAPI returned unexpected status: {data.get('status')}")
             return NewsApiResponse(success=False, error=f"Unexpected status from NewsAPI: {data.get('status')}")

    except requests.exceptions.Timeout:
         logger.warning(f"Timeout during NewsAPI call for query: '{query}'")
         return NewsApiResponse(success=False, error=f"Timeout after {config.URL_TIMEOUT}s during NewsAPI call")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during NewsAPI request for query '{query}': {e}", exc_info=False)
        error_detail = str(e)
        try: # Add response details if available
             if e.response is not None: error_detail = f"{e} - Response: {e.response.text[:200]}"
        except Exception: pass
        return NewsApiResponse(success=False, error=f"NewsAPI request failed: {error_detail}")
