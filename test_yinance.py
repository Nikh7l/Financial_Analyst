# tests/test_stock_data_integration.py
import os
import sys
import logging
import pytest
import time # For delays
from dotenv import load_dotenv

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# Import necessary components
# import config # Ensure config is loaded
# from state import AgentState # Not used directly, but good practice
import tools # Need the real tool functions
from tools import StockData # Import the response model

# Import yfinance potentially for error types, though not strictly necessary
import yfinance as yf
from config import logger

# --- Integration Test Function ---
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv('RUN_INTEGRATION_TESTS'), reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests")
@pytest.mark.parametrize("ticker_symbol, is_valid_ticker", [
    # US Stocks
    ("AAPL", True),
    # ("MSFT", True),
    # ("GOOGL", True),
    # # Indian Stocks (NSE)
    # ("RELIANCE.NS", True),
    # ("TCS.NS", True),
    # ("INFY.NS", True),
    # Invalid Ticker
    ("INVALIDTICKERXYZ123", False),
    # Potentially delisted or problematic ticker
    # ("SOME_DELISTED_TICKER", False), # Add if you know one
])
def test_get_stock_data_live(ticker_symbol, is_valid_ticker):
    """
    Performs an integration test of get_stock_data using live yfinance calls.
    Uses flexible assertions due to the dynamic nature of market data.
    """
    logger.info(f"\n--- Running LIVE Stock Data Test for: {ticker_symbol} ---")

    result = None
    try:
        # Run the REAL tool function - this makes network calls
        result = tools.get_stock_data(ticker_symbol=ticker_symbol)

        logger.info(f"Tool returned: Success={result.success}, Data Keys={list(result.data.keys()) if result.data else 'None'}, Error={result.error}")

    except Exception as e:
         # Catch any unexpected errors during the tool call itself
         pytest.fail(f"Tool call failed unexpectedly for {ticker_symbol}: {e}", pytrace=True)

    # --- Flexible Assertions ---
    assert result is not None, "Tool run did not return a result."
    assert isinstance(result, StockData)
    assert result.symbol == ticker_symbol

    if is_valid_ticker:
        # For valid tickers, we expect success, though data might be partial
        assert result.success is True, f"Expected success for valid ticker {ticker_symbol}, but got error: {result.error}"
        assert result.error is None, f"Expected no error for valid ticker {ticker_symbol}, but got: {result.error}"
        assert result.data is not None, f"Expected data dict for valid ticker {ticker_symbol}, but got None."
        assert isinstance(result.data, dict), f"Expected data to be a dict for {ticker_symbol}."
        # Check for presence of *some* key fields - these might vary based on market hours/data availability
        # Prioritize checking for a price-like field first
        assert any(key in result.data for key in ["currentPrice", "regularMarketPrice", "previousClose", "open"]), \
            f"Expected some price field (currentPrice, previousClose, etc.) for {ticker_symbol}, but found none in {result.data.keys()}"
        # Check for market cap as another indicator
        assert "marketCap" in result.data, f"Expected 'marketCap' for {ticker_symbol}, but not found in {result.data.keys()}"
        logger.info(f"  -> Data checks passed for valid ticker {ticker_symbol}.")
    else:
        # For invalid tickers, we expect failure
        assert result.success is False, f"Expected failure for invalid ticker {ticker_symbol}, but success was True."
        assert result.error is not None, f"Expected an error message for invalid ticker {ticker_symbol}, but got None."
        assert result.data is None or not result.data, f"Expected no data for invalid ticker {ticker_symbol}, but got: {result.data}"
        logger.info(f"  -> Error checks passed for invalid ticker {ticker_symbol} (Error: {result.error})")

    # Add a delay to avoid potential rate limits from Yahoo Finance
    logger.info("Pausing briefly before next integration test case...")
    time.sleep(1) # Pause for 1 second