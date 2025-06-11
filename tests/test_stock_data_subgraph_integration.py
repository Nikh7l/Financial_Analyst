# tests/test_stock_data_subgraph_integration.py
import os
import sys
import logging
import pytest
import time
import pprint
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

# --- Add project root to sys.path ---
# TODO: Remove sys.path manipulation block below.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# Import necessary components
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
import config # Ensure config is loaded
# Import the subgraph's state definition AND the compiled runnable
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
from agents.stock_agent_subgraph import StockDataSubgraphState, stock_data_subgraph_runnable
# Import yfinance exceptions for potential skipping

# Import Gemini library (needed for the internal ReAct agent)
from google import genai

# Configure logging if running directly
if __name__ == "__main__":
     if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from config import logger

# --- Test Cases ---
# Include valid US, valid Indian, and invalid tickers
TEST_TICKERS = [
    ("APPLE", True),        # Valid US
    ("Microsoft", True),        # Valid US
    ("RELIANCE", True), # Valid Indian
    ("Infosys", True),     # Valid Indian
    ("INVALIDTICKERXYZ123", False), # Invalid
    # ("", True),        # Valid US (Class C)
]

# --- Integration Test Function ---
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv('RUN_INTEGRATION_TESTS'), reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests")
@pytest.mark.parametrize("ticker_symbol, is_valid_ticker", TEST_TICKERS)
def test_stock_data_subgraph_live(ticker_symbol, is_valid_ticker):
    """
    Performs an integration test of the stock_data_subgraph_runnable
    using live LLM and yfinance/search calls.
    """
    logger.info(f"\n--- Running LIVE Stock Data Subgraph Test for: {ticker_symbol} ---")

    # Ensure API key is available
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=dotenv_path)
    if not config.GOOGLE_API_KEY:
        pytest.skip("GOOGLE_API_KEY not found, skipping integration test.")

    # Configure genai if needed (although agent graph might do it internally)
    # try:
    #     genai.configure(api_key=config.GOOGLE_API_KEY)
    # except Exception as e:
    #      logger.warning(f"Optional genai configure failed (might be ok): {e}")


    # Prepare initial state for the subgraph run
    # Set max_attempts (e.g., 2 for one initial try + one retry)
    initial_subgraph_state: StockDataSubgraphState = {
        "company_name": ticker_symbol,
        "attempt": 1,
        "max_attempts": 2,
        "messages": [], # ReAct agent starts with empty message history in subgraph state
        "stock_data": None,
        "subgraph_error": None,
    }

    final_subgraph_state = None
    start_time = time.time()
    try:
        # Invoke the compiled subgraph directly
        logger.info(f"Invoking stock data subgraph for {ticker_symbol}...")
        # Provide a recursion limit for the internal ReAct agent
        # and potentially overall config for the subgraph itself if needed
        # graph_config = {"recursion_limit": 30} # Limit for internal ReAct loop
        final_subgraph_state = stock_data_subgraph_runnable.invoke(
            initial_subgraph_state,
            # config=graph_config
        )
    except Exception as e:
         # Catch unexpected errors during the subgraph invocation
         pytest.fail(f"Subgraph invocation failed unexpectedly for {ticker_symbol}: {e}", pytrace=True)

    end_time = time.time()
    logger.info(f"Subgraph execution finished in {end_time - start_time:.2f} seconds.")
    logger.info("Final Subgraph State:")
    pprint.pprint(final_subgraph_state, indent=2) # Pretty print the final state

    # --- Assertions on the Final Subgraph State ---
    assert final_subgraph_state is not None, "Subgraph did not return a final state."
    assert isinstance(final_subgraph_state, dict), "Final state is not a dictionary."

    final_data = final_subgraph_state.get("stock_data")
    final_error = final_subgraph_state.get("subgraph_error")

    if is_valid_ticker:
        # For valid tickers, we expect *either* data OR an error if retries failed
        assert final_data is not None or final_error is not None, \
            f"Expected either data or an error for valid ticker {ticker_symbol}, got neither."

        if final_data is not None:
             # If data is present, error should ideally be None (or maybe a minor warning)
             assert final_error is None or "Essential data" not in final_error, \
                 f"Got data for {ticker_symbol} but also an unexpected final error: {final_error}"
             assert isinstance(final_data, dict), f"Expected data to be dict for {ticker_symbol}"
             # Flexible check for price/market cap
             assert any(k in final_data for k in ["currentPrice", "regularMarketPrice", "previousClose", "open"]), \
                 f"Expected price field for {ticker_symbol}, found none in {final_data.keys()}"
             assert "marketCap" in final_data or "error" in final_data.get("marketCap",""),\
                  f"Expected 'marketCap' for {ticker_symbol}, but not found or error in {final_data.keys()}" #allow for error in marketCap field
             logger.info(f"  -> Data checks passed for valid ticker {ticker_symbol}.")
        else:
             # If no data, there MUST be an error explaining why
             assert final_error is not None, \
                 f"Expected an error message for {ticker_symbol} when no data was returned, got None."
             logger.warning(f"  -> No data returned for valid ticker {ticker_symbol}, Error: {final_error}")

    else: # For invalid tickers
        assert final_data is None or not final_data, \
            f"Expected no data for invalid ticker {ticker_symbol}, but got: {final_data}"
        assert final_error is not None, \
            f"Expected an error message for invalid ticker {ticker_symbol}, got None."
        logger.info(f"  -> Error checks passed for invalid ticker {ticker_symbol} (Error: {final_error})")


    logger.info("Pausing briefly...")
    time.sleep(2) # Slightly longer pause maybe needed


# --- Main Block for Direct Execution ---
if __name__ == "__main__":
     import pprint # Ensure imported
     import re # Ensure imported

     print("--- Running Stock Data Subgraph Integration Test Script ---")
     load_dotenv(os.path.join(project_root, '.env'))

     # Configure client if needed globally for tools called by agent
     if not config.GOOGLE_API_KEY: print("Missing GOOGLE_API_KEY"); exit()
    #  try: genai.configure(api_key=config.GOOGLE_API_KEY)
    #  except Exception as e: print(f"Client config failed: {e}"); exit()

     # (Optional) Apply requests monkey-patch here if needed
     # ...

     test_results = {}
     # Use the same list as parametrize for consistency
     for ticker, is_valid in TEST_TICKERS:
         print(f"\n>>> Testing Stock Data Subgraph for: {ticker} <<<\n")
         try:
             # Call the test function directly for its logic & assertions
             # Note: pytest.skip won't work here, test will error on RateLimit
             test_stock_data_subgraph_live(ticker, is_valid)
             test_results[ticker] = "Passed (or Skipped)" # Assume pass if no exception from fail/assert
        #  except yf.exceptions.YFRateLimitError as rle:
        #       print(f"!!! Test for {ticker} SKIPPED due to Rate Limit: {rle} !!!")
        #       test_results[ticker] = "Skipped (Rate Limit)"
         except (pytest.fail.Exception, AssertionError) as fail_err: # Catch pytest.fail or Assert errors
              print(f"!!! Test for {ticker} FAILED: {fail_err} !!!")
              test_results[ticker] = f"Failed: {fail_err}"
         except Exception as e:
              print(f"!!! Critical error during test for {ticker}: {e} !!!")
              test_results[ticker] = f"ERROR: {e}"
         print("\n>>> Test Completed <<<\n")


     print("\n--- Stock Data Subgraph Integration Test Script Finished ---")
     print("Summary of results:")
     pprint.pprint(test_results)