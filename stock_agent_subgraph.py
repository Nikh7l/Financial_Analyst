# agents/stock_data_worker.py
import logging
import json
import re
from typing import Dict, Any, List, Optional, TypedDict
import os 
import sys
# Langchain/LangGraph components
from langchain_core.pydantic_v1 import BaseModel, Field # Keep for potential internal use
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Project components
import config
# from prompts import STOCK_DATA_WORKER_PROMPT
from tools import search_duck_duck_go, get_web_page_content,google_search

from config import logger

class StockDataSubgraphState(TypedDict, total=False):
    company_name: str
    attempt: int
    max_attempts: int
    messages: List[BaseMessage]
    stock_data: Optional[Dict[str, Any]] # This will now hold the nested dict from the LLM
    source_urls_used: Optional[List[str]] # NEW: To store URLs from LLM output
    subgraph_error: Optional[str]
    _route_decision: Optional[str]

# --- List of Basic Tools for the ReAct Node ---
stock_data_react_tools = [
    google_search, 
    search_duck_duck_go,
    get_web_page_content,
]


try:
    worker_llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL_NAME,
        temperature=0.0,
        convert_system_message_to_human=True
    )
except Exception as e:
    logger.exception("Failed to initialize LLM for Stock Data Worker.")
    worker_llm = None

STOCK_DATA_WORKER_PROMPT_REVISED = """
You are a specialized financial data retrieval agent. Your goal is to fetch key stock data for the company mentioned in the human query using only web search tools.

**Target Data Points (Prioritize):**
- Current Price (or Previous Close), Currency, Market Cap, P/E Ratio (Trailing or Forward), 52 Week High/Low, Trading Volume, and the **primary URL(s)** from which this data was sourced.

**Your Process:**
1. **Identify Company/Ticker:** Extract the company name or ticker from the human query.
2. **Web Search:** Use `google_search` or `search_duck_duck_go` to search for up-to-date stock data for the company/ticker. Prioritize official financial sites (e.g., Yahoo Finance, Google Finance, major exchanges).
3. **Extract Data:** Review search snippets for the required data points. If snippets are insufficient, use `get_page_content` ONCE or TWICE on highly promising source URLs to extract missing data. Note the URL(s) you primarily use.
4. **Synthesize & Final Answer:** Combine data from the best source(s). Respond ONLY as a JSON object with two top-level keys:
   - "stock_data": A dictionary containing all the target data points found.
   - "source_urls_used": A list of strings (URLs) primarily used to gather the stock data (max 2-3 URLs).
   If essential data (like price) is missing, you can return an "error" key within the "stock_data" dictionary instead of the data, or provide what you found and note limitations.

Example Output:
```json
{{
  "stock_data": {{
    "ticker": "XYZ",
    "current_price": "150.00",
    "currency": "USD",
    "market_cap": "300B"
    // ... other data points ...
  }},
  "source_urls_used": ["https://finance.yahoo.com/quote/XYZ", "https://www.google.com/finance/quote/XYZ:NASDAQ"]
}}
"""

# --- End Revised Prompt ---

def create_stock_data_react_agent():
    if not worker_llm:
        raise RuntimeError("LLM for Stock Data Worker not initialized.")
    return create_react_agent(
            model=worker_llm,
            tools=stock_data_react_tools,
            prompt=STOCK_DATA_WORKER_PROMPT_REVISED ,\
            debug = True
        )

stock_data_react_runnable = create_stock_data_react_agent()

# --- Node Logic Functions for Subgraph ---

def react_stock_agent_node(state: StockDataSubgraphState) -> Dict[str, Any]:
    """Runs the ReAct agent for one attempt."""
    ticker = state["company_name"]
    attempt = state["attempt"]
    max_attempts = state["max_attempts"] # Needed for potential retry prompt logic
    logger.info(f"[StockWorker Subgraph] Attempt {attempt}/{max_attempts}: Running ReAct agent for {ticker}")

    # Prepare initial message based on attempt number
    if attempt == 1:
        human_message = f"Fetch stock data for {ticker}."
    else:
        prev_error = state.get("subgraph_error", "Previous attempt failed or yielded insufficient data.")
        human_message = f"Retry fetching stock data for {ticker}. {prev_error}. Focus on alternative methods (like search) if direct API failed previously."

    # Use the pre-compiled runnable
    react_input = {"messages": [HumanMessage(content=human_message)]}
    try:
        # We only need the final messages list from the ReAct run for parsing
        react_output_state = stock_data_react_runnable.invoke(react_input, {"recursion_limit": 15})
        return {"messages": react_output_state.get("messages", [])}
    except Exception as e:
        logger.error(f"[StockWorker Subgraph] Exception during ReAct execution for {ticker}: {e}", exc_info=True)
        return {"subgraph_error": f"ReAct agent execution failed: {e}", "messages": state.get("messages", []) + [AIMessage(content=f"Error: {e}")]}


def check_and_parse_result_node(state: StockDataSubgraphState) -> Dict[str, Any]:
    ticker = state["company_name"]
    attempt = state.get("attempt", 1)
    max_attempts = state.get("max_attempts", 1)
    messages = state.get("messages", [])
    logger.info(f"[StockWorker Subgraph] Checking result for {ticker} (Attempt {attempt})")

    actual_stock_data: Optional[Dict[str, Any]] = None # Renamed for clarity
    source_urls: Optional[List[str]] = None
    parse_error: Optional[str] = None
    next_step: str = "fail"

    if not messages or not isinstance(messages[-1], AIMessage):
        parse_error = "No messages or not an AIMessage from ReAct agent."
    else:
        final_message_content = messages[-1].content.strip()
        logger.debug(f"[StockWorker Subgraph] Final AI Message to parse: {final_message_content}")
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", final_message_content, re.DOTALL | re.IGNORECASE)
            if match: json_str = match.group(1)
            else:
                json_start = final_message_content.find('{'); json_end = final_message_content.rfind('}') + 1
                if json_start != -1 and json_end != -1: json_str = final_message_content[json_start:json_end]
                else: raise ValueError("JSON object not found")
            
            parsed_json = json.loads(json_str) # This is the full JSON from LLM {"stock_data":{}, "source_urls_used":[]}

            # Extract the main stock data dictionary and source URLs
            data_dict_from_llm = parsed_json.get("stock_data")
            source_urls_from_llm = parsed_json.get("source_urls_used", []) # Default to empty list

            if isinstance(source_urls_from_llm, list):
                source_urls = [url for url in source_urls_from_llm if isinstance(url, str) and url.startswith('http')]
            else:
                logger.warning("LLM 'source_urls_used' was not a list. Ignoring.")
                source_urls = []


            if isinstance(data_dict_from_llm, dict):
                if "error" in data_dict_from_llm: # LLM reported an error finding data
                    parse_error = f"Agent reported error: {data_dict_from_llm['error']}"
                    actual_stock_data = data_dict_from_llm # Store the error dict
                    if attempt < max_attempts: next_step = "retry"
                    else: next_step = "fail"
                else: # Data found
                    data_lower_keys = {k.lower(): v for k, v in data_dict_from_llm.items()}
                    has_price = any(k in data_lower_keys for k in ["current_price", "regularmarketprice", "previous_close"])
                    if data_dict_from_llm and has_price:
                        logger.info(f"Essential stock data found for {ticker}.")
                        actual_stock_data = data_dict_from_llm # Store the actual data
                        parse_error = None
                        next_step = "success"
                    else:
                        parse_error = "Essential data (e.g., price) missing in parsed stock_data from LLM."
                        actual_stock_data = data_dict_from_llm # Store what was found anyway
                        if attempt < max_attempts: next_step = "retry"
                        else: next_step = "fail"
            else:
                parse_error = "LLM JSON response missing 'stock_data' dictionary."
                if attempt < max_attempts: next_step = "retry"
                else: next_step = "fail"
        except (json.JSONDecodeError, ValueError) as e:
            parse_error = f"Failed to parse JSON from LLM response: {e}"
            if attempt < max_attempts: next_step = "retry"
            else: next_step = "fail"

    logger.info(f"[StockWorker Subgraph] Check complete. Routing: {next_step}. Error: {parse_error}. URLs: {source_urls}")
    return {
        "stock_data": actual_stock_data, # This is the dict under "stock_data" key from LLM
        "source_urls_used": source_urls, # NEW
        "subgraph_error": parse_error,
        "attempt": attempt + 1,
        "_route_decision": next_step
    }

# --- Graph Definition Function ---
def create_stock_data_graph() :
    """Builds the ReAct agent subgraph with retry/checking logic."""
    builder = StateGraph(StockDataSubgraphState)
    builder.add_node("react_agent", react_stock_agent_node)
    builder.add_node("check_result", check_and_parse_result_node)
    builder.add_edge(START, "react_agent")
    builder.add_edge("react_agent", "check_result")
    builder.add_conditional_edges(
        "check_result",
        lambda state: state.get("_route_decision", "fail"),
        {"success": END, "retry": "react_agent", "fail": END}
    )
    graph = builder.compile()
    logger.info("Stock Data Worker Subgraph compiled.")
    return graph

# --- Compiled Subgraph ---
# Compile it once for use in the main graph node
stock_data_subgraph_runnable = create_stock_data_graph()



# --- Main Block for Direct Subgraph Testing ---
if __name__ == '__main__':
    import pprint
    import time
    from dotenv import load_dotenv
    from google import genai # To configure client
    from config import logger
    # Configure logging for direct run
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Running Stock Data Worker Subgraph Directly ---")

    # Load .env variables
    load_dotenv()
    if not config.GOOGLE_API_KEY: print("ERROR: GOOGLE_API_KEY not set."); exit()
    if not config.GOOGLE_SEARCH_API_KEY: print("ERROR: GOOGLE_SEARCH_API_KEY not set."); exit() # Also check search key
    if not config.GOOGLE_CSE_ID: print("ERROR: GOOGLE_CSE_ID not set."); exit()

    # --- Test Cases ---
    test_cases = [
        # {"company_name": "Nvidia"},
        {"company_name": "Apple"},
        {"company_name": "Reliance Industries"}, # Test Indian market
        {"company_name": "NonExistentCompanyXYZ987"}, # Test failure case
    ]

    # Compile the graph runnable (already done above)
    subgraph_app = stock_data_subgraph_runnable

    # --- Run Tests ---
    all_final_states = {}
    for case in test_cases:
        company = case["company_name"]
        print(f"\n>>> Testing Stock Data Subgraph for: {company} <<<\n")

        # Initial state for the subgraph run
        initial_subgraph_state: StockDataSubgraphState = {
            "company_name": company, # Changed from ticker_symbol to match state
            "attempt": 1,
            "max_attempts": 2, # Allow one retry
            "messages": [],
            "stock_data": None,
            "subgraph_error": None,
        }

        final_state = None
        start_time = time.time()
        try:
            # Stream the execution to see intermediate states
            print("--- Streaming Subgraph Execution ---")
            for step_count, step_state in enumerate(subgraph_app.stream(
                initial_subgraph_state,
                {"recursion_limit": 30} # Limit for ReAct agent inside node
            )):
                node_name = list(step_state.keys())[0]
                node_output = step_state[node_name]
                print(f"\n=== Step {step_count+1}: Node '{node_name}' Output ===")
                # Print relevant parts of the state after this node ran
                current_state = node_output # The output dict IS the state update
                pprint.pprint({k: v for k, v in current_state.items() if k != 'messages'}, indent=2)
                # Print last message if messages updated
                if current_state.get('messages'):
                    last_msg = current_state['messages'][-1]
                    print(f"\n  Last Message ({last_msg.type}):")
                    try:
                        # Pretty print tool call args if present
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            print("    Tool Calls:")
                            for tc in last_msg.tool_calls:
                                print(f"      - Name: {tc['name']}")
                                print(f"        Args: {tc['args']}")
                        elif hasattr(last_msg, 'content'):
                             print(f"    Content: {str(last_msg.content)[:1000]}...") # Truncate long content
                    except Exception:
                         print(f"    Content: {last_msg}") # Fallback


                final_state = current_state # Keep track of the latest state snapshot
                time.sleep(0.5) # Small delay to make streaming readable

            print("\n--- Subgraph Execution Complete ---")
            logger.info("Final Subgraph State:")
            logger.info(f"Final State: {final_state}")
        except Exception as e:
            print(f"\n--- Subgraph Execution Failed for {company} ---")
            logger.exception("Error during subgraph execution:")
            print(f"Error: {e}")
            final_state = {"error": f"Subgraph execution failed: {e}"} # Store error

        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds.")
        print("\nFinal State Snapshot:")
        pprint.pprint(final_state, indent=2)
        all_final_states[company] = final_state
        print("\n>>> Test Completed <<<\n")
        print("Pausing before next test...")
        time.sleep(10) # Longer pause to help with rate limits

    print("\n--- Stock Data Subgraph Test Script Finished ---")