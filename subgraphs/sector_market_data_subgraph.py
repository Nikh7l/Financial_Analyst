import logging
import json
import re
from typing import List, Optional, Dict, Any, TypedDict, Literal

# --- Project Imports ---
from config import config
from core.prompts import SECTOR_MARKET_DATA_PROMPT_TEMPLATE_STATIC
from core.tools import google_search, search_duck_duck_go, get_web_page_content

# --- Langchain/LangGraph Imports ---
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage

# --- Constants ---
MAX_MARKET_DATA_ATTEMPTS = 2
logger = config.logger

# --- State Definition for this Subgraph ---
# Re-defining SectorMarketDataDetail here for clarity within the subgraph context
class SectorMarketDataDetailInternal(TypedDict, total=False):
    market_size_estimate: Optional[str]
    projected_cagr: Optional[str]
    key_market_segments: Optional[List[str]]
    key_geographies: Optional[List[str]]
    primary_growth_drivers: Optional[List[str]]
    primary_market_challenges: Optional[List[str]]

class SectorMarketDataSubgraphState(TypedDict, total=False):
    sector_name: str # Input
    messages: List[BaseMessage] # For ReAct agent
    attempt: int
    max_attempts: int
    # Output: The structure the ReAct agent will produce
    extracted_market_data: Optional[Dict[str, Any]] # Will hold {"market_data": {...}, "source_urls_used": [...]}
    subgraph_error: Optional[str]
    _route_decision: Optional[str]

# --- LLM Initialization ---
try:
    market_data_llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL_NAME, # Or a model good for research/extraction
        temperature=0.1, # Low temp for factual extraction
        convert_system_message_to_human=True
    )
    logger.info("LLM for SECTOR Market Data Agent initialized successfully.")
except Exception as e:
    logger.exception("Failed to initialize LLM for SECTOR Market Data Agent.")
    market_data_llm = None

# --- Tool List ---
sector_market_data_tools = [
    google_search,
    search_duck_duck_go,
    get_web_page_content,
]

# --- Create the ReAct Agent Runnable ---
try:
    if not market_data_llm:
        raise RuntimeError("LLM for SECTOR Market Data Agent not initialized.")
    sector_market_data_react_runnable = create_react_agent(
        model=market_data_llm,
        tools=sector_market_data_tools,
        prompt=SECTOR_MARKET_DATA_PROMPT_TEMPLATE_STATIC,
        debug=True
    )
    logger.info("Core ReAct SECTOR Market Data Agent runnable created successfully.")
except RuntimeError as e:
    logger.error(f"Error creating sector market data ReAct agent: {str(e)}")
    sector_market_data_react_runnable = None
except Exception as e:
    logger.exception("Unexpected error creating ReAct SECTOR Market Data Agent runnable.")
    sector_market_data_react_runnable = None

# --- Node Logic Functions for Subgraph ---

def start_market_data_collection_node(state: SectorMarketDataSubgraphState) -> Dict[str, Any]:
    sector_name = state["sector_name"]
    attempt = state.get("attempt", 1)
    logger.info(f"[SectorMarketDataSubgraph] Starting data collection for SECTOR: {sector_name}, Attempt: {attempt}")

    if attempt == 1:
        human_query_content = (
            f"Gather key market data (size, CAGR, segments, geographies, drivers, challenges) "
            f"for the **{sector_name} sector**. Adhere to the JSON output format specified in the system prompt."
        )
    else:
        prev_error = state.get("subgraph_error", "A previous attempt failed to retrieve complete data.")
        human_query_content = (
            f"Retry gathering market data for the **{sector_name} sector**. "
            f"Previous issue: {prev_error}. "
            "Focus on finding all requested data points and provide sources."
        )
    return {"messages": [HumanMessage(content=human_query_content)]}


def parse_and_check_market_data_node(state: SectorMarketDataSubgraphState) -> Dict[str, Any]:
    sector_name = state.get("sector_name", "Unknown Sector")
    attempt = state.get("attempt", 1)
    max_attempts_total = state.get("max_attempts", MAX_MARKET_DATA_ATTEMPTS)
    messages = state.get("messages", [])
    logger.info(f"[SectorMarketDataSubgraph] Parsing result for SECTOR: {sector_name} (Attempt {attempt})")

    parsed_output_data: Optional[Dict[str, Any]] = None
    current_subgraph_error: Optional[str] = state.get("subgraph_error")
    routing_decision: str = "fail"

    if not messages or not isinstance(messages[-1], AIMessage):
        parse_error_msg = f"ReAct agent ended with non-AI message: {type(messages[-1]) if messages else 'No Messages'}."
        current_subgraph_error = current_subgraph_error or parse_error_msg
        if attempt < max_attempts_total: routing_decision = "retry"
    else:
        ai_content = messages[-1].content.strip()
        logger.debug(f"[SectorMarketDataSubgraph] Final AI Message: {ai_content[:500]}...")
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", ai_content, re.DOTALL | re.IGNORECASE)
            if match: json_str = match.group(1)
            else:
                json_start = ai_content.find('{'); json_end = ai_content.rfind('}') + 1
                if json_start != -1 and json_end != -1 and json_end > json_start: json_str = ai_content[json_start:json_end]
                else: raise ValueError("JSON object not found in AI response.")
            
            parsed_json = json.loads(json_str) # This is the dict like {"market_data": {...}, "source_urls_used": []}
            if not isinstance(parsed_json, dict):
                raise ValueError("Parsed content is not a JSON object.")

            # Validate the main structure
            market_data_details = parsed_json.get("market_data")
            source_urls = parsed_json.get("source_urls_used", []) # Default to empty list

            if isinstance(market_data_details, dict) and isinstance(source_urls, list):
                # Further validate inner market_data_details keys if necessary, but prompt is quite specific
                # For now, we trust the LLM to follow the sub-key structure if "market_data" is a dict.
                parsed_output_data = parsed_json # Store the whole thing
                logger.info(f"[SectorMarketDataSubgraph] Successfully parsed market data for SECTOR {sector_name}.")
                current_subgraph_error = None
                routing_decision = "success"
            else:
                validation_error_msg = "Parsed JSON missing 'market_data' dictionary or 'source_urls_used' is not a list."
                logger.warning(f"[SectorMarketDataSubgraph] {validation_error_msg} Data: {parsed_json}")
                current_subgraph_error = current_subgraph_error or validation_error_msg
                if attempt < max_attempts_total: routing_decision = "retry"
        except (json.JSONDecodeError, ValueError) as e:
            json_parse_error_msg = f"Failed to parse/validate JSON for market data: {e}."
            current_subgraph_error = current_subgraph_error or json_parse_error_msg
            if attempt < max_attempts_total: routing_decision = "retry"
        except Exception as e:
            unexpected_parse_error = f"Unexpected error parsing market data output: {e}."
            current_subgraph_error = current_subgraph_error or unexpected_parse_error
            if attempt < max_attempts_total: routing_decision = "retry"

    if current_subgraph_error and routing_decision != "retry":
        routing_decision = "fail"
    
    updates: Dict[str, Any] = {
        "extracted_market_data": parsed_output_data, # Store the entire parsed JSON dict
        "subgraph_error": current_subgraph_error,
        "attempt": attempt + 1,
        "_route_decision": routing_decision
    }
    if routing_decision == "fail" and parsed_output_data is None:
        updates["extracted_market_data"] = {"market_data": {}, "source_urls_used": []} # Provide empty structure on hard fail
    return updates

# --- Graph Definition Function ---
def create_sector_market_data_graph():
    if not sector_market_data_react_runnable:
        logger.error("Cannot build SECTOR Market Data graph: ReAct runnable not available.")
        return None
    
    builder = StateGraph(SectorMarketDataSubgraphState)
    builder.add_node("prepare_market_data_message", start_market_data_collection_node)
    builder.add_node("sector_market_data_agent_executor", sector_market_data_react_runnable)
    builder.add_node("parse_market_data_result", parse_and_check_market_data_node)

    builder.set_entry_point("prepare_market_data_message")
    builder.add_edge("prepare_market_data_message", "sector_market_data_agent_executor")
    builder.add_edge("sector_market_data_agent_executor", "parse_market_data_result")

    builder.add_conditional_edges(
        "parse_market_data_result",
        lambda state: state.get("_route_decision", "fail"),
        {"success": END, "retry": "prepare_market_data_message", "fail": END}
    )
    try:
        graph = builder.compile()
        logger.info("SECTOR Market Data Analysis Subgraph compiled successfully.")
        return graph
    except Exception as e:
        logger.exception("Failed to compile SECTOR Market Data Analysis Subgraph.")
        return None

sector_market_data_subgraph_runnable = create_sector_market_data_graph()

# --- Main Block for Direct Subgraph Testing ---
if __name__ == '__main__':
    import pprint
    import time
    import os
    import sys
    from pathlib import Path
    from config import logger, config
    from core.state import SectorMarketDataSubgraphState

    if not config.GOOGLE_API_KEY: logger.error("CRITICAL: GOOGLE_API_KEY missing."); exit(1)

    test_sectors = ["Global EV Battery Market", "Quantum Computing Software"]
    if not sector_market_data_subgraph_runnable:
        logger.error("SECTOR Market Data Subgraph runnable not created. Exiting test."); exit(1)

    for sector_name_test in test_sectors:
        print(f"\n>>> Testing SECTOR Market Data Subgraph for: {sector_name_test} <<<\n")
        initial_state: SectorMarketDataSubgraphState = {
            "sector_name": sector_name_test,
            "messages": [],
            "attempt": 1,
            "max_attempts": MAX_MARKET_DATA_ATTEMPTS,
            "extracted_market_data": None,
            "subgraph_error": None,
        }
        final_state_dict = None
        start_time = time.time()
        try:
            logger.info(f"--- Invoking SECTOR Market Data Subgraph for {sector_name_test} ---")
            final_state_dict = sector_market_data_subgraph_runnable.invoke(
                initial_state, {"recursion_limit": 50} # Allow more steps for research
            )
            logger.info(f"--- SECTOR Market Data Subgraph Invocation Complete for {sector_name_test} ---")
        except Exception as e:
            logger.exception(f"--- SECTOR Market Data Subgraph Invocation FAILED for {sector_name_test} ---")
            final_state_dict = {**initial_state, "subgraph_error": f"Invoke failed: {e}"}
        
        end_time = time.time()
        logger.info(f"Execution time for {sector_name_test}: {end_time - start_time:.2f}s.")
        print("\nFinal State Snapshot:")
        pprint.pprint(final_state_dict)
        if len(test_sectors) > 1 and sector_name_test != test_sectors[-1]:
            logger.info("Pausing..."); time.sleep(15) # Longer pause for research tasks
            
    print("\n--- SECTOR Market Data Subgraph Test Script Finished ---")