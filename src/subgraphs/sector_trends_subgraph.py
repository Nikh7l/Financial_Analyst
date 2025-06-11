import logging
import json
import re
from typing import List, Optional, Dict, Any, TypedDict, Literal

# --- Project Imports ---
# TODO: Remove sys.path manipulation block below.
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.prompts import ..., from src import config, from src.agents... import ...)
import config
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.prompts import ..., from src import config, from src.agents... import ...)
from config import logger
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.prompts import ..., from src import config, from src.agents... import ...)
from prompts import SECTOR_TRENDS_PROMPT_TEMPLATE_STATIC
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.prompts import ..., from src import config, from src.agents... import ...)
from tools import google_search, search_duck_duck_go, get_news_articles, get_web_page_content

# --- Langchain/LangGraph Imports ---
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage

# --- Constants ---
MAX_TRENDS_ATTEMPTS = 2

# --- State Definition for this Subgraph ---
class SectorTrendsSubgraphState(TypedDict, total=False):
    sector_name: str # Input
    messages: List[BaseMessage] # For ReAct agent
    attempt: int
    max_attempts: int
    # Output: The structure the ReAct agent will produce
    extracted_trends_data: Optional[Dict[str, Any]] # Will hold {"trends_data": {...}, "source_urls_used": [...]}
    subgraph_error: Optional[str]
    _route_decision: Optional[str]

# --- LLM Initialization ---
try:
    trends_llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL_NAME, # Or a model good for synthesis/research
        temperature=0.2, # Slightly higher temp might help with identifying nuanced trends
        convert_system_message_to_human=True
    )
    logger.info("LLM for SECTOR Trends Agent initialized successfully.")
except Exception as e:
    logger.exception("Failed to initialize LLM for SECTOR Trends Agent.")
    trends_llm = None

# --- Tool List ---
sector_trends_tools = [
    google_search,
    search_duck_duck_go,
    get_news_articles,
    get_web_page_content, # Likely important for this task
]

# --- Create the ReAct Agent Runnable ---
try:
    if not trends_llm:
        raise RuntimeError("LLM for SECTOR Trends Agent not initialized.")
    sector_trends_react_runnable = create_react_agent(
        model=trends_llm,
        tools=sector_trends_tools,
        prompt=SECTOR_TRENDS_PROMPT_TEMPLATE_STATIC,
        debug=True
    )
    logger.info("Core ReAct SECTOR Trends Agent runnable created successfully.")
except RuntimeError as e:
    logger.error(f"Error creating sector trends ReAct agent: {str(e)}")
    sector_trends_react_runnable = None
except Exception as e:
    logger.exception("Unexpected error creating ReAct SECTOR Trends Agent runnable.")
    sector_trends_react_runnable = None

# --- Node Logic Functions for Subgraph ---

def start_trends_analysis_node(state: SectorTrendsSubgraphState) -> Dict[str, Any]:
    sector_name = state["sector_name"]
    attempt = state.get("attempt", 1)
    logger.info(f"[SectorTrendsSubgraph] Starting trends analysis for SECTOR: {sector_name}, Attempt: {attempt}")

    if attempt == 1:
        human_query_content = (
            f"Research and identify key trends, recent innovations, challenges, and emerging opportunities "
            f"for the **{sector_name} sector**. Follow the JSON output format specified in your system prompt."
        )
    else:
        prev_error = state.get("subgraph_error", "A previous attempt failed.")
        human_query_content = (
            f"Retry researching trends, innovations, challenges, and opportunities for the **{sector_name} sector**. "
            f"Previous issue: {prev_error}. "
            "Please ensure comprehensive research and accurate JSON formatting."
        )
    return {"messages": [HumanMessage(content=human_query_content)]}


def parse_and_check_trends_node(state: SectorTrendsSubgraphState) -> Dict[str, Any]:
    sector_name = state.get("sector_name", "Unknown Sector")
    attempt = state.get("attempt", 1)
    max_attempts_total = state.get("max_attempts", MAX_TRENDS_ATTEMPTS)
    messages = state.get("messages", [])
    logger.info(f"[SectorTrendsSubgraph] Parsing result for SECTOR: {sector_name} (Attempt {attempt})")

    parsed_output_data: Optional[Dict[str, Any]] = None
    current_subgraph_error: Optional[str] = state.get("subgraph_error")
    routing_decision: str = "fail"

    if not messages or not isinstance(messages[-1], AIMessage):
        parse_error_msg = f"ReAct agent ended with non-AI message: {type(messages[-1]) if messages else 'No Messages'}."
        current_subgraph_error = current_subgraph_error or parse_error_msg
        if attempt < max_attempts_total: routing_decision = "retry"
    else:
        ai_content = messages[-1].content.strip()
        logger.debug(f"[SectorTrendsSubgraph] Final AI Message: {ai_content[:500]}...")
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", ai_content, re.DOTALL | re.IGNORECASE)
            if match: json_str = match.group(1)
            else:
                json_start = ai_content.find('{'); json_end = ai_content.rfind('}') + 1
                if json_start != -1 and json_end != -1 and json_end > json_start: json_str = ai_content[json_start:json_end]
                else: raise ValueError("JSON object not found in AI response.")
            
            parsed_json = json.loads(json_str) # This is like {"trends_data": {...}, "source_urls_used": []}
            if not isinstance(parsed_json, dict):
                raise ValueError("Parsed content is not a JSON object.")

            trends_data_details = parsed_json.get("trends_data")
            source_urls = parsed_json.get("source_urls_used", [])

            # Validate trends_data_details structure
            if isinstance(trends_data_details, dict) and isinstance(source_urls, list):
                # Check for presence of the list keys inside trends_data
                required_list_keys = ["major_trends", "recent_innovations", "key_challenges", "emerging_opportunities"]
                all_keys_valid = True
                for key in required_list_keys:
                    if key in trends_data_details and not isinstance(trends_data_details[key], list):
                        logger.warning(f"[SectorTrendsSubgraph] Key '{key}' in 'trends_data' is not a list. Data: {trends_data_details[key]}")
                        trends_data_details[key] = [] # Default to empty list if not a list
                    elif key not in trends_data_details:
                         trends_data_details[key] = [] # Add empty list if key is missing

                parsed_output_data = parsed_json # Store the (potentially corrected) parsed_json
                logger.info(f"[SectorTrendsSubgraph] Successfully parsed trends data for SECTOR {sector_name}.")
                current_subgraph_error = None
                routing_decision = "success"
            else:
                validation_error_msg = "Parsed JSON missing 'trends_data' dictionary or 'source_urls_used' is not a list."
                logger.warning(f"[SectorTrendsSubgraph] {validation_error_msg} Data: {parsed_json}")
                current_subgraph_error = current_subgraph_error or validation_error_msg
                if attempt < max_attempts_total: routing_decision = "retry"
        except (json.JSONDecodeError, ValueError) as e:
            json_parse_error_msg = f"Failed to parse/validate JSON for trends data: {e}."
            current_subgraph_error = current_subgraph_error or json_parse_error_msg
            if attempt < max_attempts_total: routing_decision = "retry"
        except Exception as e:
            unexpected_parse_error = f"Unexpected error parsing trends data output: {e}."
            current_subgraph_error = current_subgraph_error or unexpected_parse_error
            if attempt < max_attempts_total: routing_decision = "retry"

    if current_subgraph_error and routing_decision != "retry":
        routing_decision = "fail"
    
    updates: Dict[str, Any] = {
        "extracted_trends_data": parsed_output_data,
        "subgraph_error": current_subgraph_error,
        "attempt": attempt + 1,
        "_route_decision": routing_decision
    }
    if routing_decision == "fail" and parsed_output_data is None:
        # Provide an empty structure on hard fail for consistency
        updates["extracted_trends_data"] = {
            "trends_data": {
                "major_trends": [], "recent_innovations": [],
                "key_challenges": [], "emerging_opportunities": []
            },
            "source_urls_used": []
        }
    return updates

# --- Graph Definition Function ---
def create_sector_trends_graph():
    if not sector_trends_react_runnable:
        logger.error("Cannot build SECTOR Trends graph: ReAct runnable not available.")
        return None
    
    builder = StateGraph(SectorTrendsSubgraphState)
    builder.add_node("prepare_trends_message", start_trends_analysis_node)
    builder.add_node("sector_trends_agent_executor", sector_trends_react_runnable)
    builder.add_node("parse_trends_result", parse_and_check_trends_node)

    builder.set_entry_point("prepare_trends_message")
    builder.add_edge("prepare_trends_message", "sector_trends_agent_executor")
    builder.add_edge("sector_trends_agent_executor", "parse_trends_result")

    builder.add_conditional_edges(
        "parse_trends_result",
        lambda state: state.get("_route_decision", "fail"),
        {"success": END, "retry": "prepare_trends_message", "fail": END}
    )
    try:
        graph = builder.compile()
        logger.info("SECTOR Trends Analysis Subgraph compiled successfully.")
        return graph
    except Exception as e:
        logger.exception("Failed to compile SECTOR Trends Analysis Subgraph.")
        return None

sector_trends_subgraph_runnable = create_sector_trends_graph()

# --- Main Block for Direct Subgraph Testing ---
if __name__ == '__main__':
    import pprint
    import time
    from dotenv import load_dotenv

    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("--- Running SECTOR Trends Analysis Subgraph Directly ---")
    dotenv_path = os.path.join(project_root, '.env')
    if os.path.exists(dotenv_path): load_dotenv(dotenv_path); logger.info(f".env loaded.")
    else: logger.warning(f".env not found at {dotenv_path}.")

    if not config.GOOGLE_API_KEY: logger.error("CRITICAL: GOOGLE_API_KEY missing."); exit(1)

    test_sectors = ["Artificial Intelligence in Healthcare", "Sustainable Packaging Solutions"]
    if not sector_trends_subgraph_runnable:
        logger.error("SECTOR Trends Subgraph runnable not created. Exiting test."); exit(1)

    for sector_name_test in test_sectors:
        print(f"\n>>> Testing SECTOR Trends Subgraph for: {sector_name_test} <<<\n")
        initial_state: SectorTrendsSubgraphState = {
            "sector_name": sector_name_test,
            "messages": [],
            "attempt": 1,
            "max_attempts": MAX_TRENDS_ATTEMPTS,
            "extracted_trends_data": None,
            "subgraph_error": None,
        }
        final_state_dict = None
        start_time = time.time()
        try:
            logger.info(f"--- Invoking SECTOR Trends Subgraph for {sector_name_test} ---")
            final_state_dict = sector_trends_subgraph_runnable.invoke(
                initial_state, {"recursion_limit": 60} # Allow more steps for comprehensive research
            )
            logger.info(f"--- SECTOR Trends Subgraph Invocation Complete for {sector_name_test} ---")
        except Exception as e:
            logger.exception(f"--- SECTOR Trends Subgraph Invocation FAILED for {sector_name_test} ---")
            final_state_dict = {**initial_state, "subgraph_error": f"Invoke failed: {e}"}
        
        end_time = time.time()
        logger.info(f"Execution time for {sector_name_test}: {end_time - start_time:.2f}s.")
        print("\nFinal State Snapshot:")
        pprint.pprint(final_state_dict)
        if len(test_sectors) > 1 and sector_name_test != test_sectors[-1]:
            logger.info("Pausing..."); time.sleep(15)
            
    print("\n--- SECTOR Trends Subgraph Test Script Finished ---")