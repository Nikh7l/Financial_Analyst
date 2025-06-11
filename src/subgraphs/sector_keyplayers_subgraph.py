# agents/sector_key_players_subgraph.py

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
from state import SectorKeyPlayerDetail 
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.prompts import ..., from src import config, from src.agents... import ...)
from prompts import SECTOR_KEY_PLAYERS_PROMPT_TEMPLATE_STATIC # Use the new detailed prompt
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.prompts import ..., from src import config, from src.agents... import ...)
from tools import google_search, search_duck_duck_go, get_web_page_content # Add get_page_content

# --- Langchain/LangGraph Imports ---
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage

# --- Constants ---
MAX_KEY_PLAYERS_ATTEMPTS = 2


class SectorKeyPlayersSubgraphState(TypedDict, total=False):
    sector_name: str # Input
    messages: List[BaseMessage] 
    attempt: int
    max_attempts: int
    # Output: The structure the ReAct agent will produce based on the prompt
    identified_key_players_data: SectorKeyPlayerDetail
    subgraph_error: Optional[str]
    _route_decision: Optional[str]

# --- LLM Initialization ---
try:
    # Can reuse LLM or initialize a new one for this specific task
    key_players_llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL_NAME, # Or a model good for research/extraction
        temperature=config.TEMPERATURE, # Or a slightly higher temp if descriptions need creativity
        convert_system_message_to_human=True
    )
    logger.info("LLM for SECTOR Key Players Agent initialized successfully.")
except Exception as e:
    logger.exception("Failed to initialize LLM for SECTOR Key Players Agent.")
    key_players_llm = None

# --- Tool List ---
sector_key_players_tools = [
    google_search,
    search_duck_duck_go,
    get_web_page_content, # Important for getting detailed descriptions from company pages
]

# --- Create the ReAct Agent Runnable ---
try:
    if not key_players_llm:
        raise RuntimeError("LLM for SECTOR Key Players Agent not initialized.")
    sector_key_players_react_runnable = create_react_agent(
        model=key_players_llm,
        tools=sector_key_players_tools,
        prompt=SECTOR_KEY_PLAYERS_PROMPT_TEMPLATE_STATIC,
        debug=True
    )
    logger.info("Core ReAct SECTOR Key Players Agent runnable created successfully.")
except RuntimeError as e:
    logger.error(f"Error creating sector key players ReAct agent: {str(e)}")
    sector_key_players_react_runnable = None
except Exception as e:
    logger.exception("Unexpected error creating ReAct SECTOR Key Players Agent runnable.")
    sector_key_players_react_runnable = None

# --- Node Logic Functions for Subgraph ---

def start_key_players_identification_node(state: SectorKeyPlayersSubgraphState) -> Dict[str, Any]:
    """Prepares the initial HumanMessage for the SECTOR Key Players ReAct agent."""
    sector_name = state["sector_name"]
    attempt = state.get("attempt", 1)
    logger.info(f"[SectorKeyPlayersSubgraph] Starting identification for SECTOR: {sector_name}, Attempt: {attempt}")

    if attempt == 1:
        human_query_content = (
            f"Identify the top key players in the **{sector_name} sector** and provide detailed descriptions "
            "as per the system prompt. Include names, detailed descriptions, and source URLs used."
        )
    else:
        prev_error = state.get("subgraph_error", "A previous attempt failed.")
        human_query_content = (
            f"Retry identifying key players in the **{sector_name} sector**. "
            f"Previous issue: {prev_error}. "
            "Ensure detailed descriptions and adherence to the JSON output format."
        )
    return {"messages": [HumanMessage(content=human_query_content)]}


def parse_and_check_key_players_node(state: SectorKeyPlayersSubgraphState) -> Dict[str, Any]:
    """Parses the final AI Message for SECTOR key players."""
    sector_name = state.get("sector_name", "Unknown Sector")
    attempt = state.get("attempt", 1)
    max_attempts_total = state.get("max_attempts", MAX_KEY_PLAYERS_ATTEMPTS)
    messages = state.get("messages", [])
    logger.info(f"[SectorKeyPlayersSubgraph] Parsing result for SECTOR: {sector_name} (Attempt {attempt})")

    parsed_output_data: Optional[Dict[str, Any]] = None
    current_subgraph_error: Optional[str] = state.get("subgraph_error")
    routing_decision: str = "fail"

    if not messages or not isinstance(messages[-1], AIMessage):
        parse_error_msg = f"ReAct agent ended with non-AI message: {type(messages[-1]) if messages else 'No Messages'}."
        current_subgraph_error = current_subgraph_error or parse_error_msg
        if attempt < max_attempts_total: routing_decision = "retry"
    else:
        ai_content = messages[-1].content.strip()
        logger.debug(f"[SectorKeyPlayersSubgraph] Final AI Message: {ai_content[:500]}...")
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", ai_content, re.DOTALL | re.IGNORECASE)
            if match: json_str = match.group(1)
            else:
                json_start = ai_content.find('{'); json_end = ai_content.rfind('}') + 1
                if json_start != -1 and json_end != -1 and json_end > json_start: json_str = ai_content[json_start:json_end]
                else: raise ValueError("JSON object not found in AI response.")
            
            parsed_json = json.loads(json_str)
            if not isinstance(parsed_json, dict):
                raise ValueError("Parsed content is not a JSON object.")

            # Validate based on SECTOR_KEY_PLAYERS_PROMPT output format
            key_players_list = parsed_json.get("key_players")
            
            if isinstance(key_players_list, list) and \
               all(isinstance(p, dict) and "name" in p and "description" in p for p in key_players_list):
                # The parsed_json IS the data structure we want to store.
                parsed_output_data = parsed_json # This dict has "key_players" and potentially "source_urls_used"
                logger.info(f"[SectorKeyPlayersSubgraph] Successfully parsed {len(key_players_list)} key players for SECTOR {sector_name}.")
                current_subgraph_error = None
                routing_decision = "success"
            else:
                validation_error_msg = "Parsed JSON 'key_players' is not a list of valid player objects."
                logger.warning(f"[SectorKeyPlayersSubgraph] {validation_error_msg} Data: {parsed_json}")
                current_subgraph_error = current_subgraph_error or validation_error_msg
                if attempt < max_attempts_total: routing_decision = "retry"
        except (json.JSONDecodeError, ValueError) as e:
            json_parse_error_msg = f"Failed to parse/validate JSON for key players: {e}."
            current_subgraph_error = current_subgraph_error or json_parse_error_msg
            if attempt < max_attempts_total: routing_decision = "retry"
        except Exception as e:
            unexpected_parse_error = f"Unexpected error parsing key players output: {e}."
            current_subgraph_error = current_subgraph_error or unexpected_parse_error
            if attempt < max_attempts_total: routing_decision = "retry"

    if current_subgraph_error and routing_decision != "retry":
        routing_decision = "fail"
    
    updates: Dict[str, Any] = {
        "identified_key_players_data": parsed_output_data, # Store the whole parsed dict
        "subgraph_error": current_subgraph_error,
        "attempt": attempt + 1,
        "_route_decision": routing_decision
    }
    if routing_decision == "fail" and parsed_output_data is None:
        updates["identified_key_players_data"] = None # Ensure it's None if failing
    return updates

# --- Graph Definition Function ---
def create_sector_key_players_graph():
    if not sector_key_players_react_runnable:
        logger.error("Cannot build SECTOR Key Players graph: ReAct runnable not available.")
        return None
    
    builder = StateGraph(SectorKeyPlayersSubgraphState)
    builder.add_node("prepare_key_players_message", start_key_players_identification_node)
    builder.add_node("sector_key_players_agent_executor", sector_key_players_react_runnable)
    builder.add_node("parse_key_players_result", parse_and_check_key_players_node)

    builder.set_entry_point("prepare_key_players_message")
    builder.add_edge("prepare_key_players_message", "sector_key_players_agent_executor")
    builder.add_edge("sector_key_players_agent_executor", "parse_key_players_result")

    builder.add_conditional_edges(
        "parse_key_players_result",
        lambda state: state.get("_route_decision", "fail"),
        {"success": END, "retry": "prepare_key_players_message", "fail": END}
    )
    try:
        graph = builder.compile()
        logger.info("SECTOR Key Players Analysis Subgraph compiled successfully.")
        return graph
    except Exception as e:
        logger.exception("Failed to compile SECTOR Key Players Analysis Subgraph.")
        return None

sector_key_players_subgraph_runnable = create_sector_key_players_graph()

# --- Main Block for Direct Subgraph Testing ---
if __name__ == '__main__':
    import pprint
    import time
    from dotenv import load_dotenv

    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("--- Running SECTOR Key Players Analysis Subgraph Directly ---")
    dotenv_path = os.path.join(project_root, '.env')
    if os.path.exists(dotenv_path): load_dotenv(dotenv_path); logger.info(f".env loaded.")
    else: logger.warning(f".env not found at {dotenv_path}.")

    if not config.GOOGLE_API_KEY: logger.error("CRITICAL: GOOGLE_API_KEY missing."); exit(1)
    # Add other API key checks if needed for search tools

    test_sectors = ["Cloud Computing Services", "Electric Vehicle Manufacturing"]
    if not sector_key_players_subgraph_runnable:
        logger.error("SECTOR Key Players Subgraph runnable not created. Exiting test."); exit(1)

    for sector_name_test in test_sectors:
        print(f"\n>>> Testing SECTOR Key Players Subgraph for: {sector_name_test} <<<\n")
        initial_state: SectorKeyPlayersSubgraphState = {
            "sector_name": sector_name_test,
            "messages": [],
            "attempt": 1,
            "max_attempts": MAX_KEY_PLAYERS_ATTEMPTS,
            "identified_key_players_data": None,
            "subgraph_error": None,
        }
        final_state_dict = None
        start_time = time.time()
        try:
            logger.info(f"--- Invoking SECTOR Key Players Subgraph for {sector_name_test} ---")
            final_state_dict = sector_key_players_subgraph_runnable.invoke(
                initial_state, {"recursion_limit": 40} # Allow enough steps for ReAct
            )
            logger.info(f"--- SECTOR Key Players Subgraph Invocation Complete for {sector_name_test} ---")
        except Exception as e:
            logger.exception(f"--- SECTOR Key Players Subgraph Invocation FAILED for {sector_name_test} ---")
            final_state_dict = {**initial_state, "subgraph_error": f"Invoke failed: {e}"}
        
        end_time = time.time()
        logger.info(f"Execution time for {sector_name_test}: {end_time - start_time:.2f}s.")
        print("\nFinal State Snapshot:")
        pprint.pprint(final_state_dict)
        if len(test_sectors) > 1 and sector_name_test != test_sectors[-1]:
            logger.info("Pausing..."); time.sleep(10) # Be mindful of API limits
            
    print("\n--- SECTOR Key Players Subgraph Test Script Finished ---")