import logging
import json
import re
from typing import List, Optional, Dict, Any, TypedDict, Literal

# --- Project Imports ---
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config
from config import logger
from prompts import SECTOR_NEWS_SENTIMENT_PROMPT_TEMPLATE_STATIC
from tools import get_news_articles, search_duck_duck_go, get_web_page_content, google_search

# --- Langchain/LangGraph Imports ---
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from state import SectorSentimentSubgraphState
# --- Constants ---
MAX_SECTOR_SENTIMENT_ATTEMPTS = 2 # Can be same or different from company attempts

# --- State Definition for this Subgraph (can be here or imported from state.py) ---


# --- LLM Initialization (Can share the same LLM instance if config is same) ---
try:
    sector_llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL_NAME, # 
        temperature= config.TEMPERATURE, #config.TEMPERATURE_SECTOR_ANALYSIS or
        convert_system_message_to_human=True
    )
    logger.info("LLM for SECTOR Sentiment Agent initialized successfully.")
except Exception as e:
    logger.exception("Failed to initialize LLM for SECTOR Sentiment Agent.")
    sector_llm = None

# --- Tool List (Can be same as company sentiment, or curated for sector analysis) ---
sector_sentiment_tools = [
    get_news_articles,
    google_search,
    search_duck_duck_go,
    get_web_page_content 
]

# --- Create the ReAct Agent Runnable ---
try:
    if not sector_llm:
        raise RuntimeError("LLM for SECTOR Sentiment Agent not initialized.")
    sector_sentiment_react_runnable = create_react_agent(
        model=sector_llm,
        tools=sector_sentiment_tools,
        prompt=SECTOR_NEWS_SENTIMENT_PROMPT_TEMPLATE_STATIC, # Use the SECTOR prompt
        debug=True
    )
    logger.info("Core ReAct SECTOR Sentiment Agent runnable created successfully.")
except RuntimeError as e:
    logger.error(f"Error creating sector ReAct agent: {str(e)}")
    sector_sentiment_react_runnable = None
except Exception as e:
    logger.exception("Unexpected error creating ReAct SECTOR Sentiment Agent runnable.")
    sector_sentiment_react_runnable = None

# --- Node Logic Functions for Subgraph ---

def start_sector_sentiment_node(state: SectorSentimentSubgraphState) -> Dict[str, Any]:
    """Prepares the initial HumanMessage for the SECTOR ReAct agent."""
    sector_name = state["sector_name"] # Use sector_name
    attempt = state.get("attempt", 1)
    logger.info(f"[SectorSentimentSubgraph] Starting analysis for SECTOR: {sector_name}, Attempt: {attempt}")

    if attempt == 1:
        human_query_content = (
            f"Determine the market sentiment for the **{sector_name} sector** and provide a JSON output "
            "with overall_sentiment, key_news_themes, recent_events, sentiment_reasoning, and source_urls_used. "
            "Follow instructions from your system prompt."
        )
    else:
        prev_error = state.get("subgraph_error", "A previous attempt failed.")
        human_query_content = (
            f"Retry determining market sentiment for the **{sector_name} sector**. "
            f"Previous issue: {prev_error}. "
            "Ensure you extract the sector name and provide the specified JSON output."
        )
    return {"messages": [HumanMessage(content=human_query_content)]}


def parse_and_check_sector_sentiment_node(state: SectorSentimentSubgraphState) -> Dict[str, Any]:
    """Parses the final AI Message for SECTOR sentiment."""
    sector_name = state.get("sector_name", "Unknown Sector")
    attempt = state.get("attempt", 1)
    max_attempts_total = state.get("max_attempts", MAX_SECTOR_SENTIMENT_ATTEMPTS)
    messages = state.get("messages", [])
    logger.info(f"[SectorSentimentSubgraph] Parsing result for SECTOR: {sector_name} (Attempt {attempt})")

    parsed_output_data: Optional[Dict[str, Any]] = None # Output key is different
    current_subgraph_error: Optional[str] = state.get("subgraph_error")
    routing_decision: str = "fail"

    if not messages or not isinstance(messages[-1], AIMessage):
        parse_error_msg = f"ReAct agent ended with a non-AI message: {type(messages[-1]) if messages else 'No Messages'}."
        current_subgraph_error = current_subgraph_error or parse_error_msg
        if attempt < max_attempts_total: routing_decision = "retry"
    else:
        ai_content = messages[-1].content.strip()
        logger.debug(f"[SectorSentimentSubgraph] Final AI Message to parse: {ai_content[:500]}...")
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

            # Validate keys specific to SECTOR_NEWS_SENTIMENT_PROMPT
            sentiment_val = parsed_json.get("overall_sentiment")
            themes_val = parsed_json.get("key_news_themes")
            events_val = parsed_json.get("recent_events")
            reasoning_val = parsed_json.get("sentiment_reasoning")
            # source_urls_val = parsed_json.get("source_urls_used") # Optional based on prompt

            if sentiment_val in ["Positive", "Negative", "Neutral"] and \
               isinstance(themes_val, list) and \
               isinstance(events_val, list) and \
               isinstance(reasoning_val, str) and reasoning_val.strip():
                # Construct the dictionary for 'sector_sentiment_analysis'
                parsed_output_data = {
                    "overall_sentiment": sentiment_val,
                    "key_news_themes": themes_val,
                    "recent_events": events_val,
                    "sentiment_reasoning": reasoning_val,
                    "source_urls_used": parsed_json.get("source_urls_used", []) # Default to empty if not found
                }
                logger.info(f"[SectorSentimentSubgraph] Successfully parsed sentiment for SECTOR {sector_name}: {sentiment_val}")
                current_subgraph_error = None
                routing_decision = "success"
            else:
                validation_error_msg = "Parsed JSON for sector sentiment has invalid or missing required fields."
                logger.warning(f"[SectorSentimentSubgraph] {validation_error_msg} Data: {parsed_json}")
                current_subgraph_error = current_subgraph_error or validation_error_msg
                if attempt < max_attempts_total: routing_decision = "retry"
        except (json.JSONDecodeError, ValueError) as e:
            json_parse_error_msg = f"Failed to parse JSON or validate sector sentiment content: {e}."
            current_subgraph_error = current_subgraph_error or json_parse_error_msg
            if attempt < max_attempts_total: routing_decision = "retry"
        except Exception as e:
            unexpected_parse_error = f"Unexpected error parsing sector sentiment output: {e}."
            current_subgraph_error = current_subgraph_error or unexpected_parse_error
            if attempt < max_attempts_total: routing_decision = "retry"

    if current_subgraph_error and routing_decision != "retry":
        routing_decision = "fail"
    
    updates: Dict[str, Any] = {
        "sector_sentiment_analysis": parsed_output_data, # Update this key
        "subgraph_error": current_subgraph_error,
        "attempt": attempt + 1,
        "_route_decision": routing_decision
    }
    if routing_decision == "fail" and parsed_output_data is None:
        updates["sector_sentiment_analysis"] = None
    return updates

# --- Graph Definition Function ---
def create_sector_sentiment_analysis_graph():
    if not sector_sentiment_react_runnable:
        logger.error("Cannot build SECTOR sentiment analysis graph: ReAct runnable not available.")
        return None
    
    builder = StateGraph(SectorSentimentSubgraphState)
    builder.add_node("prepare_sector_initial_message", start_sector_sentiment_node)
    builder.add_node("sector_sentiment_react_executor", sector_sentiment_react_runnable)
    builder.add_node("parse_sector_sentiment_result", parse_and_check_sector_sentiment_node)

    builder.set_entry_point("prepare_sector_initial_message")
    builder.add_edge("prepare_sector_initial_message", "sector_sentiment_react_executor")
    builder.add_edge("sector_sentiment_react_executor", "parse_sector_sentiment_result")

    builder.add_conditional_edges(
        "parse_sector_sentiment_result",
        lambda state: state.get("_route_decision", "fail"),
        {"success": END, "retry": "prepare_sector_initial_message", "fail": END}
    )
    try:
        graph = builder.compile()
        logger.info("SECTOR Sentiment Analysis Subgraph compiled successfully.")
        return graph
    except Exception as e:
        logger.exception("Failed to compile SECTOR Sentiment Analysis Subgraph.")
        return None

sector_sentiment_analysis_subgraph_runnable = create_sector_sentiment_analysis_graph()

# --- Main Block for Direct Subgraph Testing ---
if __name__ == '__main__':
    import pprint
    import time
    from dotenv import load_dotenv

    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("--- Running SECTOR Sentiment Analysis Subgraph Directly ---")
    dotenv_path = os.path.join(project_root, '.env')
    if os.path.exists(dotenv_path): load_dotenv(dotenv_path); logger.info(f".env loaded.")
    else: logger.warning(f".env not found at {dotenv_path}.")

    if not config.GOOGLE_API_KEY: logger.error("CRITICAL: GOOGLE_API_KEY missing."); exit(1)
    if not config.NEWS_API_KEY: logger.warning("NEWS_API_KEY missing. News tool may fail.")

    test_sectors = ["Renewable Energy", "Artificial Intelligence Hardware"]
    if not sector_sentiment_analysis_subgraph_runnable:
        logger.error("SECTOR Subgraph runnable not created. Exiting test."); exit(1)

    for sector in test_sectors:
        print(f"\n>>> Testing SECTOR Sentiment Subgraph for: {sector} <<<\n")
        initial_state: SectorSentimentSubgraphState = {
            "sector_name": sector,
            "messages": [],
            "attempt": 1,
            "max_attempts": MAX_SECTOR_SENTIMENT_ATTEMPTS,
            "sector_sentiment_analysis": None,
            "subgraph_error": None,
        }
        final_state_dict = None
        start_time = time.time()
        try:
            logger.info(f"--- Invoking SECTOR Sentiment Subgraph for {sector} ---")
            final_state_dict = sector_sentiment_analysis_subgraph_runnable.invoke(
                initial_state, {"recursion_limit": 40}
            )
            logger.info(f"--- SECTOR Subgraph Invocation Complete for {sector} ---")
        except Exception as e:
            logger.exception(f"--- SECTOR Subgraph Invocation FAILED for {sector} ---")
            final_state_dict = {**initial_state, "subgraph_error": f"Invoke failed: {e}"}
        
        end_time = time.time()
        logger.info(f"Execution time for {sector}: {end_time - start_time:.2f}s.")
        print("\nFinal State Snapshot:")
        pprint.pprint(final_state_dict)
        if len(test_sectors) > 1 and sector != test_sectors[-1]:
            logger.info("Pausing..."); time.sleep(10)
            
    print("\n--- SECTOR Sentiment Subgraph Test Script Finished ---")