# agents/sentiment_agent_graph.py
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
# Use the new static prompt template
from prompts import SENTIMENT_WORKER_PROMPT_TEMPLATE_STATIC
from tools import get_news_articles, search_duck_duck_go, get_web_page_content, google_search

# --- Langchain/LangGraph Imports ---
from langgraph.graph import StateGraph, START, END # Use START for clarity
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage # Added ToolMessage

# --- Constants ---
MAX_SENTIMENT_ATTEMPTS = 2

# --- State Definition for this Subgraph ---
class SentimentSubgraphState(TypedDict, total=False):
    company_name: str 

    # ReAct agent messages (includes initial human query with company name)
    messages: List[BaseMessage]

    # Control flow / Retries
    attempt: int
    max_attempts: int

    # Output Data
    sentiment_analysis: Optional[Dict[str, str]]  # {"sentiment": "Positive", "detailed_sentiment_report": "..."}
    subgraph_error: Optional[str]
    _route_decision: Optional[str]

# --- LLM Initialization ---
try:
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL_NAME,
        temperature=config.TEMPERATURE,
        convert_system_message_to_human=True 
    )
    logger.info("LLM for Sentiment Agent initialized successfully.")
except Exception as e:
    logger.exception("Failed to initialize LLM for Sentiment Agent.")
    llm = None

# --- Tool List ---
sentiment_analysis_tools = [
    get_news_articles,
    google_search,
    search_duck_duck_go,
    get_web_page_content,
]

# --- Create the ReAct Agent Runnable (using static system prompt) ---
try:
    if not llm:
        raise RuntimeError("LLM for Sentiment Agent not initialized.")
    # Pass the static system prompt string directly
    sentiment_react_runnable = create_react_agent(
        model=llm,
        tools=sentiment_analysis_tools,
        prompt=SENTIMENT_WORKER_PROMPT_TEMPLATE_STATIC, # Static system prompt string
        debug=True
    )
    logger.info("Core ReAct Sentiment Agent runnable created successfully with static prompt.")
except RuntimeError as e:
    logger.error(str(e))
    sentiment_react_runnable = None
except Exception as e:
    logger.exception("Unexpected error creating ReAct Sentiment Agent runnable.")
    sentiment_react_runnable = None


# --- Node Logic Functions for Subgraph ---

def start_sentiment_analysis_node(state: SentimentSubgraphState) -> Dict[str, Any]:
    """
    Prepares the initial HumanMessage for the ReAct agent.
    The ReAct agent will use its static system prompt and this human message.
    """
    company_name = state["company_name"]
    attempt = state.get("attempt", 1)
    logger.info(f"[Sentiment Subgraph] Starting analysis for {company_name}, Attempt: {attempt}")

    if attempt == 1:
        # Embed company name and task directly in the human message
        human_query_content = (
            f"Determine the market sentiment for {company_name} and produce a detailed report. "
            "Follow the instructions from your system prompt regarding process and output format."
        )
    else:
        prev_error = state.get("subgraph_error", "A previous attempt failed.")
        human_query_content = (
            f"Retry determining market sentiment for {company_name} and producing a detailed report. "
            f"Previous issue: {prev_error}. "
            "Please ensure you extract the company name correctly from this message and follow all instructions."
        )
        
    return {"messages": [HumanMessage(content=human_query_content)]}


def parse_and_check_sentiment_node(state: SentimentSubgraphState) -> Dict[str, Any]:
    """
    Parses the final AI Message from the ReAct agent, validates the sentiment data,
    and sets the routing decision for the subgraph.
    """
    company_name = state.get("company_name", "Unknown Company") # Fallback for logging
    attempt = state.get("attempt", 1) # Should be correctly managed by the graph flow
    max_attempts_total = state.get("max_attempts", MAX_SENTIMENT_ATTEMPTS)
    messages = state.get("messages", [])

    logger.info(f"[Sentiment Subgraph] Parsing result for {company_name} (Attempt {attempt})")

    sentiment_data_output: Optional[Dict[str, str]] = None
    # Preserve any error that might have occurred if ReAct agent itself failed before producing a message
    current_subgraph_error: Optional[str] = state.get("subgraph_error")
    routing_decision: str = "fail"

    if not messages: # Should not happen if start_node ran
        parse_error_msg = "No messages found in state to parse."
        logger.error(f"[Sentiment Subgraph] {parse_error_msg}")
        current_subgraph_error = current_subgraph_error or parse_error_msg
    elif not isinstance(messages[-1], AIMessage):
        # This can happen if the agent ends on a tool call or other non-AI message
        parse_error_msg = f"ReAct agent ended with a non-AI message: {type(messages[-1])}."
        if isinstance(messages[-1], ToolMessage):
            parse_error_msg += f" Content: {messages[-1].content[:100]}..."
        logger.warning(f"[Sentiment Subgraph] {parse_error_msg}")
        current_subgraph_error = current_subgraph_error or parse_error_msg
        if attempt < max_attempts_total: # Allow retry if agent ended unexpectedly
            routing_decision = "retry"
    else: # Last message is AIMessage
        ai_content = messages[-1].content.strip()
        logger.debug(f"[Sentiment Subgraph] Final AI Message to parse: {ai_content[:500]}...")
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", ai_content, re.DOTALL | re.IGNORECASE)
            if match: json_str = match.group(1)
            else:
                json_start = ai_content.find('{'); json_end = ai_content.rfind('}') + 1
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_str = ai_content[json_start:json_end]
                else: raise ValueError("JSON object not found in AI response.")
            parsed_json = json.loads(json_str)
            if not isinstance(parsed_json, dict):
                raise ValueError("Parsed content is not a JSON object (dictionary).")

            sentiment_value = parsed_json.get("sentiment")
            report_value = parsed_json.get("detailed_sentiment_report")

            if sentiment_value in ["Positive", "Negative", "Neutral"] and \
               isinstance(report_value, str) and report_value.strip():
                sentiment_data_output = {"sentiment": sentiment_value, "detailed_sentiment_report": report_value}
                logger.info(f"[Sentiment Subgraph] Successfully parsed sentiment: {sentiment_value} for {company_name}")
                current_subgraph_error = None
                routing_decision = "success"
            else:
                validation_error_msg = "Parsed JSON has invalid sentiment or missing/empty detailed_sentiment_report."
                logger.warning(f"[Sentiment Subgraph] {validation_error_msg} Data: {parsed_json}")
                current_subgraph_error = current_subgraph_error or validation_error_msg
                if attempt < max_attempts_total: routing_decision = "retry"
        except (json.JSONDecodeError, ValueError) as e:
            json_parse_error_msg = f"Failed to parse JSON or validate content: {e}."
            logger.warning(f"[Sentiment Subgraph] {json_parse_error_msg} AI Content: {ai_content[:200]}")
            current_subgraph_error = current_subgraph_error or json_parse_error_msg
            if attempt < max_attempts_total: routing_decision = "retry"
        except Exception as e:
            unexpected_parse_error = f"Unexpected error parsing sentiment output: {e}."
            logger.error(f"[Sentiment Subgraph] {unexpected_parse_error} AI Content: {ai_content[:200]}", exc_info=True)
            current_subgraph_error = current_subgraph_error or unexpected_parse_error
            if attempt < max_attempts_total: routing_decision = "retry"

    if current_subgraph_error and routing_decision != "retry":
        routing_decision = "fail"
    logger.info(f"[Sentiment Subgraph] Check complete. Routing: {routing_decision}. Error: {current_subgraph_error}")
    
    # Prepare updates for the state
    updates: Dict[str, Any] = {
        "sentiment_analysis": sentiment_data_output,
        "subgraph_error": current_subgraph_error,
        "attempt": attempt + 1, # Increment attempt number for the next potential cycle
        "_route_decision": routing_decision
    }
    # If parsing failed but we are not retrying, ensure sentiment_analysis is None explicitly
    if routing_decision == "fail" and sentiment_data_output is None :
        updates["sentiment_analysis"] = None # or some error placeholder if preferred by main graph

    return updates


# --- Graph Definition Function ---
def create_sentiment_analysis_graph():
    if not sentiment_react_runnable:
        logger.error("Cannot build sentiment analysis graph: ReAct runnable is not available.")
        return None
    builder = StateGraph(SentimentSubgraphState)

    # Node 1: Prepare initial message (entry point for each attempt)
    builder.add_node("prepare_initial_message", start_sentiment_analysis_node)

    # Node 2: The ReAct Agent itself (compiled graph)
    # The input to this node is the full SentimentSubgraphState.
    # create_react_agent expects a dictionary with a "messages" key.
    # It will use the 'prompt' it was created with as the system message.
    builder.add_node("sentiment_react_agent_executor", sentiment_react_runnable)

    # Node 3: Parse the output
    builder.add_node("parse_sentiment_result", parse_and_check_sentiment_node)

    # Define Edges
    builder.set_entry_point("prepare_initial_message")
    builder.add_edge("prepare_initial_message", "sentiment_react_agent_executor")
    # The output of sentiment_react_runnable will update the 'messages' in the state.
    builder.add_edge("sentiment_react_agent_executor", "parse_sentiment_result")

    builder.add_conditional_edges(
        "parse_sentiment_result",
        lambda state: state.get("_route_decision", "fail"),
        {
            "success": END,
            "retry": "prepare_initial_message", # Go back to prepare for another attempt
            "fail": END
        }
    )
    try:
        graph = builder.compile()
        logger.info("Sentiment Analysis Subgraph compiled successfully (Static Prompt Method).")
        return graph
    except Exception as e:
        logger.exception("Failed to compile Sentiment Analysis Subgraph (Static Prompt Method).")
        return None

sentiment_analysis_subgraph_runnable = create_sentiment_analysis_graph()

# --- Main Block for Direct Subgraph Testing ---
if __name__ == '__main__':
    import pprint
    import time
    from dotenv import load_dotenv

    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Running Sentiment Analysis Worker Subgraph Directly (Static Prompt Method) ---")
    dotenv_path = os.path.join(project_root, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path); logger.info(f".env file loaded from {dotenv_path}")
    else:
        logger.warning(f".env file not found at {dotenv_path}.")

    if not config.GOOGLE_API_KEY: logger.error("CRITICAL: GOOGLE_API_KEY not found."); exit(1)
    if not config.NEWS_API_KEY: logger.warning("NEWS_API_KEY not found. 'get_news_articles' tool may fail.")

    test_companies = ["Nvidia", "Microsoft"]
    if not sentiment_analysis_subgraph_runnable:
        logger.error("Subgraph runnable could not be created. Exiting test."); exit(1)

    all_final_states = {}
    for company_val in test_companies:
        print(f"\n>>> Testing Sentiment Analysis Subgraph for: {company_val} <<<\n")
        initial_subgraph_state: SentimentSubgraphState = {
            "company_name": company_val, # For context and initial message creation
            "messages": [],             # `prepare_initial_message` will populate this for the agent
            "attempt": 1,
            "max_attempts": MAX_SENTIMENT_ATTEMPTS,
            "sentiment_analysis": None,
            "subgraph_error": None,
        }
        final_test_state = None
        start_time = time.time()
        try:
            logger.info(f"--- Invoking Sentiment Subgraph for {company_val} ---")
            # Stream to observe states (optional, invoke is simpler for final state)
            # for i, s_state in enumerate(sentiment_analysis_subgraph_runnable.stream(initial_subgraph_state, {"recursion_limit": 40})):
            #     node_name = list(s_state.keys())[0]
            #     logger.info(f"Stream step {i}, node {node_name}, state update: {s_state[node_name].get('messages', s_state[node_name])}")
            #     final_test_state = s_state[node_name] # Keep last state if streaming

            final_test_state_dict = sentiment_analysis_subgraph_runnable.invoke(
                initial_subgraph_state,
                {"recursion_limit": 40} # Overall graph recursion limit
            )
            final_test_state = final_test_state_dict # invoke returns the full final state
            logger.info(f"--- Subgraph Invocation Complete for {company_val} ---")

        except Exception as e:
            logger.exception(f"--- Subgraph Invocation FAILED for {company_val} ---")
            final_test_state = {
                **initial_subgraph_state, # Keep some initial context
                "subgraph_error": f"Subgraph execution failed at invoke level: {e}"
            }
        end_time = time.time()
        execution_duration = end_time - start_time
        logger.info(f"Execution time for {company_val}: {execution_duration:.2f} seconds.")
        print("\nFinal State Snapshot for Test Run:")
        logger.info(f"Final State for {company_val}:{final_test_state}")
        pprint.pprint(final_test_state) # This is the full final state dictionary
        all_final_states[company_val] = final_test_state
        if len(test_companies) > 1 and company_val != test_companies[-1]:
            logger.info("Pausing before next test..."); time.sleep(15) # Longer pause
    print("\n--- Sentiment Analysis Subgraph Test Script Finished ---")