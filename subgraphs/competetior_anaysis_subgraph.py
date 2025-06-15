import json
import re
from typing import Dict, Any, List, Optional, TypedDict, Callable # Added Callable

# Langchain/LangGraph components
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

# Project components
from core.prompts import COMPETITOR_WORKER_PROMPT
from core.tools import google_search, search_duck_duck_go, get_web_page_content
from config import config

logger = config.logger

# --- State Definition ---
class CompetitorInfo(TypedDict):
    name: str
    description: Optional[str]

class CompetitorSubgraphState(TypedDict, total=False):
    company_name: str
    attempt: int
    max_attempts: int
    messages: List[BaseMessage]
    competitors: Optional[List[CompetitorInfo]] # List of dicts
    subgraph_error: Optional[str]
    _route_decision: Optional[str]

# --- List of Basic Tools for the ReAct Node ---
competitor_agent_tools = [
    google_search,
    search_duck_duck_go,
    get_web_page_content,
]

# --- Initialize LLM ---
try:
    worker_llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL_NAME,
        temperature=config.TEMPERATURE, # Allow some creativity for descriptions
        convert_system_message_to_human=True
    )
except Exception as e:
    logger.exception("Failed to initialize LLM for Competitor Worker.")
    worker_llm = None

# --- ReAct Agent Creation ---
def create_competitor_react_agent():
    if not worker_llm:
        raise RuntimeError("LLM for Competitor Worker not initialized.")
    # Prompt expects company_name, pass via initial message
    return create_react_agent(
            model=worker_llm,
            tools=competitor_agent_tools,
            prompt=COMPETITOR_WORKER_PROMPT # Pass base prompt
        )

competitor_react_runnable = create_competitor_react_agent()

# --- Node Logic Functions for Subgraph ---

def react_competitor_agent_node(state: CompetitorSubgraphState) -> Dict[str, Any]:
    """Runs the ReAct agent to attempt finding competitors."""
    company_name = state["company_name"]
    attempt = state["attempt"]
    max_attempts = state["max_attempts"]
    logger.info(f"[CompetitorWorker Subgraph] Attempt {attempt}/{max_attempts}: Running ReAct agent for {company_name}")

    if attempt == 1:
        human_message = f"Identify the top 3-5 primary competitors for {company_name}, providing a brief description for each."
    else:
        prev_error = state.get("subgraph_error", "Previous attempt failed or yielded insufficient results.")
        human_message = f"Retry identifying competitors for {company_name}. {prev_error}. Ensure you are using search effectively and extracting names and descriptions."

    react_input = {"messages": [HumanMessage(content=human_message)]}
    try:
        react_output_state = competitor_react_runnable.invoke(react_input, {"recursion_limit": 15})
        return {"messages": react_output_state.get("messages", [])}
    except Exception as e:
        logger.error(f"[CompetitorWorker Subgraph] Exception during ReAct execution for {company_name}: {e}", exc_info=True)
        return {"subgraph_error": f"ReAct agent execution failed: {e}", "messages": state.get("messages", []) + [AIMessage(content=f"Error: {e}")]}


def check_and_parse_competitor_result_node(state: CompetitorSubgraphState) -> Dict[str, Any]:
    """Checks the ReAct agent output, parses competitor list, decides next step."""
    company_name = state["company_name"]
    attempt = state.get("attempt", 1)
    max_attempts = state.get("max_attempts", 1)
    messages = state.get("messages", [])
    logger.info(f"[CompetitorWorker Subgraph] Checking result for {company_name} (Attempt {attempt})")

    competitors_list: Optional[List[CompetitorInfo]] = None
    parse_error: Optional[str] = None
    next_step: str = "fail"

    if not messages:
        parse_error = "No messages found from ReAct agent."
    else:
        final_message = messages[-1]
        if final_message.type == "ai" and isinstance(final_message.content, str):
            response_text = final_message.content.strip()
            logger.debug(f"Final AI Message to parse: {response_text}")
            try:
                match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
                if match: json_str = match.group(1)
                else:
                    json_start = response_text.find('{'); json_end = response_text.rfind('}') + 1
                    if json_start != -1 and json_end != -1: json_str = response_text[json_start:json_end]
                    else: raise ValueError("JSON object not found")

                parsed_json = json.loads(json_str)

                # Check for 'competitors' list or 'error'
                if "competitors" in parsed_json and isinstance(parsed_json["competitors"], list):
                    raw_list = parsed_json["competitors"]
                    # Basic validation of list structure
                    if all(isinstance(item, dict) and "name" in item for item in raw_list):
                         # Optionally cast to CompetitorInfo for stronger typing, though dict is fine for state
                         competitors_list = [{"name": item.get("name"), "description": item.get("description")} for item in raw_list]
                         logger.info(f"Successfully parsed {len(competitors_list)} competitors for {company_name}.")
                         parse_error = None
                         next_step = "success"
                    else:
                        parse_error = "Parsed 'competitors' list has invalid item structure."
                        if attempt < max_attempts: next_step = "retry"
                        else: next_step = "fail"

                elif "error" in parsed_json and isinstance(parsed_json["error"], str):
                     parse_error = f"ReAct agent reported error: {parsed_json['error']}"
                     if attempt < max_attempts: next_step = "retry"
                     else: next_step = "fail"
                else:
                    parse_error = "LLM JSON response missing 'competitors' list or 'error' string."
                    if attempt < max_attempts: next_step = "retry"
                    else: next_step = "fail"
            except (json.JSONDecodeError, ValueError) as e:
                parse_error = f"Failed to parse JSON from LLM response: {e}"
                if attempt < max_attempts: next_step = "retry"
                else: next_step = "fail"
        else:
            parse_error = "ReAct agent did not end with expected AI response."
            next_step = "fail"

    logger.info(f"[CompetitorWorker Subgraph] Check complete. Routing decision: {next_step}. Error: {parse_error}")
    return {
        "competitors": competitors_list, # Will be None if not successful
        "subgraph_error": parse_error,
        "attempt": attempt + 1,
        "_route_decision": next_step
    }


# --- Graph Definition Function ---
def create_competitor_analysis_graph():
    """Builds the ReAct agent subgraph for competitor analysis."""
    builder = StateGraph(CompetitorSubgraphState)
    builder.add_node("react_agent", react_competitor_agent_node)
    builder.add_node("check_result", check_and_parse_competitor_result_node)
    builder.add_edge(START, "react_agent")
    builder.add_edge("react_agent", "check_result")
    builder.add_conditional_edges(
        "check_result",
        lambda state: state.get("_route_decision", "fail"),
        {"success": END, "retry": "react_agent", "fail": END}
    )
    graph = builder.compile()
    logger.info("Competitor Analysis Worker Subgraph compiled.")
    return graph

# --- Compiled Subgraph ---
competitor_analysis_subgraph_runnable = create_competitor_analysis_graph()

# --- Main Block for Direct Testing ---
if __name__ == '__main__':
    import pprint
    import time
    from google import genai
    import os
    import sys
    from pathlib import Path
    from config import config

    logger = config.logger

    logger.info("--- Running Competitor Analysis Worker Subgraph Directly ---")

    if not config.GOOGLE_API_KEY: 
        print("ERROR: GOOGLE_API_KEY not set.")
        exit()
    # Check search keys if using google_search tool
    if not config.GOOGLE_SEARCH_API_KEY or not config.GOOGLE_CSE_ID: 
        print("WARN: Google Search keys not set, may fallback to DDG.")
    # try: genai.configure(api_key=config.GOOGLE_API_KEY); logger.info("Gemini client configured.")
    # except Exception as e: print(f"ERROR: Failed to configure Gemini client: {e}"); exit()

    # Test Cases
    test_companies = ["Apple", "Infosys", "Tesla"]

    # Compile graph
    subgraph_app = competitor_analysis_subgraph_runnable

    all_final_states = {}
    for company in test_companies:
        print(f"\n>>> Testing Competitor Analysis Subgraph for: {company} <<<\n")
        initial_subgraph_state: CompetitorSubgraphState = {
            "company_name": company,
            "attempt": 1,
            "max_attempts": 2,
            "messages": [],
            "competitors": None,
            "subgraph_error": None,
        }
        final_state = None
        start_time = time.time()
        try:
            print("--- Streaming Subgraph Execution ---")
            for step_count, step_state in enumerate(subgraph_app.stream(
                initial_subgraph_state,
                # {"recursion_limit": 30}
            )):
                 # ... (Same detailed logging as stock data test) ...
                node_name = list(step_state.keys())[0]
                node_output = step_state[node_name]
                print(f"\n=== Step {step_count+1}: Node '{node_name}' Output ===")
                current_state = node_output
                pprint.pprint({k: v for k, v in current_state.items() if k != 'messages'}, indent=2)
                if current_state.get('messages'):
                    last_msg = current_state['messages'][-1]
                    print(f"\n  Last Message ({last_msg.type}):")
                    try:
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            print("    Tool Calls:"); pprint.pprint(last_msg.tool_calls)
                        elif hasattr(last_msg, 'content'): print(f"    Content: {str(last_msg.content)[:1000]}...")
                    except Exception: print(f"    Content: {last_msg}")
                final_state = current_state
                time.sleep(0.5)


            print("\n--- Subgraph Execution Complete ---")
            # Re-invoke to get final state easily if stream state isn't reliable
            # final_state = subgraph_app.invoke(initial_subgraph_state, {"recursion_limit": 30})

        except Exception as e:
            print(f"\n--- Subgraph Execution Failed for {company} ---")
            logger.exception("Error during subgraph execution:")
            print(f"Error: {e}")
            final_state = {"subgraph_error": f"Subgraph execution failed: {e}"}

        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds.")
        logger.info(f"Final Subgraph State for {company}: {final_state}")
        print("\nFinal State Snapshot:")
        pprint.pprint(final_state, indent=2)
        all_final_states[company] = final_state
        print("\n>>> Test Completed <<<\n")
        print("Pausing before next test...")
        time.sleep(10)

    print("\n--- Competitor Analysis Subgraph Test Script Finished ---")