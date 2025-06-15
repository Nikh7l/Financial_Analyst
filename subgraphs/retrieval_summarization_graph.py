# retrieval_summarization_graph.py
import json
import re
from typing import Dict, Any, List, Optional, TypedDict

# LangGraph components
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage

# Project imports
from agents.retrieval_agent import retrieval_agent_runnable
from agents.summarization_agent import summarize_documents_node
from core.state import RetrievalSummarizationState

import os
import sys
from pathlib import Path
from config import config

logger = config.logger
# --- Node Logic Functions ---

def process_retrieval_output(state: RetrievalSummarizationState) -> Dict[str, Any]:
    """
    Parses the final message from the retrieval agent's state
    to extract document URLs and update the main subgraph state.
    Also checks for errors during retrieval.
    """
    logger.info("[Subgraph] Processing Retrieval Agent Output...")
    messages = state.get("messages", [])
    if not messages:
        logger.error("[Subgraph] No messages found in state after retrieval agent.")
        return {"subgraph_error": "Retrieval agent did not produce any messages."}

    final_message = messages[-1]
    document_urls = []
    subgraph_error = state.get("subgraph_error") # Preserve previous errors if any

    if isinstance(final_message, AIMessage) and isinstance(final_message.content, str):
        response_text = final_message.content.strip()
        logger.debug(f"[Subgraph] Final Retrieval AI Message: {response_text}")
        try:
            # Robust JSON finding
            match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
            if match: json_str = match.group(1)
            else:
                json_start = response_text.find('{'); json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end != -1: json_str = response_text[json_start:json_end]
                else: raise ValueError("JSON object not found")

            parsed_json = json.loads(json_str)
            if "document_urls" in parsed_json and isinstance(parsed_json["document_urls"], list):
                document_urls = [url for url in parsed_json["document_urls"] if isinstance(url, str) and url.startswith('http')]
                logger.info(f"[Subgraph] Extracted URLs: {document_urls}")
                # Clear previous error if we successfully got URLs
                subgraph_error = None
            else:
                logger.warning("[Subgraph] 'document_urls' key missing or invalid in retrieval JSON.")
                subgraph_error = subgraph_error or "Retrieval agent JSON missing document_urls."
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"[Subgraph] Failed to parse JSON from retrieval output: {e}. Response: {response_text}")
            subgraph_error = subgraph_error or f"Failed to parse retrieval JSON: {e}"
        except Exception as e:
             logger.exception("[Subgraph] Unexpected error parsing retrieval output.")
             subgraph_error = subgraph_error or f"Unexpected parsing error: {e}"
    elif isinstance(final_message, ToolMessage):
         logger.warning(f"[Subgraph] Retrieval agent ended with a ToolMessage, indicating potential loop/error: {final_message.content}")
         subgraph_error = subgraph_error or "Retrieval agent ended unexpectedly after tool call."
    else:
        logger.warning(f"[Subgraph] Retrieval agent ended with unexpected message type: {type(final_message)}")
        subgraph_error = subgraph_error or "Retrieval agent ended with unexpected message type."

    # Always return the extracted URLs (even if empty) and any error found
    return {"document_urls": document_urls, "subgraph_error": subgraph_error}


# --- Graph Definition ---

def create_retrieval_summarization_graph() :
    """Builds and compiles the retrieval-summarization subgraph."""
    builder = StateGraph(RetrievalSummarizationState)

    # Node 1: The ReAct Retrieval Agent (added directly as a compiled graph)
    builder.add_node("retrieve_docs_agent", retrieval_agent_runnable)

    # Node 2: Process the output of the retrieval agent
    builder.add_node("process_retrieval_output", process_retrieval_output)

    # Node 3: The Summarization Node function
    # We need to make sure the client is available here if summarize_documents_node needs it.
    # Option A: Assume summarize_documents_node uses default client (preferred)
    builder.add_node("summarize_docs_node", summarize_documents_node)
    # Option B: Pass client via config/context (complex)
    # Option C: Modify summarize_documents_node to accept it from state (less ideal)

    # Define Edges
    builder.add_edge(START, "retrieve_docs_agent")
    builder.add_edge("retrieve_docs_agent", "process_retrieval_output")

    # Conditional edge after processing retrieval output
    def should_summarize(state: RetrievalSummarizationState) -> str:
        if state.get("subgraph_error"):
             logger.warning(f"[Subgraph] Skipping summarization due to error: {state['subgraph_error']}")
             return END
        elif not state.get("document_urls"):
            logger.info("[Subgraph] Skipping summarization as no documents were retrieved/extracted.")
            return END
        else:
            logger.info("[Subgraph] Proceeding to summarization.")
            return "summarize_docs_node"

    builder.add_conditional_edges(
         "process_retrieval_output",
         should_summarize,
         {
             "summarize_docs_node": "summarize_docs_node",
             END: END
         }
     )

    # Final edge
    builder.add_edge("summarize_docs_node", END)

    graph = builder.compile()
    logger.info("Retrieval-Summarization Subgraph compiled.")
    return graph

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    from config import logger
    logger.info("Testing subgraph execution directly...")
    from dotenv import load_dotenv
    load_dotenv()
    # if not config.GOOGLE_API_KEY: print("ERROR: GOOGLE_API_KEY not set."); exit()
    # # try: genai.configure(api_key=config.GOOGLE_API_KEY); logger.info("Gemini client configured.")
    # except Exception as e: print(f"ERROR: Failed to configure Gemini client: {e}"); exit()
# 
    subgraph_app = create_retrieval_summarization_graph()

    company = "Microsoft" # "Apple" # "Infosys"
    # Initial state only needs company_name and empty messages
    initial_state: RetrievalSummarizationState = {
        "company_name": company,
        "document_urls": None,
        "document_summaries": None,
        "subgraph_error": None,
        "messages": [("human", f"Find the latest annual and quarterly financial reports for {company}.")] # Initial message for ReAct agent
    }

    import pprint
    logger.info(f"\n--- Invoking Subgraph for {company} ---")
    final_state = None
    try:
        # Use invoke to get final state easily for testing subgraph output
        final_state_dict = subgraph_app.invoke(initial_state, {"recursion_limit": 30}) # Increase limit for safety
        logger.info("\n--- Subgraph Execution Complete ---")
        logger.info("Final Subgraph State:")
        final_state = final_state_dict.copy()  # Copy the final state for logging
        logger.info(f"Final State: {final_state}")
        # pprint.pprint(final_state_dict, indent=2)

    except Exception as e:
        print(f"\n--- Subgraph Execution Failed ---")
        logger.exception("Error during subgraph execution:")
        print(f"Error: {e}")