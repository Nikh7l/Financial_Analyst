# agents/router_agent.py
import logging
import json
from typing import Dict, Any
from .base_agent import BaseAgent
from state import AgentState
from prompts import ROUTER_PROMPT_TEMPLATE 
from google import genai
# from google.genai import types

# Get logger
from config import logger

class RouterAgent(BaseAgent):
    """
    Agent responsible for classifying the user's query and extracting key entities.
    Determines if the query is about a specific 'company' or a 'sector'.
    Uses a prompt defined in prompts.py.
    """

    def __init__(self, client: genai.Client, model_name: str):
        super().__init__(client=client, model_name=model_name, tools=None)

    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Analyzes the user query using a predefined prompt template.

        Args:
            state: The current AgentState containing the user 'query'.

        Returns:
            A dictionary with updates for 'query_type', 'company_name',
            and/or 'sector_name', plus potentially 'error_message'.
        """
        query = state.get('query')
        if not query:
            logger.error("RouterAgent: No query found in state.")
            return {"error_message": "RouterAgent: Query is missing."}

        logger.info(f"RouterAgent processing query: '{query}'")

        # Format the imported prompt template with the actual query
        prompt = ROUTER_PROMPT_TEMPLATE.format(query=query) # <-- Use the imported template

        # Prepare messages for the LLM call
        messages = [{'role': 'user', 'parts': [prompt]}]

        try:
            # Call the LLM - No tools needed for routing
            response = self._call_llm(messages, use_tools=False)

            # --- Response parsing logic remains the same ---
            if response.candidates and response.candidates[0].content.parts:
                response_text = response.candidates[0].content.parts[0].text.strip()
                logger.debug(f"RouterAgent LLM raw response: {response_text}")
                try:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = response_text[json_start:json_end]
                        parsed_json = json.loads(json_str)
                        logger.info(f"RouterAgent parsed LLM response: {parsed_json}")

                        query_type = parsed_json.get("query_type")
                        entity_name = parsed_json.get("entity_name")

                        updates: Dict[str, Any] = {"query_type": query_type}
                        if query_type == "company" and entity_name:
                            updates["company_name"] = entity_name
                        elif query_type == "sector" and entity_name:
                            updates["sector_name"] = entity_name
                        elif query_type == "unknown":
                            logger.warning(f"RouterAgent classified query as 'unknown'. Query: {query}")
                            # Optionally add error message if needed by graph logic
                            # updates["error_message"] = "Query type is unknown."
                        else:
                            logger.error(f"RouterAgent received invalid type/entity: Type='{query_type}', Entity='{entity_name}'")
                            updates["query_type"] = "unknown"
                            updates["error_message"] = "RouterAgent failed to get valid classification from LLM."

                        return updates
                    else:
                         logger.error(f"RouterAgent could not find JSON object in LLM response: {response_text}")
                         return {"query_type": "unknown", "error_message": "RouterAgent failed to parse LLM response (JSON not found)."}
                except json.JSONDecodeError as e:
                    logger.error(f"RouterAgent failed to parse JSON response: {e}\nResponse text: {response_text}")
                    return {"query_type": "unknown", "error_message": f"RouterAgent failed to parse LLM response (JSON error: {e})."}
            else:
                logger.error(f"RouterAgent received no valid response candidate from LLM. Response: {response}")
                error_msg = "RouterAgent received no valid response from LLM."
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    error_msg += f" Reason: {response.prompt_feedback.block_reason}"
                return {"query_type": "unknown", "error_message": error_msg}

        except Exception as e:
            logger.error(f"RouterAgent encountered an exception during LLM call: {e}", exc_info=True)
            return {"query_type": "unknown", "error_message": f"RouterAgent failed: {e}"}