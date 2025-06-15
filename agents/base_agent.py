# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Any, Optional
import logging
import inspect # Keep inspect

# Import NEW Gemini libraries and types
from google import genai
from google.genai import types

# Import the state definition
from core.state import AgentState

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for agents using the NEW google-genai SDK (Client-based).
    Revised tool handling.
    """
    def __init__(self, client: genai.Client, model_name: str, tools: Optional[List[Callable]] = None):
        """
        Initializes the BaseAgent with a Gemini Client, model name, and optional tools.

        Args:
            client: An initialized google.genai.Client instance.
            model_name: The string identifier of the Gemini model to use.
            tools: An optional list of callable tool functions available to this agent.
                   These functions will be passed directly to the Gemini API.
        """
        self.client = client
        self.model_name = model_name
        # Store the raw list of functions
        self.callable_tools = tools if tools else []
        # Create the mapping from function name to function object for execution
        self.tool_mapping = {func.__name__: func for func in self.callable_tools}
        # The gemini_tool object is no longer needed here, tools passed directly in _call_llm


    def _call_llm(self, messages: List[Dict[str, Any]], use_tools: bool = True) -> types.GenerateContentResponse:
        """
        Helper method to invoke the Gemini model via the client. Passes tools directly.
        """
        logger.debug(f"Calling model '{self.model_name}' with {len(messages)} messages. Tools enabled: {use_tools}")

        gen_config_args = {}
        tool_config_args = {}

        # Pass the list of callable functions directly if tools are enabled
        if use_tools and self.callable_tools:
            gen_config_args['tools'] = self.callable_tools # <-- Pass callables directly
            # Optional: Configure function calling mode if needed
            # tool_config_args['function_calling_config'] = types.FunctionCallingConfig(mode='AUTO')

        final_tool_config = types.ToolConfig(**tool_config_args) if tool_config_args else None
        if final_tool_config:
             gen_config_args['tool_config'] = final_tool_config

        final_gen_config = types.GenerateContentConfig(**gen_config_args) if gen_config_args else None

        try:
            formatted_contents = self._format_contents_for_genai(messages)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=formatted_contents,
                config=final_gen_config,
            )

            # ... (Logging and basic error checking) ...
            log_msg = f"LLM Raw Response: {str(response)[:500]}..." # Truncate log
            logger.debug(log_msg)
            return response

        except Exception as e:
            logger.error(f"Error during LLM call via client: {e}", exc_info=True)
            raise

    # --- _execute_tool and _format_contents_for_genai remain the same ---
    def _execute_tool(self, function_call: types.FunctionCall) -> types.Part:
        tool_name = function_call.name
        args = dict(function_call.args)
        logger.info(f"Attempting to execute tool: {tool_name} with args: {args}")
        if tool_name not in self.tool_mapping:
            logger.error(f"Tool '{tool_name}' not found in agent's mapping.")
            raise ValueError(f"Tool '{tool_name}' not found.")
        func = self.tool_mapping[tool_name]
        try:
            logger.debug(f"Executing function: {func.__name__}")
            result = func(**args)
            logger.debug(f"Tool '{tool_name}' raw result: {result}")
            if isinstance(result, (str, int, float, bool, list, dict)):
                response_data = result
            elif hasattr(result, 'model_dump'):
                 response_data = result.model_dump(mode="json")
            else:
                 response_data = str(result)
                 logger.warning(f"Tool '{tool_name}' result type {type(result)} converted to string.")
            return types.Part.from_function_response(name=tool_name, response={'result': response_data})
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}' with args {args}: {e}", exc_info=True)
            return types.Part.from_function_response(name=tool_name, response={'error': str(e)})


    def _format_contents_for_genai(self, messages: List[Dict[str, Any]]) -> List[types.Content]:
        contents = []
        for msg in messages:
            role = msg.get('role')
            parts_data = msg.get('parts')
            if not role or not parts_data: continue
            if role == 'assistant': role = 'model'
            if role == 'system': role = 'user'
            if not isinstance(parts_data, list):
                if isinstance(parts_data, str): parts_data = [{'text': parts_data}]
                elif isinstance(msg.get('content'), str): parts_data = [{'text': msg['content']}]
                else: continue
            content_parts = []
            for part_item in parts_data:
                 if isinstance(part_item, dict):
                    if 'text' in part_item:
                        content_parts.append(types.Part.from_text(text=part_item['text']))
                    elif 'function_call' in part_item:
                        fc_data = part_item['function_call']
                        content_parts.append(types.Part.from_function_call(name=fc_data['name'], args=fc_data.get('args', {})))
                    elif 'function_response' in part_item:
                        fr_data = part_item['function_response']
                        content_parts.append(types.Part.from_function_response(name=fr_data['name'], response=fr_data.get('response', {})))
                 elif isinstance(part_item, str):
                      content_parts.append(types.Part.from_text(text=part_item))
            if content_parts:
                contents.append(types.Content(role=role, parts=content_parts))
        return contents

    @abstractmethod
    def run(self, state: AgentState) -> Dict[str, Any]:
        """ Abstract method for agent-specific logic. """
        pass