# In your main graph file or a new file like sector_analysis_nodes.py
import os
import sys
import json
import re
from typing import Dict, Any, List, Optional, Set

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from state import AgentState, SectorOutlookPayload, SectorOutlookDetail # Your state types
from prompts import SECTOR_SYNTHESIS_OUTLOOK_PROMPT_TEMPLATE
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import config
from config import logger
from datetime import date


# Initialize LLM for sector outlook synthesis
try:
    sector_outlook_llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL_NAME_PRO, # Use a powerful model for synthesis
        temperature=0.2,
        convert_system_message_to_human=True
    )
    logger.info("Sector Outlook LLM initialized.")
except Exception as e:
    logger.error(f"Failed to initialize Sector Outlook LLM: {e}")
    sector_outlook_llm = None


def format_list_for_prompt(data_list: Optional[List[str]], default_text="N/A or not specified.") -> str:
    """Helper to format a list into a newline-separated string for prompts."""
    if data_list and isinstance(data_list, list):
        return "\n".join([f"- {item}" for item in data_list]) if data_list else default_text
    return default_text

def prepare_data_for_sector_outlook_synthesis(state: AgentState) -> Dict[str, str]:
    """
    Collects and formats all necessary data from AgentState to be used
    as context for the LLM sector outlook synthesis prompt.
    """
    logger.info("Preparing data context for LLM sector outlook synthesis...")
    data_context = {}

    sector_name = state.get("sector_name", "N/A")
    data_context["sector_name"] = sector_name

    # B. Recent News & Market Sentiment Analysis
    sentiment_payload = state.get("sector_news_sentiment_output", {})
    sentiment_error = sentiment_payload.get("error")
    sentiment_analysis_data = sentiment_payload.get("analysis_data", {}) # This is the dict with sentiment, themes etc.
    if sentiment_error:
        data_context["sector_sentiment_value"] = "Error"
        data_context["sector_key_news_themes_text"] = f"- Error: {sentiment_error}"
        data_context["sector_recent_events_text"] = f"- Error: {sentiment_error}"
        data_context["sector_sentiment_reasoning_text"] = f"Error: {sentiment_error}"
    elif not sentiment_analysis_data:
        data_context["sector_sentiment_value"] = "Data Unavailable"
        data_context["sector_key_news_themes_text"] = "- Data Unavailable"
        data_context["sector_recent_events_text"] = "- Data Unavailable"
        data_context["sector_sentiment_reasoning_text"] = "Data Unavailable"
    else:
        data_context["sector_sentiment_value"] = sentiment_analysis_data.get("overall_sentiment", "N/A")
        data_context["sector_key_news_themes_text"] = format_list_for_prompt(sentiment_analysis_data.get("key_news_themes"))
        data_context["sector_recent_events_text"] = format_list_for_prompt(sentiment_analysis_data.get("recent_events"))
        data_context["sector_sentiment_reasoning_text"] = sentiment_analysis_data.get("sentiment_reasoning", "N/A")

    # C. Key Players in the Sector
    players_payload = state.get("sector_key_players_output", {})
    players_error = players_payload.get("error")
    key_players_list = players_payload.get("key_players", [])
    if players_error:
        data_context["sector_key_players_text"] = f"- Error: {players_error}"
    elif not key_players_list:
        data_context["sector_key_players_text"] = "- No key players identified or data unavailable."
    else:
        player_texts = [f"- {p.get('name', 'N/A')}: {p.get('description', 'N/A')}" for p in key_players_list]
        data_context["sector_key_players_text"] = "\n".join(player_texts)

    # D. Sector Market Data
    market_data_payload = state.get("sector_market_data_output", {})
    market_error = market_data_payload.get("error")
    market_data_details = market_data_payload.get("market_data", {}) # This is the dict with size, cagr etc.
    if market_error:
        data_context["market_size_estimate_text"] = f"Error: {market_error}"
        data_context["projected_cagr_text"] = "Error"
        data_context["key_market_segments_text"] = "- Error"
        data_context["key_geographies_text"] = "- Error"
        data_context["market_data_growth_drivers_text"] = "- Error"
        data_context["market_data_challenges_text"] = "- Error"
    elif not market_data_details:
        data_context["market_size_estimate_text"] = "Data Unavailable"
        data_context["projected_cagr_text"] = "Data Unavailable"
        # ... and so on for all keys
        data_context["key_market_segments_text"] = "- Data Unavailable"
        data_context["key_geographies_text"] = "- Data Unavailable"
        data_context["market_data_growth_drivers_text"] = "- Data Unavailable"
        data_context["market_data_challenges_text"] = "- Data Unavailable"
    else:
        data_context["market_size_estimate_text"] = market_data_details.get("market_size_estimate", "Not found")
        data_context["projected_cagr_text"] = market_data_details.get("projected_cagr", "Not found")
        data_context["key_market_segments_text"] = format_list_for_prompt(market_data_details.get("key_market_segments"))
        data_context["key_geographies_text"] = format_list_for_prompt(market_data_details.get("key_geographies"))
        data_context["market_data_growth_drivers_text"] = format_list_for_prompt(market_data_details.get("primary_growth_drivers"))
        data_context["market_data_challenges_text"] = format_list_for_prompt(market_data_details.get("primary_market_challenges"))

    # E. Sector Trends, Innovations, Challenges, and Opportunities
    trends_payload = state.get("sector_trends_innovations_output", {})
    trends_error = trends_payload.get("error")
    trends_data_details = trends_payload.get("trends_data", {}) # This is the dict with trends, innovations, etc.
    if trends_error:
        data_context["trends_major_trends_text"] = f"- Error: {trends_error}"
        # ... and so on for all keys
        data_context["trends_recent_innovations_text"] = "- Error"
        data_context["trends_key_challenges_text"] = "- Error"
        data_context["trends_emerging_opportunities_text"] = "- Error"
    elif not trends_data_details:
        data_context["trends_major_trends_text"] = "- Data Unavailable"
        # ... and so on
        data_context["trends_recent_innovations_text"] = "- Data Unavailable"
        data_context["trends_key_challenges_text"] = "- Data Unavailable"
        data_context["trends_emerging_opportunities_text"] = "- Data Unavailable"
    else:
        data_context["trends_major_trends_text"] = format_list_for_prompt(trends_data_details.get("major_trends"))
        data_context["trends_recent_innovations_text"] = format_list_for_prompt(trends_data_details.get("recent_innovations"))
        data_context["trends_key_challenges_text"] = format_list_for_prompt(trends_data_details.get("key_challenges"))
        data_context["trends_emerging_opportunities_text"] = format_list_for_prompt(trends_data_details.get("emerging_opportunities"))
        
    logger.info("Data context prepared for sector outlook synthesis.")
    return data_context


def synthesize_sector_outlook_node(state: AgentState) -> Dict[str, SectorOutlookPayload]:
    """
    Synthesizes all gathered sector data using an LLM to provide an overall outlook.
    """
    sector_name = state.get("sector_name", "N/A")
    logger.info(f"--- Synthesizing Sector Outlook for: {sector_name} ---")

    default_error_payload: SectorOutlookPayload = {
        "outlook_data": {"overall_outlook": "N/A", "outlook_summary": "Failed to synthesize outlook."},
        "error": "Synthesis failed"
    }

    if not sector_outlook_llm:
        logger.error("[SectorOutlookNode] LLM not available.")
        default_error_payload["error"] = "Sector Outlook LLM not initialized."
        return {"sector_outlook_output": default_error_payload}

    # 1. Prepare context for LLM
    try:
        outlook_context_data = prepare_data_for_sector_outlook_synthesis(state)
        final_llm_prompt = SECTOR_SYNTHESIS_OUTLOOK_PROMPT_TEMPLATE.format(**outlook_context_data)
        
        # Basic truncation if needed
        max_prompt_len = 28000 # Adjust
        if len(final_llm_prompt) > max_prompt_len:
            logger.warning(f"Sector outlook prompt too long ({len(final_llm_prompt)}), truncating.")
            # A simple truncation, could be made smarter
            final_llm_prompt = final_llm_prompt[:max_prompt_len-500] + "\n\n[CONTEXT TRUNCATED]"

        logger.debug(f"[SectorOutlookNode] LLM Prompt for Outlook (first 500 chars):\n{final_llm_prompt[:500]}...")

    except Exception as e:
        logger.error(f"[SectorOutlookNode] Error preparing data for LLM: {e}", exc_info=True)
        default_error_payload["error"] = f"Error preparing context: {e}"
        return {"sector_outlook_output": default_error_payload}

    # 2. Call LLM
    try:
        messages = [HumanMessage(content=final_llm_prompt)]
        ai_response = sector_outlook_llm.invoke(messages)
        response_content = ai_response.content.strip()
        logger.debug(f"[SectorOutlookNode] LLM raw response: {response_content}")
    except Exception as e:
        logger.error(f"[SectorOutlookNode] LLM call failed: {e}", exc_info=True)
        default_error_payload["error"] = f"LLM call failed: {e}"
        return {"sector_outlook_output": default_error_payload}

    # 3. Parse and Validate LLM Output
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_content, re.DOTALL | re.IGNORECASE)
        if match: json_str = match.group(1)
        else:
            json_start = response_content.find('{'); json_end = response_content.rfind('}') + 1
            if json_start != -1 and json_end != -1 and json_end > json_start: json_str = response_content[json_start:json_end]
            else: raise ValueError("JSON object not found in LLM response for sector outlook.")
        
        parsed_outlook_data_dict = json.loads(json_str) # This is the dict for SectorOutlookDetail

        # Validate required fields
        if not parsed_outlook_data_dict.get("overall_outlook") or \
           not parsed_outlook_data_dict.get("outlook_summary"):
            raise ValueError("Missing 'overall_outlook' or 'outlook_summary' in LLM response.")

        # Ensure list fields are lists, default to empty if not or missing
        list_keys = ["key_growth_drivers_summary", "key_risks_challenges_summary"]
        for key in list_keys:
            if key in parsed_outlook_data_dict and not isinstance(parsed_outlook_data_dict[key], list):
                logger.warning(f"LLM output '{key}' was not a list, defaulting to empty.")
                parsed_outlook_data_dict[key] = []
            elif key not in parsed_outlook_data_dict:
                 parsed_outlook_data_dict[key] = []
        
        if "investment_considerations" not in parsed_outlook_data_dict:
            parsed_outlook_data_dict["investment_considerations"] = "Not specified."


        # Explicitly cast to SectorOutlookDetail for type checking if needed,
        # but if parsed_outlook_data_dict matches the structure, it's fine.
        # final_outlook_detail: SectorOutlookDetail = parsed_outlook_data_dict

        payload: SectorOutlookPayload = {
            "outlook_data": parsed_outlook_data_dict, # Store the validated dict
            "error": None
        }
        logger.info(f"[SectorOutlookNode] Successfully synthesized outlook for {sector_name}: {payload['outlook_data'].get('overall_outlook')}")
        return {"sector_outlook_output": payload}

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"[SectorOutlookNode] Failed to parse/validate outlook JSON: {e}. Response: {response_content}", exc_info=True)
        default_error_payload["error"] = f"Failed to parse/validate outlook: {e}"
        # Try to put the raw response in summary for debugging if outlook_data is expected
        if default_error_payload["outlook_data"]:
            default_error_payload["outlook_data"]["outlook_summary"] = f"Could not parse LLM output: {response_content}"
        return {"sector_outlook_output": default_error_payload}
    except Exception as e:
        logger.error(f"[SectorOutlookNode] Unexpected error processing LLM response: {e}", exc_info=True)
        default_error_payload["error"] = f"Unexpected error processing outlook: {e}"
        return {"sector_outlook_output": default_error_payload}
    

def prepare_data_for_llm_sector_report(state: AgentState) -> Dict[str, Any]:
    """
    Collects and formats all necessary data from AgentState for the LLM Sector Report prompt.
    """
    logger.info("Preparing data context for LLM Sector Report generation...")
    data_context = {}
    all_urls_collected_for_report: Set[str] = set()

    # I. General Information
    sector_name = state.get("sector_name", "N/A")
    data_context["sector_name"] = sector_name
    data_context["original_query"] = state.get("query", "N/A")
    data_context["report_date"] = date.today().isoformat()

    # II. Sector News & Market Sentiment Analysis
    sentiment_payload = state.get("sector_news_sentiment_output", {})
    data_context["sector_sentiment_error"] = sentiment_payload.get("error", "None")
    sentiment_analysis_data = sentiment_payload.get("analysis_data", {})
    if data_context["sector_sentiment_error"] != "None" and data_context["sector_sentiment_error"] is not None:
        data_context["sector_sentiment_value"] = "Error"
        data_context["sector_key_news_themes_text"] = f"- Error: {data_context['sector_sentiment_error']}"
        data_context["sector_recent_events_text"] = f"- Error: {data_context['sector_sentiment_error']}"
        data_context["sector_sentiment_reasoning_text"] = f"Error: {data_context['sector_sentiment_error']}"
    elif not sentiment_analysis_data:
        data_context["sector_sentiment_value"] = "Data Unavailable"
        data_context["sector_key_news_themes_text"] = "- Data Unavailable"
        data_context["sector_recent_events_text"] = "- Data Unavailable"
        data_context["sector_sentiment_reasoning_text"] = "Data Unavailable"
    else:
        data_context["sector_sentiment_value"] = sentiment_analysis_data.get("overall_sentiment", "N/A")
        data_context["sector_key_news_themes_text"] = format_list_for_prompt(sentiment_analysis_data.get("key_news_themes"))
        data_context["sector_recent_events_text"] = format_list_for_prompt(sentiment_analysis_data.get("recent_events"))
        data_context["sector_sentiment_reasoning_text"] = sentiment_analysis_data.get("sentiment_reasoning", "N/A")
        all_urls_collected_for_report.update(sentiment_analysis_data.get("source_urls_used", []))

    # III. Key Players in the Sector
    players_payload = state.get("sector_key_players_output", {})
    data_context["sector_key_players_error"] = players_payload.get("error", "None")
    key_players_list = players_payload.get("key_players", [])
    if data_context["sector_key_players_error"] != "None" and data_context["sector_key_players_error"] is not None:
        data_context["sector_key_players_text"] = f"- Error: {data_context['sector_key_players_error']}"
    elif not key_players_list:
        data_context["sector_key_players_text"] = "- No key players identified or data unavailable."
    else:
        player_texts = [f"- **{p.get('name', 'N/A')}**: {p.get('description', 'N/A')}" for p in key_players_list]
        data_context["sector_key_players_text"] = "\n".join(player_texts)
    all_urls_collected_for_report.update(players_payload.get("source_urls_used", []))


    # IV. Sector Market Data
    market_data_payload = state.get("sector_market_data_output", {})
    data_context["sector_market_data_error"] = market_data_payload.get("error", "None")
    market_data_details = market_data_payload.get("market_data", {})
    if data_context["sector_market_data_error"] != "None" and data_context["sector_market_data_error"] is not None:
        # Set all market data fields to error
        for key in ["market_size_estimate_text", "projected_cagr_text", "key_market_segments_text", "key_geographies_text", "market_data_growth_drivers_text", "market_data_challenges_text"]:
            data_context[key] = f"Error: {data_context['sector_market_data_error']}"
    elif not market_data_details:
        for key in ["market_size_estimate_text", "projected_cagr_text", "key_market_segments_text", "key_geographies_text", "market_data_growth_drivers_text", "market_data_challenges_text"]:
            data_context[key] = "Data Unavailable"
    else:
        data_context["market_size_estimate_text"] = market_data_details.get("market_size_estimate", "Not found")
        data_context["projected_cagr_text"] = market_data_details.get("projected_cagr", "Not found")
        data_context["key_market_segments_text"] = format_list_for_prompt(market_data_details.get("key_market_segments"))
        data_context["key_geographies_text"] = format_list_for_prompt(market_data_details.get("key_geographies"))
        data_context["market_data_growth_drivers_text"] = format_list_for_prompt(market_data_details.get("primary_growth_drivers"))
        data_context["market_data_challenges_text"] = format_list_for_prompt(market_data_details.get("primary_market_challenges"))
    all_urls_collected_for_report.update(market_data_payload.get("source_urls_used", []))

    # V. Sector Trends, Innovations, Challenges, and Opportunities
    trends_payload = state.get("sector_trends_innovations_output", {})
    data_context["sector_trends_error"] = trends_payload.get("error", "None")
    trends_data_details = trends_payload.get("trends_data", {})
    if data_context["sector_trends_error"] != "None" and data_context["sector_trends_error"] is not None:
        for key in ["trends_major_trends_text", "trends_recent_innovations_text", "trends_key_challenges_text", "trends_emerging_opportunities_text"]:
            data_context[key] = f"Error: {data_context['sector_trends_error']}"
    elif not trends_data_details:
        for key in ["trends_major_trends_text", "trends_recent_innovations_text", "trends_key_challenges_text", "trends_emerging_opportunities_text"]:
            data_context[key] = "Data Unavailable"
    else:
        data_context["trends_major_trends_text"] = format_list_for_prompt(trends_data_details.get("major_trends"))
        data_context["trends_recent_innovations_text"] = format_list_for_prompt(trends_data_details.get("recent_innovations"))
        data_context["trends_key_challenges_text"] = format_list_for_prompt(trends_data_details.get("key_challenges"))
        data_context["trends_emerging_opportunities_text"] = format_list_for_prompt(trends_data_details.get("emerging_opportunities"))
    all_urls_collected_for_report.update(trends_payload.get("source_urls_used", []))


    # VI. Synthesized Sector Outlook
    outlook_payload = state.get("sector_outlook_output", {})
    data_context["sector_outlook_error"] = outlook_payload.get("error", "None")
    outlook_data_details = outlook_payload.get("outlook_data", {})
    if data_context["sector_outlook_error"] != "None" and data_context["sector_outlook_error"] is not None:
        data_context["outlook_overall_value"] = "Error"
        data_context["outlook_summary_text"] = f"Error: {data_context['sector_outlook_error']}"
        data_context["outlook_growth_drivers_text"] = f"- Error: {data_context['sector_outlook_error']}"
        data_context["outlook_risks_challenges_text"] = f"- Error: {data_context['sector_outlook_error']}"
        data_context["outlook_investment_considerations_text"] = f"Error: {data_context['sector_outlook_error']}"
    elif not outlook_data_details:
        data_context["outlook_overall_value"] = "Data Unavailable"
        # ... and so on
        data_context["outlook_summary_text"] = "Data Unavailable"
        data_context["outlook_growth_drivers_text"] = "- Data Unavailable"
        data_context["outlook_risks_challenges_text"] = "- Data Unavailable"
        data_context["outlook_investment_considerations_text"] = "Data Unavailable"
    else:
        data_context["outlook_overall_value"] = outlook_data_details.get("overall_outlook", "N/A")
        data_context["outlook_summary_text"] = outlook_data_details.get("outlook_summary", "N/A")
        data_context["outlook_growth_drivers_text"] = format_list_for_prompt(outlook_data_details.get("key_growth_drivers_summary"))
        data_context["outlook_risks_challenges_text"] = format_list_for_prompt(outlook_data_details.get("key_risks_challenges_summary"))
        data_context["outlook_investment_considerations_text"] = outlook_data_details.get("investment_considerations", "N/A")

    # VII. Source URLs
    if not all_urls_collected_for_report:
        data_context["all_source_urls_list_text"] = "- No specific source URLs were collected during this analysis."
    else:
        data_context["all_source_urls_list_text"] = "\n".join([f"- {url}" for url in sorted(list(all_urls_collected_for_report))])
    
    # Add the final consolidated URL list to the main state for direct access later (will be returned by the node)
    data_context["_consolidated_urls_for_state_update"] = sorted(list(all_urls_collected_for_report))

    logger.info("Data context fully prepared for LLM sector report generation.")
    return data_context
