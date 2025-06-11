# In your main graph file or a new file like sector_analysis_nodes.py

import json
from datetime import date
from typing import Dict, Any, List, Set,Optional
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.state import AgentState)
from state import AgentState # Assuming state is defined
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.config import logger)
from config import logger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src import config)
import config # For LLM model name, etc.
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.prompts import ...)
from prompts import LLM_SECTOR_REPORT_GENERATION_PROMPT_TEMPLATE
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.nodes.sector_analysis_node import ...)
from agents.sector_analysis_node import prepare_data_for_llm_sector_report

try:
    report_generation_llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL_NAME,
        temperature=0.2,
        convert_system_message_to_human=True
    )
    logger.info("Sector Report Generation LLM initialized.")
except Exception as e:
    logger.error(f"Failed to initialize Sector Report Generation LLM: {e}")
    report_generation_llm = None


def generate_llm_sector_report_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates the final sector analysis report using an LLM.
    """
    sector_name = state.get("sector_name", "N/A")
    logger.info(f"--- Generating LLM-based Final Sector Report for: {sector_name} ---")

    # Use the same report_generation_llm as for company reports, or a dedicated one
    if not report_generation_llm: # Check if it's initialized
        logger.error("[SectorReportNode] Report Generation LLM not available.")
        # Try to get URLs even if report gen fails
        urls_prepared = state.get("_consolidated_urls_for_state_update", []) # Assuming helper might have set this
        return {
            "final_sector_report_markdown": "Error: Report generation LLM is not initialized.",
            "sector_report_urls": urls_prepared
        }

    # 1. Prepare context for LLM
    try:
        report_context_data = prepare_data_for_llm_sector_report(state)
        final_llm_prompt = LLM_SECTOR_REPORT_GENERATION_PROMPT_TEMPLATE.format(**report_context_data)
        
        # Basic truncation
        max_prompt_length = 30000 # Adjust as needed
        if len(final_llm_prompt) > max_prompt_length:
            logger.warning(f"Sector report prompt for {sector_name} too long ({len(final_llm_prompt)}), truncating.")
            # Simplified truncation for now
            final_llm_prompt = final_llm_prompt[:max_prompt_length - 500] + "\n\n[CONTEXT TRUNCATED DUE TO LENGTH]"
        
        logger.debug(f"[SectorReportNode] LLM Prompt (first 500 chars):\n{final_llm_prompt[:500]}...")

    except Exception as e:
        logger.error(f"[SectorReportNode] Error preparing data for LLM sector report: {e}", exc_info=True)
        return {
            "final_sector_report_markdown": f"Error: Could not prepare data for sector report. Details: {e}",
            "sector_report_urls": []
        }

    # 2. Call LLM
    try:
        messages = [HumanMessage(content=final_llm_prompt)]
        ai_response = report_generation_llm.invoke(messages) # Use the appropriate LLM instance
        generated_report_markdown = ai_response.content.strip()

        if len(generated_report_markdown) < 500 or not generated_report_markdown.startswith("#"):
            logger.warning(f"[SectorReportNode] LLM response seems too short or not Markdown. Response: {generated_report_markdown[:300]}")

        logger.info(f"LLM-based sector report generated for {sector_name}. Length: {len(generated_report_markdown)} chars.")
        
        # Get URLs from the prepared context (helper function should add this)
        consolidated_urls = report_context_data.get("_consolidated_urls_for_state_update", [])

        return {
            "final_sector_report_markdown": generated_report_markdown,
            "sector_report_urls": consolidated_urls # Store the consolidated URLs
        }

    except Exception as e:
        logger.error(f"[SectorReportNode] LLM call failed during sector report generation: {e}", exc_info=True)
        urls_prepared_fallback = prepare_data_for_llm_sector_report(state).get("_consolidated_urls_for_state_update", [])
        return {
            "final_sector_report_markdown": f"Error: LLM call failed during sector report generation. Details: {e}",
            "sector_report_urls": urls_prepared_fallback
        }