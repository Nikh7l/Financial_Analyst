# agents/report_generation_node.py
import json
from datetime import date
from typing import Dict, Any, List, Set
from langchain_google_genai import ChatGoogleGenerativeAI # Or your chosen LLM
from langchain_core.messages import HumanMessage
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src import config)
import config # For LLM model name, etc.
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.state import AgentState)
from state import AgentState
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.prompts import LLM_REPORT_GENERATION_PROMPT_TEMPLATE)
from prompts import LLM_REPORT_GENERATION_PROMPT_TEMPLATE
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.config import logger)
from config import logger

try:
    report_generation_llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL_NAME_PRO, # Use a model good at long-form generation
        temperature=0.2,
        convert_system_message_to_human=True
    )
    logger.info("Report Generation LLM initialized.")
except Exception as e:
    logger.error(f"Failed to initialize report generation LLM: {e}")
    report_generation_llm = None


def prepare_data_for_llm_report_generation(state: AgentState) -> Dict[str, str]:
    """
    Collects and formats all necessary data from AgentState to be used
    as context for the LLM report generation prompt.
    """
    data_context = {}
    all_urls_collected: Set[str] = set()

    # Basic Info
    data_context["company_name"] = state.get("company_name", "N/A")
    data_context["original_query"] = state.get("query", "N/A")
    data_context["report_date"] = date.today().isoformat()

    # A. Financial Report Summaries
    report_data = state.get("report_summaries_output", {})
    report_error = report_data.get("error")
    summaries_dict = report_data.get("document_summaries", {})
    report_urls = report_data.get("retrieved_document_urls", [])
    all_urls_collected.update(report_urls)
    if report_error:
        data_context["report_summaries_context"] = f"Error: {report_error}"
    elif not summaries_dict:
        data_context["report_summaries_context"] = "No financial report summaries were available."
    else:
        texts = [f"Summary from Document ({url}):\n{summary}" for url, summary in summaries_dict.items() if summary]
        data_context["report_summaries_context"] = "\n---\n".join(texts) if texts else "No summary content."
    
    logger.info(f"Report summaries context prepared with {data_context["report_summaries_context"]} summaries.")

    # B. Market Sentiment Analysis
    sentiment_payload = state.get("sentiment_analysis_output", {})
    sentiment_error = sentiment_payload.get("error")
    sentiment_data_dict = sentiment_payload.get("sentiment_data", {})
    if sentiment_error:
        data_context["sentiment_value"] = "Error"
        data_context["sentiment_report_text"] = f"Error: {sentiment_error}"
    elif not sentiment_data_dict:
        data_context["sentiment_value"] = "Unavailable"
        data_context["sentiment_report_text"] = "Sentiment data unavailable."
    else:
        data_context["sentiment_value"] = sentiment_data_dict.get("sentiment", "N/A")
        data_context["sentiment_report_text"] = sentiment_data_dict.get("detailed_sentiment_report", "N/A")
    # Add sentiment source URLs if available and collected by sentiment agent's wrapper
    # sentiment_source_urls = sentiment_data.get("source_urls", [])
    # all_urls_collected.update(sentiment_source_urls)

    logger.info(f"Sentiment analysis context prepared. Sentiment value: {data_context['sentiment_value']} - Report: {data_context['sentiment_report_text']}")

    # C. Current Stock Data
    stock_info_payload = state.get("stock_data_output", {})
    stock_error = stock_info_payload.get("error")
    actual_stock_metrics = stock_info_payload.get("stock_metrics", {})
    if stock_error:
        data_context["stock_data_json"] = json.dumps({"error": f"Error: {stock_error}"})
    elif not actual_stock_metrics:
        data_context["stock_data_json"] = json.dumps({"message": "Stock data unavailable."})
    else:
        data_context["stock_data_json"] = json.dumps(actual_stock_metrics, indent=2)
    
    logger.info(f"Stock data context prepared. Stock metrics: {data_context['stock_data_json']}")

    # D. Competitor Information
    competitor_payload = state.get("competitor_info_output", {})
    competitor_error = competitor_payload.get("error")
    actual_competitors_list = competitor_payload.get("competitors_list", [])
    if competitor_error:
        data_context["competitor_info_text"] = f"Error: {competitor_error}"
    elif not actual_competitors_list:
        data_context["competitor_info_text"] = "No competitor information available."
    else:
        texts = [f"- {c.get('name', 'N/A')}: {c.get('description', 'N/A')}" for c in actual_competitors_list]
        data_context["competitor_info_text"] = "\n".join(texts)
    # Add competitor source URLs if available
    # competitor_source_urls = competitor_data.get("source_urls", [])
    # all_urls_collected.update(competitor_source_urls)

    logger.info(f"Competitor information context prepared. Competitors: {data_context['competitor_info_text']} found.")

    # E. Investment Prediction
    prediction_data = state.get("prediction_output", {})
    pred_process_error = prediction_data.get("error")
    if pred_process_error:
        logger.warning(f"Prediction process reported an error: {pred_process_error}")
        # Provide error info to the report LLM
        data_context["prediction_recommendation"] = "ERROR"
        data_context["prediction_confidence"] = "N/A"
        data_context["prediction_reasoning"] = f"Prediction could not be generated due to an error: {pred_process_error}"
        data_context["prediction_positive_factors_text"] = "N/A"
        data_context["prediction_negative_factors_text"] = "N/A"
        data_context["prediction_data_limitations_text"] = "N/A"
        data_context["prediction_error"] = pred_process_error # Pass the specific error
    else:
        # Prediction process succeeded, format the data
        data_context["prediction_recommendation"] = prediction_data.get("recommendation", "UNCERTAIN")
        data_context["prediction_confidence"] = prediction_data.get("confidence", "N/A")
        data_context["prediction_reasoning"] = prediction_data.get("reasoning", "No reasoning provided.")
        # Format lists into multi-line strings for the prompt
        pos_factors = prediction_data.get("key_positive_factors", [])
        data_context["prediction_positive_factors_text"] = "\n".join([f"- {f}" for f in pos_factors]) if pos_factors else "None listed."
        neg_factors = prediction_data.get("key_negative_factors", [])
        data_context["prediction_negative_factors_text"] = "\n".join([f"- {f}" for f in neg_factors]) if neg_factors else "None listed."
        limitations = prediction_data.get("data_limitations", [])
        data_context["prediction_data_limitations_text"] = "\n".join([f"- {f}" for f in limitations]) if limitations else "None stated."
        data_context["prediction_error"] = "None" # No error in the prediction process itself

    logger.info(f"Investment prediction context prepared. Recommendation: {data_context['prediction_recommendation']} - Confidence: {data_context['prediction_confidence']}")

    # F. Source URLs
    if not all_urls_collected:
        data_context["all_source_urls_list_text"] = "No specific source URLs were collected."
    else:
        data_context["all_source_urls_list_text"] = "\n".join([f"- {url}" for url in sorted(list(all_urls_collected))])

    # Add the final consolidated URL list to the main state for direct access later
    data_context["consolidated_urls_for_state"] = sorted(list(all_urls_collected))
    logger.info(f"Data context prepared for LLM report. Collected {len(all_urls_collected)} unique URLs.")

    return data_context

def generate_llm_based_report_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates the final report using an LLM based on all aggregated data.
    """
    company_name = state.get("company_name", "N/A")
    logger.info(f"--- Generating LLM-based Final Report for: {company_name} ---")

    if not report_generation_llm:
        logger.error("[ReportNode] Report Generation LLM not available. Cannot generate report.")
        final_urls = state.get("consolidated_source_urls", []) # Try get URLs from state if already prepared
        return {
            "final_report_markdown": "Error: Report generation LLM is not initialized.",
            "consolidated_source_urls": final_urls
        }


    # 1. Prepare the full context for the LLM prompt
    try:
        report_context_data = prepare_data_for_llm_report_generation(state)
        prepared_urls = report_context_data.pop("consolidated_urls_for_state", [])
        final_llm_prompt = LLM_REPORT_GENERATION_PROMPT_TEMPLATE.format(**report_context_data)
        # Be mindful of token limits for the LLM
        # Add truncation logic if final_llm_prompt is too long
        max_prompt_length = 200000 # Example, adjust based on your model's context window
        if len(final_llm_prompt) > max_prompt_length:
            logger.warning(f"Report generation prompt for {company_name} is too long ({len(final_llm_prompt)} chars), truncating context sections.")
            # Implement more intelligent truncation if needed, e.g., shorten individual context pieces
            # For now, a simple overall truncation:
            final_llm_prompt = final_llm_prompt[:max_prompt_length-500] + "\n\n[CONTEXT TRUNCATED DUE TO LENGTH]"
        
        logger.debug(f"[ReportNode] LLM Prompt for Report Generation (first 500 chars):\n{final_llm_prompt[:500]}...")

    except Exception as e:
        logger.error(f"[ReportNode] Error preparing data for LLM report generation for {company_name}: {e}", exc_info=True)
        return {
            "final_report": f"Error: Could not prepare data for report generation. Details: {e}",
            "all_source_urls": [] # Or try to extract from state if possible
        }

    # 2. Call the LLM to generate the report
    try:
        messages = [HumanMessage(content=final_llm_prompt)]
        ai_response = report_generation_llm.invoke(messages)
        generated_report_markdown = ai_response.content

        # Basic check to see if LLM followed instructions (e.g., didn't just say "Okay, I will generate...")
        if len(generated_report_markdown) < 500 or not generated_report_markdown.strip().startswith("#"): # Heuristic
            logger.warning(f"[ReportNode] LLM response for report seems too short or not Markdown. Response: {generated_report_markdown[:200]}")
            # Potentially retry or fallback to a simpler template
            # For now, we'll use what it gave.

        logger.info(f"LLM-based report generated for {company_name}. Length: {len(generated_report_markdown)} chars.")
        
        return {
            "final_report_markdown": generated_report_markdown,
            "consolidated_source_urls": prepared_urls # Use URLs prepared earlier
        }

    except Exception as e:
        logger.error(f"[ReportNode] LLM call failed during report generation for {company_name}: {e}", exc_info=True)
        return {
            "final_report_markdown": f"Error: LLM call failed during report generation. Details: {e}",
            "consolidated_source_urls": prepared_urls # Return URLs even if LLM failed
        }