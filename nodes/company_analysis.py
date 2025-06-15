# agents/company_analysis.py
import json
import re
from typing import Dict, Any, List, Optional
from state import AgentState ,StockPredictionOutput
from prompts import STOCK_PREDICTION_PROMPT_TEMPLATE 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.messages import HumanMessage
import config 
from config import logger

try:
    prediction_llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL_NAME_PRO, 
        temperature=0.1, 
        convert_system_message_to_human=True 
    )
    logger.info("Prediction LLM initialized.")
except Exception as e:
    logger.error(f"Failed to initialize prediction LLM: {e}")
    prediction_llm = None


def format_prediction_context(state: AgentState) -> str:
    """
    Safely extracts and formats data from AgentState for the prediction prompt.
    Assumes wrapper nodes for subgraphs place their main data payload under a consistent key
    (e.g., 'summaries', 'data', 'competitors') and always include an 'error' key.
    """
    company_name_from_state = state.get("company_name", "The Specified Company")
    logger.info(f"Formatting prediction context for: {company_name_from_state}")

    # 1. Financial Report Summaries
    # Expected state['report_summaries_output'] = {"summaries": {"url1": "text1", ...}, "urls": ["url1", ...], "error": "message or None"}
    report_data = state.get("report_summaries_output", {})
    report_error = report_data.get("error")
    summaries_dict = report_data.get("document_summaries", {}) # This is the dict {'URL': 'Summary'}

    if report_error:
        report_summaries_text = f"Error retrieving financial report summaries: {report_error}"
    elif not summaries_dict:
        report_summaries_text = "No financial report summaries were found or are available."
    else:
        texts = [f"Summary from Financial Document (Source URL: {url}):\n{summary}" for url, summary in summaries_dict.items() if summary]
        report_summaries_text = "\n\n---\n\n".join(texts) if texts else "Financial report summaries dictionary was present but contained no text content."
    logger.debug(f"Formatted report_summaries_text (first 100 chars): {report_summaries_text[:100]}")


    # 2. Sentiment Analysis
    # Accessing state['sentiment_analysis_output'] which should be SentimentAnalysisPayload
    sentiment_data = state.get("sentiment_analysis_output", {})
    sentiment_error = sentiment_data.get("error")
    sentiment_data_dict = sentiment_data.get("sentiment_data",{})

    if sentiment_error:
        sentiment_value_for_prompt = "Error"
        sentiment_report_text_for_prompt = f"Error during sentiment analysis: {sentiment_error}"
        logger.warning(f"Error present in sentiment_analysis_output: {sentiment_error}")
    elif not sentiment_data_dict: # If sentiment key itself is missing or None/empty
        sentiment_value_for_prompt = "Unavailable"
        sentiment_report_text_for_prompt = "Sentiment analysis data is not available."
        logger.info("sentiment_analysis_output['sentiment_data'] is missing or empty.")
    else:
        sentiment_value_for_prompt = sentiment_data_dict.get("sentiment", "N/A")
        sentiment_report_text_for_prompt = sentiment_data_dict.get("detailed_sentiment_report", "No detailed sentiment report was provided.")
        if sentiment_value_for_prompt == "N/A":
            logger.warning("Sentiment value was 'N/A' within sentiment_analysis_output['sentiment_data']")
    logger.debug(f"Formatted sentiment_value_for_prompt: {sentiment_value_for_prompt}")


    # 3. Current Stock Data
    # Expected state['stock_data_output'] = {"data": {"ticker": "AAPL", ...}, "error": "message or None"}
    stock_data_container = state.get("stock_data_output", {})
    stock_error = stock_data_container.get("error")
    actual_stock_data_dict = stock_data_container.get("stock_metrics", {}) # The dict with ticker, price, etc.

    if stock_error:
        stock_data_json_for_prompt = json.dumps({"error": f"Error retrieving stock data: {stock_error}"})
        logger.warning(f"Error present in stock_data_output: {stock_error}")
    elif not actual_stock_data_dict: # If the 'data' dict is empty or missing
        stock_data_json_for_prompt = json.dumps({"message": "Stock data is unavailable or not retrieved."})
        logger.info("stock_data_output['stock_metrics'] is missing or empty.")
    else:
        stock_data_json_for_prompt = json.dumps(actual_stock_data_dict, indent=2)
    logger.debug(f"Formatted stock_data_json_for_prompt (first 100 chars): {stock_data_json_for_prompt[:100]}")


    # 4. Competitor Information
    # Expected state['competitor_info_output'] = {"competitors": [{"name": "X", "description": "Y"}, ...], "error": "message or None"}
    competitor_data = state.get("competitor_info_output", {})
    competitor_error = competitor_data.get("error")
    competitors_list = competitor_data.get("competitors_list", []) # List of competitor dicts

    if competitor_error:
        competitor_info_text_for_prompt = f"Error retrieving competitor information: {competitor_error}"
        logger.warning(f"Error present in competitor_info_output: {competitor_error}")
    elif not competitors_list: 
        competitor_info_text_for_prompt = "No competitor information is available or no competitors were listed."
        logger.info("competitor_info_output['competitors_list'] is missing or empty.")
    else:
        competitor_details_list = []
        for comp in competitors_list:
            name = comp.get("name", "N/A")
            description = comp.get("description", "No description provided.")
            competitor_details_list.append(f"- {name}: {description}")
        competitor_info_text_for_prompt = "\n".join(competitor_details_list)
    logger.debug(f"Formatted competitor_info_text_for_prompt (first 100 chars): {competitor_info_text_for_prompt[:100]}")

    # Construct the full prompt using the centrally defined STOCK_PREDICTION_PROMPT_TEMPLATE
    try:
        final_prompt_for_llm = STOCK_PREDICTION_PROMPT_TEMPLATE.format(
            company_name=company_name_from_state,
            report_summaries_text=report_summaries_text or "NA",
            sentiment_value=sentiment_value_for_prompt or "NA",
            sentiment_report_text=sentiment_report_text_for_prompt or "NA",
            stock_data_json=stock_data_json_for_prompt or "{}",
            competitor_info_text=competitor_info_text_for_prompt or "NA",
        )
    except KeyError as e:
        logger.error(f"KeyError formatting STOCK_PREDICTION_PROMPT_TEMPLATE: Missing key {e}. Check prompt template and available data.", exc_info=True)
        return f"Error: Could not format prediction prompt due to missing key: {e}. Please check data availability for all sections."
    except Exception as e:
        logger.error(f"Unexpected error during prompt formatting: {e}", exc_info=True)
        return f"Error: An unexpected error occurred during prompt formatting: {e}."


    return final_prompt_for_llm

def predict_stock_price_node(state: AgentState) -> Dict[str, StockPredictionOutput]:
    """
    Node to predict stock price movement (Buy/Sell/Hold) based on aggregated data.
    Ensures the output structure matches StockPredictionOutput.
    """

    default_error_output: StockPredictionOutput = {
        "recommendation": "UNCERTAIN", "error": "Prediction failed",
        "confidence": "N/A", "reasoning": "Failed to generate prediction.",
        "key_positive_factors": [], "key_negative_factors": [], "data_limitations": []
    }

    if not prediction_llm:
        logger.error("[PredictionNode] LLM not available.")
        default_error_output["error"] = "Prediction LLM not initialized."
        return {"prediction_output": default_error_output}

    company_name = state.get("company_name", "The Company")
    logger.info(f"--- Generating Stock Prediction for: {company_name} ---")

    # 1. Format context
    try:
        formatted_prompt_text = format_prediction_context(state)
        if formatted_prompt_text.startswith("Error:"): 
             raise ValueError(formatted_prompt_text) 
        logger.debug(f"[PredictionNode] Formatted prompt length: {len(formatted_prompt_text)}")
    except Exception as e:
        logger.error(f"[PredictionNode] Error formatting context: {e}", exc_info=True)
        default_error_output["error"] = f"Failed to format prediction context: {e}"
        return {"prediction_output": default_error_output}

    # 2. Call LLM
    try:
        messages = [HumanMessage(content=formatted_prompt_text)]
        ai_response = prediction_llm.invoke(messages)
        response_content = ai_response.content
        logger.debug(f"[PredictionNode] LLM raw response: {response_content}")
    except Exception as e:
        logger.error(f"[PredictionNode] LLM call failed: {e}", exc_info=True)
        default_error_output["error"] = f"LLM call failed: {e}"
        return {"prediction_output": default_error_output}

    # 3. Parse and VALIDATE LLM Output, ensuring all keys are present
    try:
        # Robust JSON extraction
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_content, re.DOTALL | re.IGNORECASE)
        if match: json_str = match.group(1)
        else:
            json_start = response_content.find('{'); json_end = response_content.rfind('}') + 1
            if json_start != -1 and json_end != -1 and json_end > json_start: json_str = response_content[json_start:json_end]
            else: raise ValueError("JSON object not found in LLM response.")

        parsed_prediction: Dict = json.loads(json_str) # Parse into generic Dict first

        # **Validation and Defaulting Section**
        recommendation = parsed_prediction.get("recommendation")
        reasoning = parsed_prediction.get("reasoning")

        if not recommendation or not reasoning:
            raise ValueError("Prediction JSON missing mandatory 'recommendation' or 'reasoning'.")

        allowed_recs = ["BUY", "SELL", "HOLD", "UNCERTAIN"]
        if recommendation not in allowed_recs:
            logger.warning(f"LLM recommendation '{recommendation}' invalid. Defaulting to UNCERTAIN.")
            recommendation = "UNCERTAIN"
            reasoning += " (LLM provided an invalid recommendation category)"

        # Get optional fields with defaults
        confidence = parsed_prediction.get("confidence", "N/A") 
        # Ensure list fields are actually lists, default to empty list if missing or wrong type
        key_pos_factors = parsed_prediction.get("key_positive_factors", [])
        if not isinstance(key_pos_factors, list):
             logger.warning("key_positive_factors from LLM was not a list, defaulting to empty list.")
             key_pos_factors = []
             
        key_neg_factors = parsed_prediction.get("key_negative_factors", [])
        if not isinstance(key_neg_factors, list):
             logger.warning("key_negative_factors from LLM was not a list, defaulting to empty list.")
             key_neg_factors = []
             
        data_limits = parsed_prediction.get("data_limitations", [])
        if not isinstance(data_limits, list):
             logger.warning("data_limitations from LLM was not a list, defaulting to empty list.")
             data_limits = []

        # Construct the final validated output dictionary matching StockPredictionOutput
        final_prediction_output: StockPredictionOutput = {
            "recommendation": recommendation,
            "confidence": confidence,
            "reasoning": reasoning,
            "key_positive_factors": key_pos_factors,
            "key_negative_factors": key_neg_factors,
            "data_limitations": data_limits,
            "error": None 
        }

        logger.info(f"[PredictionNode] Successfully generated prediction for {company_name}: {recommendation}")
        return {"prediction_output": final_prediction_output}

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"[PredictionNode] Failed to parse/validate prediction JSON: {e}. Response: {response_content}", exc_info=True)
        default_error_output["error"] = f"Failed to parse/validate prediction: {e}"
        default_error_output["reasoning"] = f"Could not parse/validate LLM output: {response_content}" 
        return {"prediction_output": default_error_output}
    except Exception as e:
        logger.error(f"[PredictionNode] Unexpected error processing LLM response: {e}", exc_info=True)
        default_error_output["error"] = f"Unexpected error processing prediction: {e}"
        return {"prediction_output": default_error_output}