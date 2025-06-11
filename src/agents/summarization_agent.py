import os
from typing import Dict, Any
from dotenv import load_dotenv
import sys
# Import project components
from src.core.state import RetrievalSummarizationState
from src.core import tools # Import the tools module
from src.core.tools import FinancialSummary # Import the response model
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

doenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)
api_key = os.getenv("GOOGLE_API_KEY")
from google import genai
from src.config import logger


MAX_SUMMARIES_TO_PROCESS = 3

# Modify function to accept an optional client, mainly for testing
def summarize_documents_node(state: RetrievalSummarizationState) -> Dict[str, Any]:
    """
    LangGraph node function to summarize documents found in the state.
    Accepts an optional Gemini client, otherwise uses default from tool.

    Args:
        state (AgentState): The current graph state, expecting 'document_urls'.
        client (Optional[genai.Client]): An initialized client to pass to the tool.

    Returns:
        Dict[str, Any]: A dictionary containing the update for 'document_summaries'.
    """
    urls_to_summarize  = state.get('document_urls')
    company_name = state.get('company_name', 'Unknown Company')
    summaries: Dict[str, str] = {}
    processed_count = 0
    error_count = 0
    skipped_count = 0

    if not urls_to_summarize :
        logger.warning(f"Summarization Node: No document URLs found for {company_name}. Skipping.")
        return {"document_summaries": {}}
    if not isinstance(urls_to_summarize , list):
        logger.error(f"Summarization Node: 'document_urls' is not a list for {company_name}.")
        return {"document_summaries": {}}

    client = genai.Client()

    logger.info(f"Summarization Node starting for {company_name}. Processing up to {MAX_SUMMARIES_TO_PROCESS} of {len(urls_to_summarize)} URL(s).")
    urls_to_process = urls_to_summarize[:MAX_SUMMARIES_TO_PROCESS]

    for i, url in enumerate(urls_to_process):
        logger.info(f"  Summarizing URL {i+1}/{len(urls_to_process)}: {url}")
        try:
            # --- Pass the client explicitly to the tool ---
            summary_result: FinancialSummary = tools.summarize_pdf_document_finance(
                doc_url=url,
                client=client 
            )

            if summary_result.success and summary_result.summary:
                 logger.info(f"-> Successfully summarized: {url}")
                 summaries[url] = summary_result.summary
                 processed_count += 1
            elif summary_result.error:
                  logger.warning(f"-> Failed to summarize {url}: {summary_result.error}")
                  summaries[url] = f"Error: Summarization failed - {summary_result.error}"
                  error_count += 1
            else:
                  logger.warning(f"-> Summarization for {url} completed but returned no summary or error.")
                  summaries[url] = "Error: Summarization completed without summary."
                  error_count += 1

        except Exception as e:
            logger.error(f"    -> Unexpected exception summarizing {url}: {e}", exc_info=True)
            summaries[url] = f"Error: Unexpected exception during summarization - {e}"
            error_count += 1


    skipped_count = len(urls_to_summarize) - len(urls_to_process)
    total_attempted = processed_count + error_count
    logger.info(f"Summarization Node finished for {company_name}. Attempted: {total_attempted}. Success: {processed_count}. Errors: {error_count}. Skipped: {skipped_count}")

    return {"document_summaries": summaries}