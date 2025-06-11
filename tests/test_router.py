# tests/test_router.py
import os
import sys
import logging
# --- Add project root to sys.path ---
# This assumes test_router.py is in /financial_advisor_tool/tests/
# TODO: Remove sys.path manipulation block below.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------
# Now you can import modules from the project root
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
import config # This will run the logging configuration
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
from state import AgentState
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
from agents.router_agent import RouterAgent
from google import genai

# Get the logger configured in config.py
logger = logging.getLogger(__name__) # Get logger for this module

def run_test():
    """Runs a test sequence for the RouterAgent."""
    logger.info("--- Starting RouterAgent Test ---")

    # 1. Create the Gemini Client
    try:
        # Configure using environment variables automatically (if set)
        # Or explicitly pass api_key if needed and not using env vars
        # client = genai.Client(api_key=config.GEMINI_API_KEY)
        client = genai.Client() # Tries to use env vars GOOGLE_API_KEY or Vertex settings
        logger.info("Gemini Client created.")
    except Exception as e:
        logger.error(f"Failed to create Gemini Client: {e}")
        return

    # 2. Instantiate the RouterAgent, passing the client and model name
    try:
        # Get model name from config
        model_name = config.GEMINI_MODEL_NAME
        router_agent = RouterAgent(client=client, model_name=model_name)
        logger.info(f"RouterAgent instantiated for model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to instantiate RouterAgent: {e}")
        return

    # 3. Define Test Queries
    test_queries = [
        "Tell me about Apple's latest earnings report.",
        "What are the current trends in the renewable energy sector?",
        "Analyze Microsoft stock.",
        "Give me an overview of the semiconductor industry.",
        "Is Google a good investment?",
        "How is the AI market performing?",
        "What is the P/E ratio for NVDA?",
        "Compare oil and gas companies.",
        "What stocks should I buy today?",
        "What happened in the markets?",
    ]

    # 6. Run Agent for each query (same as before)
    for query in test_queries:
        logger.info(f"\nTesting query: '{query}'")
        initial_state = AgentState(
            query=query, query_type=None, company_name=None, sector_name=None,
            workflow_plan=None, document_urls=None, document_summaries=None,
            sentiment_summary=None, stock_prediction=None, stock_data=None,
            competitors=None, sector_analysis_summary=None, candidate_companies=None,
            selected_companies=None, companies_to_analyze=[], current_company_in_loop=None,
            final_output=None, error_message=None
        )
        try:
            result_updates = router_agent.run(initial_state)
            logger.info(f"RouterAgent Result: {result_updates}")
        except Exception as e:
            logger.error(f"An error occurred while running the agent for query '{query}': {e}", exc_info=True)

    logger.info("\n--- RouterAgent Test Finished ---")

def test_router_agent():
    run_test()
    
if __name__ == "__main__":
    run_test()