# tests/test_planner.py
import os
import sys
import logging
import pytest # Using pytest features like fixtures and parametrize
from dotenv import load_dotenv

# --- Add project root to sys.path ---
# TODO: Remove sys.path manipulation block below.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# Import necessary components from your project
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
import config # This will run the logging configuration
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
from state import AgentState
# Import the agent and the workflow constants it defines
# TODO: Update this import. Path may need significant revision if 'Financial_advisor.old.planner_agent' was moved/refactored.
from Financial_advisor.old.planner_agent import PlannerAgent, COMPANY_WORKFLOW, SECTOR_WORKFLOW

# Import Gemini library
from google import genai

# Get the logger configured in config.py
logger = logging.getLogger(__name__)

# --- Test Setup Fixture ---
@pytest.fixture(scope="module") # Run once for all tests in this file
def configured_client():
    """Fixture to set up the Gemini client once."""
    logger.info("--- Setting up Gemini Client for PlannerAgent tests ---")
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found, skipping PlannerAgent tests.")

    try:
        # Configure client using environment variables or explicit key
        client = genai.Client()
        logger.info("Gemini Client created for tests.")
        return client
    except Exception as e:
        pytest.fail(f"Failed to create Gemini Client: {e}")

# --- Test Cases using parametrize ---
@pytest.mark.parametrize("query_type_input, entity_name_input, expected_plan, expect_error", [
    # Valid Cases
    ("company", "Apple", COMPANY_WORKFLOW, False),
    ("sector", "Renewable Energy", SECTOR_WORKFLOW, False),
    ("unknown", "Some general query", [], False), # Expect empty plan, no error

    # Edge/Error Cases
    (None, "Test None", [], True), # Missing query_type
    ("", "Test Empty String", [], True), # Empty query_type (should be treated as invalid)
    ("invalid_type", "Test Invalid", [], True), # Unexpected query_type value
])
def test_planner_agent_logic(configured_client, query_type_input, entity_name_input, expected_plan, expect_error):
    """Tests the PlannerAgent's rule-based logic for different query types."""
    logger.info(f"\nTesting Planner with query_type='{query_type_input}'")

    # Instantiate the PlannerAgent
    # It needs the client and model name for __init__, though run() doesn't use them here
    try:
        planner_agent = PlannerAgent(client=configured_client, model_name=config.GEMINI_MODEL_NAME)
    except Exception as e:
         pytest.fail(f"Failed to instantiate PlannerAgent: {e}")

    # Create a mock AgentState for input
    # Only query_type is strictly needed by the current planner logic,
    # but include others with default values for robustness.
    test_state = AgentState(
        query="Dummy query for testing planner", # Not used by planner
        query_type=query_type_input,
        # Set one entity name for logging clarity in the agent
        company_name=entity_name_input if query_type_input == "company" else None,
        sector_name=entity_name_input if query_type_input == "sector" else None,
        workflow_plan=None, # Planner should overwrite this
        document_urls=None,
        document_summaries=None,
        sentiment_summary=None,
        stock_prediction=None,
        stock_data=None,
        competitors=None,
        sector_analysis_summary=None,
        candidate_companies=None,
        selected_companies=None,
        companies_to_analyze=[],
        current_company_in_loop=None,
        final_output=None,
        error_message=None
    )

    # Run the planner agent
    try:
        result_updates = planner_agent.run(test_state)
        logger.info(f"PlannerAgent Result: {result_updates}")

        # --- Assertions ---
        # Check if the workflow plan matches expectations
        assert result_updates.get("workflow_plan") == expected_plan, \
            f"Expected plan {expected_plan} but got {result_updates.get('workflow_plan')}"

        # Check if an error message was expected or not
        error_message = result_updates.get("error_message")
        if expect_error:
            assert error_message is not None and error_message != "", \
                f"Expected an error message for input type '{query_type_input}', but got none or empty."
        else:
            assert error_message is None or error_message == "", \
                f"Did not expect an error message for input type '{query_type_input}', but got: '{error_message}'"

    except Exception as e:
         pytest.fail(f"PlannerAgent run failed unexpectedly for query_type '{query_type_input}': {e}")

# Optional: Add a main block if you want to run it directly sometimes
# (though 'pytest' command is the standard way)
# if __name__ == "__main__":
#     # You could call pytest programmatically here if needed,
#     # or run a simplified manual test sequence.
#     print("Run this test using 'pytest'")