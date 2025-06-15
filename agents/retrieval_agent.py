# agents/retrieval_agent.py
import logging
import warnings
from datetime import date

# Third-party imports
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI

# Local imports
from config.config import GEMINI_MODEL_NAME, TEMPERATURE
from core.prompts import REACT_RETRIEVAL_SYSTEM_PROMPT
from core.tools import (
    search_duck_duck_go,
    check_pdf_url_validity,
    get_web_page_content,
    google_search
)
from config import config
# Ignore all warnings
warnings.filterwarnings("ignore")

logger = config.logger

# --- List of Tools for this specific agent ---
retrieval_tools = [
    search_duck_duck_go,
    google_search,
    check_pdf_url_validity,
    get_web_page_content,
]

# --- Prepare the System Prompt ---
current_date_str = date.today().isoformat()
formatted_system_prompt = REACT_RETRIEVAL_SYSTEM_PROMPT.format(
    current_date=current_date_str
)

MAX_RETRIEVAL_ATTEMPTS = 2

# --- Initialize the LLM ---
try:
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        temperature=TEMPERATURE,
        convert_system_message_to_human=True 
    )
except Exception as e:
    logger.exception("Failed to initialize LLM for retrieval agent.")
    raise

# --- Create the ReAct Agent Graph ---
try:
    retrieval_agent_runnable = create_react_agent(
        model=llm,
        tools=retrieval_tools,
        prompt=formatted_system_prompt,
        debug=True,
    )
    logger.info("Standalone ReAct Retrieval Agent graph created successfully.")
except Exception as e:
    logger.exception("Failed to create ReAct agent graph.")
    raise
