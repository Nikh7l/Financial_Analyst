# config.py
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file located in the project root
# Ensure this runs before accessing os.getenv
dotenv_path = os.path.join(os.path.dirname(__file__), '.env') # Assumes config.py is in root
load_dotenv(dotenv_path=dotenv_path)

# --- Logging Configuration ---
LOG_LEVEL_STR = os.getenv('LOG_LEVEL', 'INFO').upper() # Load level from env, default INFO
LOG_LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
LOG_LEVEL = LOG_LEVEL_MAP.get(LOG_LEVEL_STR, logging.DEBUG)
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Configure the root logger. Should run only once.
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__) # Get a logger for config file itself if needed
logger.info(f"Logging configured with level: {logging.getLevelName(LOG_LEVEL)}")

# --- Add file logging ---
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "financial_advisor.log")
file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
file_handler.setLevel(LOG_LEVEL)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logging.getLogger().addHandler(file_handler)
logger.info(f"Logging to file: {LOG_FILE_PATH}")

# --- Gemini API Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY environment variable not found. Gemini Client might fail.")
# Model name used by agents and PDF summarization tool
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-preview-04-17")
GEMINI_MODEL_NAME_PRO = "gemini-2.5-pro-exp-03-25"
# Default temperature for LLM generation (can be overridden)
# Load as float, handle potential ValueError
try:
    TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
except ValueError:
    logger.warning("Invalid LLM_TEMPERATURE in .env, using default 0.2")
    TEMPERATURE = 0.2

logger.info(f"Using Gemini Model: {GEMINI_MODEL_NAME}")
logger.info(f"Default LLM Temperature: {TEMPERATURE}")


# --- Google Search API Configuration ---
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID") # Custom Search Engine ID
if not GOOGLE_SEARCH_API_KEY or not GOOGLE_CSE_ID:
    logger.warning("GOOGLE_SEARCH_API_KEY or GOOGLE_CSE_ID environment variable not found. Google Search tool will fail.")


# --- Tool Configuration ---
# Default number of results for search tools
try:
    SEARCH_RESULTS_LIMIT = int(os.getenv("SEARCH_RESULTS_LIMIT", "5"))
except ValueError:
    logger.warning("Invalid SEARCH_RESULTS_LIMIT in .env, using default 5")
    SEARCH_RESULTS_LIMIT = 5

# Default timeouts for network requests within tools (in seconds)
try:
    URL_TIMEOUT = int(os.getenv("URL_TIMEOUT", "15")) # For general requests, HEAD checks, API calls
except ValueError:
    logger.warning("Invalid URL_TIMEOUT in .env, using default 15")
    URL_TIMEOUT = 15

try:
    PDF_DOWNLOAD_TIMEOUT = int(os.getenv("PDF_DOWNLOAD_TIMEOUT", "60")) # For potentially large PDF downloads
except ValueError:
    logger.warning("Invalid PDF_DOWNLOAD_TIMEOUT in .env, using default 60")
    PDF_DOWNLOAD_TIMEOUT = 60

logger.info(f"Search results limit: {SEARCH_RESULTS_LIMIT}")
logger.info(f"URL request timeout: {URL_TIMEOUT}s")
logger.info(f"PDF download timeout: {PDF_DOWNLOAD_TIMEOUT}s")

# --- Add other application-wide configurations as needed ---
# MAX_WEB_CONTENT_CHARS = 4000 # Example if you want to configure this
VALIDATION_CANDIDATE_LIMIT = 20 # Limit for validation candidates in the PDF tool
MAX_REPORTS_TO_FETCH = 3 # Limit for the number of reports to fetch in the retrieval agent

NEWS_API_KEY = "7d2d56624e68498a97a6303723ebbb61"
NEWS_API_PAGE_SIZE = 20 # Number of articles to fetch per request
NEWS_API_DEFAULT_DAYS_AGO = 14 # Default number of days ago for news articles
