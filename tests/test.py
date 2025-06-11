# Run this in a script or notebook, not as a unit test
# TODO: Update this import to reflect the new 'src.' structure (e.g., from src.core.tools import ..., from src import config, from src.agents... import ...)
from tools import search_duck_duck_go, google_search,get_news_articles,check_pdf_url_validity,get_stock_data

# print("Running test for DuckDuckGo and Google search tools...")
# # Test DuckDuckGo search
# print("Testing DuckDuckGo search...")
# print(search_duck_duck_go("Apple Annual Report 2024"))
# # Test Google search
# print("Testing Google search...")
# print(google_search("Apple Annual Report 2024"))
# # Test news articles retrieval
# print("Testing news articles retrieval...")
# print(get_news_articles("Apple Annual Report 2024"))

# # Test PDF URL validity check
# print("Testing PDF URL validity check...")
# print(check_pdf_url_validity("https://microsoft.gcs-web.com/static-files/1c864583-06f7-40cc-a94d-d11400c83cc8"))

# Test stock data retrieval
print("Testing stock data retrieval...")
print(get_stock_data("AAPL"))
print("Testing stock data retrieval for a non-existent stock...")
print(get_stock_data("APPLE"))
