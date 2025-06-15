# Subgraphs Documentation

This document provides a detailed, code-verified overview of the subgraphs used in the Financial Analyst project. Each subgraph is a modular, stateful, and reusable component built with LangGraph that performs a specific, targeted task within the larger financial analysis workflow.

## Core Concepts

- **Stateful Execution**: Each subgraph operates on its own state object (`TypedDict`), which carries information through the graph's nodes. This state includes inputs, intermediate results, and control flow variables (like retry counters).
- **ReAct Agents**: Most subgraphs employ a ReAct (Reasoning and Acting) agent. This agent uses a Large Language Model (LLM) to reason about a task and decide which tool to use (e.g., `google_search`, `get_news_articles`) to accomplish it.
- **Conditional Logic**: Subgraphs use conditional edges to manage their execution flow. This typically includes a loop for retrying a failed operation and routing to an end state upon success or final failure.
- **Structured Output**: Agents are prompted to return data in a structured JSON format, which is then parsed and validated by a dedicated node in the graph.

---

## Subgraph Details

### 1. Competitor Analysis Subgraph
- **File**: `subgraphs/competetior_anaysis_subgraph.py`
- **Purpose**: To identify the top competitors for a given company.
- **Workflow**:
    1.  A ReAct agent is tasked with searching the web to find competitors.
    2.  The agent uses tools like `google_search` to find relevant information.
    3.  A parsing node extracts the list of competitors and their descriptions from the agent's final output.
    4.  Includes retry logic if the initial search fails to produce a valid, structured list.
- **Output**: A list of competitor names and descriptions.

### 2. Retrieval Summarization Subgraph
- **File**: `subgraphs/retrieval_summarization_graph.py`
- **Purpose**: To retrieve relevant documents for a company and generate a concise summary.
- **Workflow**:
    1.  A retrieval agent fetches documents based on the company name.
    2.  The retrieved documents are passed to a summarization node.
    3.  The summarization node generates a summary of the content.
    4.  Conditional logic handles cases where retrieval or summarization fails.
- **Output**: A text summary of relevant documents.

### 3. Sector Key Players Subgraph
- **File**: `subgraphs/sector_keyplayers_subgraph.py`
- **Purpose**: To identify the most significant companies (key players) within a specific market sector.
- **Workflow**:
    1.  A ReAct agent searches the web for key players in the specified sector.
    2.  It uses search tools to gather information and identify top companies.
    3.  A parsing node validates and extracts the structured list of key players.
    4.  Includes retry logic to ensure a comprehensive list is generated.
- **Output**: A structured list of key players with names, descriptions, and source URLs.

### 4. Sector Market Data Subgraph
- **File**: `subgraphs/sector_market_data_subgraph.py`
- **Purpose**: To collect high-level market data for a given sector.
- **Workflow**:
    1.  A ReAct agent is prompted to find key market data points (market size, CAGR, segments, geographies, drivers, challenges).
    2.  The agent uses web search tools to find reports and articles containing this data.
    3.  A parsing node extracts the structured market data from the agent's response.
    4.  Includes a retry mechanism if the initial search is unsuccessful.
- **Output**: A JSON object containing the requested market data points.

### 5. Sector Sentiment Subgraph
- **File**: `subgraphs/sector_sentiment_subgraph.py`
- **Purpose**: To analyze the overall market sentiment for a specific sector.
- **Workflow**:
    1.  A ReAct agent searches for recent news, articles, and financial reports related to the sector.
    2.  It analyzes the information to determine the prevailing sentiment (positive, negative, neutral).
    3.  A parsing node extracts the structured sentiment analysis, including key themes and reasoning.
    4.  Includes retry logic.
- **Output**: A JSON object detailing the sector's sentiment, key news themes, and source URLs.

### 6. Sector Trends Subgraph
- **File**: `subgraphs/sector_trends_subgraph.py`
- **Purpose**: To identify key trends, innovations, challenges, and opportunities within a sector.
- **Workflow**:
    1.  A ReAct agent conducts research using web search tools.
    2.  The agent is prompted to look for emerging technologies, market shifts, and potential hurdles.
    3.  A parsing node extracts this information into a structured format.
    4.  Includes retry logic for comprehensive research.
- **Output**: A JSON object detailing major trends, innovations, challenges, and opportunities.

### 7. Company Sentiment Subgraph
- **File**: `subgraphs/sentiment_agent_subgraph.py`
- **Purpose**: To analyze the market sentiment for a *specific company*.
- **Workflow**:
    1.  A ReAct agent uses news and web search tools to find recent information about the company.
    2.  It synthesizes the findings to produce a detailed sentiment report.
    3.  A parsing node validates and extracts the final sentiment analysis.
    4.  Includes retry logic.
- **Output**: A JSON object containing the company's sentiment and a detailed report.

### 8. Stock Data Subgraph
- **File**: `subgraphs/stock_agent_subgraph.py`
- **Purpose**: To fetch key stock data for a specific company.
- **Workflow**:
    1.  A ReAct agent is tasked with finding stock information (price, market cap, P/E ratio, etc.).
    2.  It uses web search tools, prioritizing reliable financial sources.
    3.  A parsing node extracts the structured stock data and the source URLs.
    4.  Includes retry logic to handle cases where data is not immediately found.
- **Output**: A JSON object containing the stock data and a list of source URLs.
