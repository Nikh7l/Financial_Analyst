# State Management Documentation

## Overview

State management in this project is built on the `StateGraph` concept from LangGraph. Instead of a single, monolithic state object, the architecture uses multiple, specialized state definitions, each tailored to a specific subgraph or the main application graph.

These state definitions are created using Python's `TypedDict` and are defined in `core/state.py`.

---

## Subgraph State

Each subgraph has its own dedicated `TypedDict` that defines its internal state. This allows each subgraph to manage its own inputs, outputs, and control flow independently. While the specific data fields vary, most subgraph states share a common structure for managing the execution flow.

### Common Subgraph State Fields:

- **`company_name` or `sector_name` (str)**: The primary input for the subgraph's analysis.
- **`messages` (List[BaseMessage])**: The history of the conversation with the ReAct agent. This list grows as the agent thinks, calls tools, and generates responses.
- **`attempt` (int)**: A counter that tracks the number of times the subgraph has tried to execute. This is used for implementing retry logic.
- **`max_attempts` (int)**: The maximum number of retries allowed before failing.
- **`subgraph_error` (Optional[str])**: A field to store an error message if the subgraph fails.
- **`_route_decision` (Optional[str])**: A special field used to control conditional routing within the graph. After a node executes, it can set this field to `"success"`, `"retry"`, or `"fail"` to direct the graph to the next appropriate step.
- **Output Fields (e.g., `competitors`, `stock_data`)**: Each subgraph state has a specific field to store the final, parsed result of its analysis.

### Example Subgraph States:

- `CompetitorSubgraphState`
- `SectorSentimentSubgraphState`
- `StockDataSubgraphState`
- `SentimentSubgraphState`
- `RetrievalSummarizationState`

---

## Main Application State (`AgentState`)

There is a top-level state definition called `AgentState`, which serves as the main state for the entire application graph. Its primary role is to orchestrate the subgraphs and aggregate their results.

### Key `AgentState` Fields:

- **Initial Inputs**: `query`, `query_type`, `company_name`, `sector_name`. These fields are populated by the `RouterAgent` at the beginning of the process.
- **Subgraph Output Payloads**: The `AgentState` contains fields to hold the results from each of the company and sector analysis subgraphs (e.g., `report_summaries_output`, `sentiment_analysis_output`, `sector_key_players_output`). These fields are populated as each subgraph completes its execution.
- **Final Report Fields**: `final_report_markdown` and `final_sector_report_markdown` are used to store the final, consolidated reports generated at the end of the analysis.
- **Error Handling**: `main_graph_error_message` is used to capture any errors that occur at the main graph level, outside of a specific subgraph.
