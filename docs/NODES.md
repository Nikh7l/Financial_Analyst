# Nodes Documentation

## Overview

In this project, "nodes" are the fundamental processing steps within a `StateGraph` defined by LangGraph. They are not class-based components but are implemented as **Python functions**. Each node takes the current state of the graph as input and returns a dictionary containing the updates to that state.

The project's subgraphs follow a consistent pattern of node implementation, defining the logic for each step of the analysis directly within the subgraph's file.

**Note on the `/nodes` Directory**: The project contains a `/nodes` directory with files like `report_generation_node.py`. Code analysis shows that these files are **not currently used or imported** by any of the active subgraphs. The true operational nodes are the functions defined within the subgraph files themselves.

---

## Common Node Patterns

Across the various subgraphs, three primary types of nodes are consistently used to structure the workflow:

### 1. The "Start" or "Prepare Message" Node
- **Example**: `start_sector_sentiment_node`, `start_trends_analysis_node`
- **Purpose**: This node is typically the entry point of the graph. Its primary responsibility is to prepare the initial `HumanMessage` that will be sent to the ReAct agent.
- **Logic**:
    - It reads the initial input (e.g., `company_name`, `sector_name`) from the state.
    - It checks if this is the first attempt or a retry. If it's a retry, it often incorporates error information from the previous run into the new prompt.
    - It returns a dictionary with the `messages` list updated with the new `HumanMessage`.

### 2. The "Agent Executor" Node
- **Example**: `sector_sentiment_react_runnable`, `competitor_react_agent_runnable`
- **Purpose**: This node is the core of the subgraph. It is the pre-built ReAct agent itself.
- **Logic**:
    - It takes the `messages` list from the state.
    - The ReAct agent executes its reasoning loop: it thinks, selects a tool, executes it, observes the result, and repeats until it has a final answer.
    - It appends all intermediate steps (tool calls, tool responses, AI thoughts) and the final `AIMessage` back into the `messages` list in the state.

### 3. The "Parse and Check" Node
- **Example**: `parse_and_check_sentiment_node`, `parse_and_check_trends_node`
- **Purpose**: This node runs after the agent executor to validate the agent's output and decide the next step.
- **Logic**:
    - It retrieves the final `AIMessage` from the state.
    - It attempts to parse the content of the message, which is expected to be a JSON string.
    - It validates that the parsed JSON contains the required fields and that the data is well-formed.
    - Based on the validation result, it sets a `_route_decision` in the state, which is used by the graph's conditional edges to determine where to go next: `success`, `retry`, or `fail`.
    - It populates the final output field in the state (e.g., `extracted_trends_data`, `competitors`) on success.
