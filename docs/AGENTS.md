# Agents & Agentic Components Documentation

This document provides a code-verified overview of the different types of agentic components used in the Financial Analyst project. The system employs a mixed architecture, where "agent" can refer to several types of components.

## 1. ReAct Agents in Subgraphs

The primary agentic logic is implemented as **ReAct (Reasoning and Acting) agents** that are defined and used directly within the project's [subgraphs](./SUBGRAPHS.md). These agents are created using `langgraph.prebuilt.create_react_agent` and are responsible for the core data gathering and analysis tasks (e.g., competitor analysis, sentiment analysis, etc.). They are not defined in the `/agents` directory but are central to the system's functionality.

## 2. Components in the `/agents` Directory

The `/agents` directory contains a mix of components that support the overall agentic architecture. These are not uniform and serve different purposes.

### `base_agent.py`
- **Purpose**: Defines an abstract base class, `BaseAgent`.
- **Architecture**: This class provides a standardized structure for creating custom, non-ReAct agents. It includes core logic for interacting with the Google Gemini LLM, handling tool definitions, and executing tool calls. It is the foundation for class-based agents in the project.

### `router_agent.py`
- **Purpose**: Defines the `RouterAgent` class, which is responsible for the initial processing of a user's query.
- **Architecture**: Inherits from `BaseAgent`. This agent performs a specific, tool-less task: it analyzes the user's query to classify it as being about a "company" or a "sector" and extracts the entity name. This classification is used to route the query to the appropriate subgraph.

### `retrieval_agent.py`
- **Purpose**: Provides a pre-configured, runnable ReAct agent for document retrieval.
- **Architecture**: This script does not define a class. Instead, it creates a runnable agent instance (`retrieval_agent_runnable`) using `langgraph.prebuilt.create_react_agent`. This agent is specialized with a specific system prompt and a set of retrieval-focused tools. It is imported and used by the `retrieval_summarization_graph`.

### `summarization_agent.py`
- **Purpose**: Provides a function for summarizing documents.
- **Architecture**: This file does not contain an agent. It defines a single function, `summarize_documents_node`, which serves as a node within a LangGraph graph. This function takes a list of document URLs from the graph's state, calls a summarization tool for each, and returns the summaries. It is a functional component, not an agent.
