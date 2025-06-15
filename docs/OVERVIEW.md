# Financial Analyst AI - Project Overview

## Introduction
This project is a sophisticated financial analysis system that leverages AI agents, subgraphs, and nodes to perform comprehensive financial analysis. The system is designed to process financial data, perform sector analysis, generate reports, and provide insights using various AI techniques.

## Core Components

### 1. Agents
AI components that perform specific tasks and make decisions. They form the intelligent core of the application.

### 2. Subgraphs
Specialized processing units that handle specific types of financial analysis or data processing tasks.

### 3. Nodes
Individual processing units that perform specific operations within the system.

### 4. State Management
Centralized state management to maintain context and data flow between components.

## Getting Started
1. Install dependencies: `pip install -e .`
2. Configure your environment variables in `.env`
3. Run the main application: `python main.py`

## Project Structure
```
.
├── agents/           # AI agent implementations
├── subgraphs/        # Specialized analysis subgraphs
├── nodes/            # Individual processing nodes
├── core/             # Core application logic
├── config/           # Configuration files
├── tests/            # Test files
├── docs/             # Documentation
└── logs/             # Application logs
```

## Documentation Index
1. [Agents Documentation](AGENTS.md)
2. [Subgraphs Documentation](SUBGRAPHS.md)
3. [Nodes Documentation](NODES.md)
4. [State Management](STATE.md)
5. [API Reference](API.md)
