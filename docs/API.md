# API Reference

## Overview
This document provides detailed information about the application programming interfaces (APIs) available in the Financial Analyst system.

## Core APIs

### 1. Agent API

#### `BaseAgent`
- `initialize()`: Initialize the agent
- `process_message(message)`: Process incoming messages
- `get_status()`: Get current agent status
- `shutdown()`: Gracefully shut down the agent

#### `RetrievalAgent`
- `retrieve_data(query, params)`: Retrieve data based on query
- `preprocess_data(data)`: Preprocess retrieved data
- `cache_data(key, data)`: Cache data for future use

#### `RouterAgent`
- `route_message(message, destination)`: Route message to destination
- `register_route(route, handler)`: Register a new route handler
- `broadcast(message, topic)`: Broadcast message to topic subscribers

### 2. Subgraph API

#### Common Methods
- `process(input_data)`: Process input data through the subgraph
- `get_metrics()`: Get performance metrics
- `reset_state()`: Reset subgraph to initial state

### 3. Node API

#### Common Methods
- `execute(inputs)`: Execute node with given inputs
- `validate_inputs(inputs)`: Validate node inputs
- `get_output_schema()`: Get output data schema
- `get_metadata()`: Get node metadata

## REST API Endpoints

### Analysis Endpoints
- `POST /api/analyze`: Submit new analysis request
- `GET /api/analysis/{id}`: Get analysis results
- `GET /api/analyses`: List all analyses
- `DELETE /api/analysis/{id}`: Cancel analysis

### Data Endpoints
- `GET /api/data/sectors`: List available sectors
- `GET /api/data/companies`: List companies
- `GET /api/data/company/{id}`: Get company details
- `GET /api/data/market`: Get market data

## WebSocket API

### Connection
- `wss://api.example.com/ws`: WebSocket endpoint

### Events
- `analysis_update`: Analysis progress update
- `market_data`: Real-time market data
- `system_alert`: System alerts and notifications

## Error Handling
All API responses follow a standard format:
```json
{
  "status": "success|error",
  "data": {},
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message"
  },
  "timestamp": "ISO-8601 timestamp"
}
```

## Authentication
All API endpoints require authentication using JWT tokens:
```
Authorization: Bearer <token>
```

## Rate Limiting
- 1000 requests per hour per token
- 10000 requests per hour per IP address

## Versioning
API version is specified in the URL:
```
/api/v1/endpoint
```

## Data Formats
All request/response bodies are in JSON format with `Content-Type: application/json`.
