# Financial Analyst AI

A sophisticated financial analysis system leveraging AI agents, subgraphs, and nodes to provide comprehensive financial insights and analysis.

## 📚 Documentation

For detailed documentation, please refer to the following files in the `docs/` directory:

- [📖 Project Overview](docs/OVERVIEW.md) - High-level system architecture and components
- [🤖 Agents](docs/AGENTS.md) - Documentation of AI agents and their responsibilities
- [🔗 Subgraphs](docs/SUBGRAPHS.md) - Specialized analysis subgraphs and their functions
- [⚙️ Nodes](docs/NODES.md) - Individual processing nodes and their operations
- [🔄 State Management](docs/STATE.md) - How state is managed across the system
- [🌐 API Reference](docs/API.md) - Detailed API documentation

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip
- (Optional) virtualenv

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -e .
   ```
3. Configure your environment variables in `.env`

### Running the Application
```bash
python main.py
```

## 🏗️ Project Structure

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

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.