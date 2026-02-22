# Agentity

A self-modifying AI agent that can read, edit, and restart its own source code to achieve goals autonomously.

## Overview

Agentity is an experimental AI agent built with LangChain that can modify its own source code to gain new capabilities on-the-fly. It analyzes the problem, adds new `@tool` functions to itself (when necessary, it changes other parts of his code too), validates the changes, and restarts to continue execution with enhanced abilities.

### Key Features

- **Self-Modification**: The agent can read and edit its own Python source code
- **Auto-Restart with Context Preservation**: After modifying itself, it restarts and resumes execution with full conversation context
- **Autonomous Self-Patching**: Automatically patches its code as needed to accomplish goals
- **Validation Pipeline**: Built-in code validation with `ruff` and `ty` before every restart
- **Exception Auto-Healing**: Catches exceptions, analyzes them, and attempts self-repair
- **Isolated (but not secure) Execution**: Runs in a temporary working directory to preserve the original source

## Requirements

- Python 3.10+ (but < 3.14)
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd agentity
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   uv sync --extra dev  # Install dev dependencies (ruff, ty)
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.template .env
   # Edit .env and add your OpenAI API key
   ```

4. **Set up your `.env` file**:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4                           # or gpt-5.2, gpt-4o, etc.
   # OPENAI_BASE_URL=https://api.openai.com/v1  # Optional: for custom endpoints
   ```

## Usage

### Basic Usage

Run the agent with a goal:

```bash
uv run agentity.py "Your goal here"
```

Or run interactively:

```bash
uv run agentity.py
# You'll be prompted to enter a goal
```

### Examples

```bash
# Example: The agent will add tools to fetch and parse geo-IP and weather data
uv run agentity.py "What's the current temperature in current city?"

# Example: The agent will add file I/O tools
uv run agentity.py "Create a file called output.txt with the Fibonacci sequence up to 100"

# Example: The agent will add calculation tools
uv run agentity.py "Calculate the factorial of 20"
```

### How It Works

1. **Initial Assessment**: The agent receives your goal and analyzes what tools it needs
2. **Tool Creation**: If capabilities are missing, it:
   - Reads its own source code with `read_source()`
   - Adds new `@tool` functions with `edit_source()` (prefered) or patches other parts of its code
   - Validates the code with `ruff` and `ty`
   - Restarts with `restart_agent(next_step="...")`
3. **Execution**: After restart, it uses the new tools to accomplish the goal
4. **Iteration**: Repeats this cycle until the goal is achieved

### Resuming from Context

The agent automatically saves its state before restarting. To manually resume from saved context:

```bash
uv run agentity.py --context
```

## Project Structure

```
agentity/
├── agentity.py           # Main agent source (self-modifying)
├── pyproject.toml        # Project dependencies
├── uv.lock               # Dependency lock file
├── .env                  # Environment configuration (not in git)
├── .env.template         # Environment template
└── README.md             # This file
```

### Working Directory

The agent creates a temporary working directory on first run (e.g., `/tmp/agentity-XXXXX/`) to:
- Preserve the original source code
- Allow safe self-modification
- Maintain context between restarts

## Development

### Code Quality Tools

The project uses:
- **ruff**: Fast Python linter and formatter
- **ty**: Type checker

Run validation manually:
```bash
uv run ruff check agentity.py
uv run ty check agentity.py
```

### Architecture Highlights

- **Tool Discovery**: `discover_tools()` auto-finds all `@tool`-decorated functions
- **Context Persistence**: Conversation state saved to `.agent_context.json` before restarts
- **Exception Recovery**: Catches runtime exceptions and attempts auto-healing
- **Validation Pipeline**: Enforces code quality before allowing restarts

## Limitations & Considerations

- **Python Standard Library Only**: Agent can only use built-in Python capabilities (no external package installation at runtime)
- **Token Limits**: Complex goals requiring many iterations may hit LLM context limits
- **Security**: This agent modifies its own code—use in non-isolated environment
- **Cost**: Multiple LLM calls during self-modification can accumulate API costs

## Contributing

This is an experimental project exploring self-modifying AI systems. Contributions, ideas, and discussions are welcome!

## Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
- [OpenAI](https://openai.com/) - Language model API
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
