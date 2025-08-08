# FinanceQA AI Agent

An intelligent financial analysis agent designed to tackle the FinanceQA benchmark with an agentic approach, significantly improving upon non-agentic baselines through strategic tool usage and iterative reasoning.

## Agent Card

**What the Agent Does:**
The FinanceQA AI Agent is a sophisticated financial analysis system that answers complex financial questions by:
- Analyzing questions to identify required formulas and data points
- Searching multiple data sources (financial APIs, web search, knowledge bases)
- Extracting values from financial documents and contexts
- Performing accurate financial calculations
- Providing transparent reasoning and citing sources

**Core Capabilities:**
- Formula identification and financial calculation (EBITDA, P/E ratios, cash flows, etc.)
- Multi-source data retrieval (Finnhub API, web search, RAG systems)
- Context-aware information extraction from financial documents
- Iterative tool selection based on information completeness assessment
- Transparent reasoning with step-by-step explanations

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key
- Google API key (for web search)
- Finnhub API key (optional, for enhanced financial data)

### Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd financeqa-ai-agent
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
CUSTOM_SEARCH_ENGINE_ID=your_search_engine_id_here
FINNHUB_API_KEY=your_finnhub_api_key_here  # Optional
```

4. **Download the FinanceQA dataset:**
```bash
python download_financeqa.py
```

### Usage

**Interactive CLI:**
```bash
python main.py
```

The agent provides two main options:
1. **Pick Question** - Select and analyze specific questions from the dataset
2. **Run Demo** - See the complete agent thought process on the first question

**Docker Deployment:**
```bash
# Build the container
docker build -t ai_app .

# Run the container
docker run -it --env-file .env ai_app
```

## Architecture Overview

The agent employs a **ReAct (Reasoning + Acting) pattern** with custom control flows optimized for financial analysis:

```
Question Input → Formula Analysis → Context Assessment → Tool Selection Loop → Calculation → Answer
```

### Core Components

1. **FinanceQAAgent Class** - Main orchestrator with 9 specialized tools
2. **Control Flow Logic** - Custom question processing with dual pathways
3. **Tool Ecosystem** - Comprehensive suite of financial analysis tools
4. **Iterative Assessment** - Dynamic tool selection based on information completeness

### Tool Suite

| Tool | Purpose | Data Source |
|------|---------|-------------|
| `formula_analysis_tool` | Extract required formulas and key terms | LLM Analysis |
| `key_terms_search_tool` | Find specific data points in documents | Text Processing |
| `financial_calculator_tool` | Perform safe mathematical calculations | Local Computation |
| `finnhub_search_tool` | Retrieve comprehensive financial metrics | Finnhub API |
| `web_search_tool` | Search for current financial information | Google Search API |
| `rag_search_tool` | Query FinanceQA knowledge base | Vector Search |
| `direct_rag_search_tool` | Fast question-specific context lookup | Dataset Cache |
| `fetch_webpage_content_tool` | Extract content from financial websites | Web Scraping |
| `knowledge_base_tool` | Define financial terms and concepts | Built-in Knowledge |

## Key Features

### 🧠 Intelligent Question Analysis
- Automatic formula extraction from natural language questions
- Context-aware key term identification with synonym mapping
- Dual pathway processing (WITH CONTEXT vs NO CONTEXT scenarios)

### 🔄 Iterative Tool Selection
- Dynamic assessment of information completeness
- Intelligent tool selection based on missing information
- Maximum 5 tool calls per question with early termination on success

### 📊 Comprehensive Financial Coverage
- Support for all major financial metrics (EBITDA, P/E ratios, cash flows, etc.)
- Handle both basic calculations and complex adjusted metrics
- Real-time financial data integration

### 🔍 Transparent Reasoning
- Step-by-step explanation of the analysis process
- Source citation for all retrieved data
- Clear presentation of intermediate calculations

## Performance & Accuracy

The agent is designed to significantly improve upon the 54.1% baseline through:
- **Structured reasoning** rather than direct LLM inference
- **Multi-source validation** of financial data
- **Iterative refinement** of information gathering
- **Safe calculation execution** with error handling

However, while I did implement many tools that offer useful functionality, I was not able get the agent working well enough to outpreform the benchmark. Many improvements can be made on top of the foundation of this project.

*Note: Formal benchmark evaluation is included in the evaluation suite.*

## Example Usage

```python
from main import FinanceQAAgent
import asyncio

# Initialize the agent
agent = FinanceQAAgent()

# Process a question
result = await agent.process_question(
    "What is the P/E ratio for Apple in 2024?"
)

print(result['answer'])
```

## Testing & Development

**Run individual tool tests:**
```bash
python test_financial_calculator.py
python test_finnhub_search.py
python test_webpage_content.py
```

**Test complete agent flow:**
```bash
python debug_agent_loop_test.py
```

**Explore the dataset:**
```bash
python explore_financeqa.py
python organize_financeqa.py
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API access for LLM capabilities |
| `GOOGLE_API_KEY` | ✅ | Google API for web search functionality |
| `CUSTOM_SEARCH_ENGINE_ID` | ✅ | Google Custom Search Engine ID |
| `FINNHUB_API_KEY` | ⚠️ | Optional: Enhanced financial data access |

## Acknowledgments

- Built for the Veris AI candidate task
- Uses the FinanceQA benchmark dataset
- Powered by OpenAI's language models and LangChain framework

---

**For detailed design decisions and technical deep-dive, see [DESIGN.md](DESIGN.md)**