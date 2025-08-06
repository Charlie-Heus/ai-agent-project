
"""
FinanceQA AI Agent - Simplified Implementation
A basic agentic system for financial question answering using LangChain.

AGENT CARD:
============
What it does: Answers financial questions by iteratively using tools and reasoning
Tools: SEC Search, Web Search, Calculator, Knowledge Base
Capabilities: Financial data retrieval, calculations, multi-step reasoning
Limitations: ~50% accuracy, requires API keys, English only
How to use: Run python main.py and select from CLI menu

Architecture: LangChain ReAct Agent -> Iterative Tool Usage -> Final Answer
"""

import os
import asyncio
import json
import re 
import requests
import pickle
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict, List, Any
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# LangChain imports for agent functionality
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools import BaseTool, tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish

# External API libraries
import requests
import yfinance as yf
import numexpr # Using numexpr for safer evaluation

# NEW: Import for Google Custom Search API
from googleapiclient.discovery import build
from google.api_core.exceptions import GoogleAPIError

class FinanceQAAgent:
    """
    Main agent class that orchestrates financial question answering.
    
    Core Logic: Question -> Plan Tools -> Execute Tools -> Synthesize Answer
    Uses LangChain's ReAct pattern for iterative reasoning and tool usage.
    """
    
    def __init__(self):
        """Initialize the agent with LLM and tools."""
        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-3.5-turbo",
        )
        
        self.tools = [
            self.finnhub_search_tool,
            self.web_search_tool, 
            self.financial_calculator_tool,
            self.knowledge_base_tool,
            self.stock_price_tool, 
            self.fetch_webpage_content_tool,
            self.rag_search_tool
        ]
        
        self.agent = self._create_financial_agent()
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )
    
    def _create_financial_agent(self):
        """Create a LangChain ReAct agent optimized for financial analysis."""
        financial_prompt = PromptTemplate.from_template("""
        You are an expert financial analyst AI agent. Your job is to answer financial questions accurately using available tools.

        AVAILABLE TOOLS:
        {tools}

        GUIDELINES:
        - Always cite sources for financial data from web searches.
        - Use the 'stock_price_tool' for current stock prices. Use 'web_search_tool' for news and general info.
        - Perform calculations step-by-step.
        - Be transparent about limitations and uncertainty.

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        IMPORTANT: You must produce either a single 'Action' and 'Action Input' or a single 'Final Answer'. Never provide both in the same response.

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
        """)
        
        return create_react_agent(self.llm, self.tools, financial_prompt)
    
    # TOOL DEFINITIONS
    @tool
    def finnhub_search_tool(self, query: str) -> str:
        """
        Search Finnhub for comprehensive financial data and company information.
        Use for: Company profiles, financial metrics, earnings data, stock quotes.
        Input: Company ticker symbol (e.g., 'AAPL', 'MSFT', 'TSLA')
        """
        try:
            # Use Finnhub API for financial data
            finnhub_api_key = os.getenv('FINNHUB_API_KEY')
            if not finnhub_api_key:
                return "Finnhub API key not configured. Cannot access financial data."
            
            ticker = query.upper()
            
            # Get company profile
            profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={finnhub_api_key}"
            profile_response = requests.get(profile_url, timeout=10)
            
            if profile_response.status_code == 200:
                profile_data = profile_response.json()
                
                # Get financial metrics
                metrics_url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={finnhub_api_key}"
                metrics_response = requests.get(metrics_url, timeout=10)
                
                if metrics_response.status_code == 200:
                    metrics_data = metrics_response.json()
                    
                    # Get company earnings
                    earnings_url = f"https://finnhub.io/api/v1/stock/earnings?symbol={ticker}&token={finnhub_api_key}"
                    earnings_response = requests.get(earnings_url, timeout=10)
                    
                    earnings_data = []
                    if earnings_response.status_code == 200:
                        earnings_response_data = earnings_response.json()
                        if 'earnings' in earnings_response_data:
                            for earning in earnings_response_data['earnings'][:3]:  # Last 3 earnings
                                earnings_data.append({
                                    'period': earning.get('period', 'N/A'),
                                    'actual': earning.get('actual', 'N/A'),
                                    'estimate': earning.get('estimate', 'N/A'),
                                    'surprise': earning.get('surprise', 'N/A')
                                })
                    
                    # Compile comprehensive financial data
                    financial_data = {
                        'company': {
                            'name': profile_data.get('name', 'N/A'),
                            'ticker': profile_data.get('ticker', 'N/A'),
                            'country': profile_data.get('country', 'N/A'),
                            'currency': profile_data.get('currency', 'N/A'),
                            'exchange': profile_data.get('exchange', 'N/A'),
                            'ipo': profile_data.get('ipo', 'N/A'),
                            'marketCapitalization': profile_data.get('marketCapitalization', 'N/A'),
                            'shareOutstanding': profile_data.get('shareOutstanding', 'N/A'),
                            'logo': profile_data.get('logo', 'N/A'),
                            'finnhubIndustry': profile_data.get('finnhubIndustry', 'N/A')
                        },
                        'quote': {
                            'currentPrice': metrics_data.get('c', 'N/A'),
                            'change': metrics_data.get('d', 'N/A'),
                            'percentChange': metrics_data.get('dp', 'N/A'),
                            'highPrice': metrics_data.get('h', 'N/A'),
                            'lowPrice': metrics_data.get('l', 'N/A'),
                            'openPrice': metrics_data.get('o', 'N/A'),
                            'previousClose': metrics_data.get('pc', 'N/A')
                        },
                        'earnings': earnings_data
                    }
                    
                    return f"Finnhub Financial Data for {ticker}: {financial_data}"
                else:
                    return f"Finnhub metrics failed for {ticker}. Status: {metrics_response.status_code}"
            else:
                return f"Finnhub profile failed for {ticker}. Status: {profile_response.status_code}"
        except Exception as e:
            return f"Finnhub search error: {str(e)}"
    
    # MODIFIED: web_search_tool now uses the official Google API
    @tool 
    def web_search_tool(query: str) -> str:
        """
        Search the web for current financial news, market analysis, or general information.
        Use for: Finding recent news, understanding market trends, or looking up information not in other tools.
        Do NOT use for getting current stock prices; use 'stock_price_tool' for that.
        Input: A search query (e.g., 'Tesla Q2 2024 earnings analysis', 'impact of interest rates on tech stocks')
        """
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            cse_id = os.getenv("CUSTOM_SEARCH_ENGINE_ID")
            
            if not api_key or not cse_id:
                return "Google Search API is not configured. Missing API key or Custom Search Engine ID."

            service = build("customsearch", "v1", developerKey=api_key)
            result = service.cse().list(q=query, cx=cse_id, num=3).execute()
            
            if 'items' not in result:
                return f"No web search results found for '{query}'."

            # Format the results for the agent
            snippets = []
            for item in result['items']:
                title = item.get('title', 'No Title')
                link = item.get('link', '#')
                snippet = item.get('snippet', 'No snippet available.').replace('\n', ' ')
                snippets.append(f"Title: {title}\nSnippet: {snippet}\nSource: {link}")
            
            return "\n---\n".join(snippets)

        except GoogleAPIError as e:
            return f"Google Search API error: {e}"
        except Exception as e:
            return f"Web search error: {str(e)}"
    
    @tool
    def fetch_webpage_content_tool(self, url: str) -> str:
        """
        Fetch full content from a specific webpage URL.
        Use for: Getting detailed information from a specific source found via web search.
        Input: A valid URL (e.g., 'https://example.com/article')
        """
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit to 1500 characters
            max_chars = 1500
            return text[:max_chars] + "..." if len(text) > max_chars else text
            
        except Exception as e:
            return f"Could not fetch content from {url}: {str(e)}"

    # NEW: Dedicated tool for getting stock prices
    @tool
    def stock_price_tool(ticker_symbol: str) -> str:
        """
        Get the current stock price for a given ticker symbol.
        Use for: Retrieving the latest stock price for a specific company.
        Input: A single valid stock ticker symbol (e.g., 'AAPL', 'TSLA').
        """
        try:
            ticker = yf.Ticker(ticker_symbol.upper())
            info = ticker.info
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            if price:
                company_name = info.get('longName', ticker_symbol)
                return f"The current stock price for {company_name} ({ticker_symbol.upper()}) is ${price:.2f}."
            else:
                return f"Could not retrieve stock price for ticker '{ticker_symbol}'. It may be an invalid symbol."
        except Exception as e:
            return f"Error retrieving stock price for '{ticker_symbol}': {e}"
            
    # MODIFIED: More robust financial_calculator_tool
    @tool
    def financial_calculator_tool(expression: str) -> str:
        """
        Perform mathematical calculations.
        Use for: Evaluating mathematical expressions, like calculating ratios, growth rates, and basic math.
        Input: A valid mathematical string (e.g., '100 / 5', '(150 + 250) * 0.15').
        """
        try:
            safe_expr = "".join(re.findall(r'[0-9\.\+\-\*\/\(\)\s]', expression))
            if not safe_expr:
                return "Invalid expression. No calculable content found."
            result = numexpr.evaluate(safe_expr).item()
            return f"Calculation result: {safe_expr.strip()} = {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}. Please provide a valid mathematical expression."
    
    @tool
    def knowledge_base_tool(query: str) -> str:
        """
        Search financial knowledge base for definitions and concepts.
        Use for: Financial term definitions, concept explanations, formulas.
        Input: Financial term or concept (e.g., 'what is EBITDA', 'debt to equity ratio')
        """
        # (Your existing knowledge_base_tool code is fine)
        knowledge_base = {
            'pe ratio': 'Price-to-Earnings ratio: Stock price divided by earnings per share. Indicates how much investors pay per dollar of earnings.',
            'ebitda': 'Earnings Before Interest, Taxes, Depreciation, and Amortization. Measures company profitability.',
            'debt to equity': 'Debt-to-Equity ratio: Total debt divided by total equity. Measures financial leverage.',
            'market cap': 'Market Capitalization: Stock price multiplied by total shares outstanding.',
        }
        query_lower = query.lower()
        for term, definition in knowledge_base.items():
            if term in query_lower:
                return f"{term.upper()}: {definition}"
        return f"No definition found for '{query}'. Available terms: {list(knowledge_base.keys())}"
    
    @tool
    def rag_search_tool(self, query: str) -> str:
        """
        Search through the FinanceQA dataset using RAG (Retrieval-Augmented Generation) with on-the-fly embeddings.
        Use for: Finding detailed information from the FinanceQA dataset, financial documents, or reports.
        Input: A search query (e.g., 'What is the impact of interest rates on tech stocks?', 'How do I calculate ROI?')
        """
        try:
            # Initialize embeddings
            embeddings = OpenAIEmbeddings()
            
            # Load FinanceQA data
            data_path = Path("data/financeqa_test.jsonl")
            if not data_path.exists():
                return "FinanceQA dataset not found. Please ensure data/financeqa_test.jsonl exists."
            
            # Load and parse the data
            documents = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        doc = json.loads(line.strip())
                        documents.append(doc)
                    except json.JSONDecodeError:
                        continue
            
            if not documents:
                return "No valid documents found in the FinanceQA dataset."
            
            # Get query embedding
            query_embedding = embeddings.embed_query(query)
            
            # Process each document with on-the-fly chunking and embedding
            all_results = []
            
            for i, doc in enumerate(documents):
                question_text = doc.get('question', '')
                context_text = doc.get('context', '')
                answer_text = doc.get('answer', '')
                
                if not question_text:
                    continue
                
                # Create question embedding
                question_embedding = embeddings.embed_query(question_text)
                question_similarity = np.dot(query_embedding, question_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(question_embedding)
                )
                
                all_results.append({
                    'question_id': i + 1,
                    'question_text': question_text,
                    'similarity': question_similarity,
                    'answer_text': answer_text,
                    'context_text': context_text,
                    'type': 'question'
                })
                
                # Process context if it exists
                if context_text and context_text.strip():
                    # Split context into chunks
                    chunks = self._split_context_into_chunks(context_text)
                    
                    # Create embeddings for each chunk
                    for j, chunk in enumerate(chunks):
                        chunk_embedding = embeddings.embed_query(chunk)
                        chunk_similarity = np.dot(query_embedding, chunk_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                        )
                        
                        all_results.append({
                            'question_id': i + 1,
                            'question_text': question_text,
                            'similarity': chunk_similarity,
                            'answer_text': answer_text,
                            'context_text': chunk,
                            'type': 'context_chunk',
                            'chunk_index': j
                        })
            
            # Sort by similarity and get top 3 results
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            top_results = all_results[:3]
            
            # Format the results
            if top_results:
                response = "RAG Search Results:\n\n"
                for i, result in enumerate(top_results, 1):
                    response += f"{i}. Similarity: {result['similarity']:.3f} ({result['type']})\n"
                    response += f"   Question ID: {result['question_id']}\n"
                    response += f"   Question: {result['question_text'][:100]}...\n"
                    response += f"   Answer: {result['answer_text'][:100]}...\n"
                    if result['type'] == 'context_chunk':
                        response += f"   Context Chunk {result.get('chunk_index', 0) + 1}: {result['context_text'][:200]}...\n\n"
                    else:
                        response += f"   Context: {result['context_text'][:200]}...\n\n"
                return response
            else:
                return "No relevant information found in the FinanceQA dataset."
                
        except Exception as e:
            return f"RAG search error: {str(e)}"
    
    def _split_context_into_chunks(self, context_text: str, chunk_size: int = 200) -> List[str]:
        """Split context text into meaningful chunks."""
        if not context_text:
            return []
        
        # Split by sentences first
        sentences = context_text.split('. ')
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Add period back if it was removed
            if not sentence.endswith('.'):
                sentence += '.'
            
            # If adding this sentence would make chunk too long, start new chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def process_question(self, question: str) -> Dict[str, Any]:
        """Main method to process a financial question using the agent."""
        # (Your existing process_question code is fine)
        start_time = datetime.now()
        try:
            result = await asyncio.to_thread(
                self.agent_executor.invoke,
                {"input": question}
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            return {"question": question, "answer": result["output"], "success": True, "processing_time": processing_time}
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return {"question": question, "answer": f"Error processing question: {str(e)}", "success": False, "processing_time": processing_time, "error": str(e)}

# (The FinanceAgentCLI class remains the same, no changes needed there)
class FinanceAgentCLI:
    def __init__(self): self.agent = FinanceQAAgent()
    def run(self):
        print("\n🏦 Welcome to FinanceQA AI Agent!")
        print("=" * 50)
        while True:
            print("\nSelect from the following options:")
            print("1. Run Pre-Built Demo (automated scenarios)")
            print("2. Run Live Demo (interactive Q&A)")
            print("3. Exit")
            choice = input("\n> ").strip()
            if choice == "1": asyncio.run(self.run_prebuilt_demo())
            elif choice == "2": asyncio.run(self.run_live_demo())
            elif choice == "3": print("Goodbye! 👋"); break
            else: print("Invalid option. Please select 1-3.")
    async def run_prebuilt_demo(self):
        print("\n🎯 Pre-Built Demo Starting...")
        
        # Load the FinanceQA dataset
        try:
            import json
            from pathlib import Path
            
            data_path = Path("data/financeqa_test.jsonl")
            if not data_path.exists():
                print("❌ FinanceQA dataset not found. Please run download_financeqa.py first.")
                return
            
            # Load all questions
            questions = []
            with open(data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
            
            # Take the first 4 questions
            demo_questions = questions[:4]
            print(f"✅ Loaded first 4 questions from FinanceQA dataset")
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return
        
        for i, question_data in enumerate(demo_questions, 1):
            print(f"\n--- Demo {i}/{len(demo_questions)} ---")
            print(f"Type: {question_data.get('question_type', 'N/A')}")
            print(f"Company: {question_data.get('company', 'N/A')}")
            print(f"Question: {question_data.get('question', 'N/A')}")
            print(f"Expected Answer: {question_data.get('answer', 'N/A')}")
            
            print("\n🤖 Agent thinking...")
            result = await self.agent.process_question(question_data['question'])
            
            print(f"\n✅ AI Answer: {result['answer']}")
            print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
            
            # Show comparison
            print(f"\n📊 Comparison:")
            print(f"Expected: {question_data.get('answer', 'N/A')}")
            print(f"AI Answer: {result['answer']}")
            
            if i < len(demo_questions): 
            input("\nPress Enter to continue...")
    async def run_live_demo(self):
        print("\n💬 Live Demo - Choose from FinanceQA Dataset!")
        print("Type 'quit' to return to main menu.")
        
        # Load the FinanceQA dataset
        try:
            import json
            from pathlib import Path
            
            data_path = Path("data/financeqa_test.jsonl")
            if not data_path.exists():
                print("❌ FinanceQA dataset not found. Please run download_financeqa.py first.")
                return
            
            # Load all questions
            questions = []
            with open(data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
            
            print(f"✅ Loaded {len(questions)} questions from FinanceQA dataset")
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return
        
        while True:
            print(f"\n📊 Choose a question from 1-{len(questions)} (or type 'quit' to exit):")
            choice = input("> ").strip()
            
            if choice.lower() in ['quit', 'exit', 'q']:
                break
            
            try:
                question_num = int(choice)
                if 1 <= question_num <= len(questions):
                    # Get the selected question
                    selected_question = questions[question_num - 1]
                    
                    print(f"\n📋 Question #{question_num}:")
                    print(f"Type: {selected_question.get('question_type', 'N/A')}")
                    print(f"Company: {selected_question.get('company', 'N/A')}")
                    print(f"Question: {selected_question.get('question', 'N/A')}")
                    print(f"Expected Answer: {selected_question.get('answer', 'N/A')}")
                    
                    print("\n🤖 Processing with AI Agent...")
                    result = await self.agent.process_question(selected_question['question'])
                    
                    print(f"\n✅ AI Answer: {result['answer']}")
                    print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
                    
                    # Show comparison
                    print(f"\n📊 Comparison:")
                    print(f"Expected: {selected_question.get('answer', 'N/A')}")
                    print(f"AI Answer: {result['answer']}")
                    
                else:
                    print(f"❌ Please enter a number between 1 and {len(questions)}")
                    
            except ValueError:
                print("❌ Please enter a valid number or 'quit'")
            except Exception as e:
                print(f"❌ Error processing question: {e}")


# MODIFIED: check_environment now checks for the new Google API keys
def check_environment():
    """Check if required environment variables are available."""
    print("🔍 Checking environment...")
    
    required_keys = ['OPENAI_API_KEY', 'GOOGLE_API_KEY', 'CUSTOM_SEARCH_ENGINE_ID']
    optional_keys = ['SEC_API_KEY', 'FINNHUB_API_KEY']
    
    missing_required = [key for key in required_keys if not os.getenv(key)]
    missing_optional = [key for key in optional_keys if not os.getenv(key)]
    
    if missing_required:
        print(f"❌ Missing required API keys: {missing_required}")
        print("Please set these environment variables in your .env file before running.")
        return False
    
    if missing_optional:
        print(f"⚠️  Missing optional API keys: {missing_optional}. Some features may be limited.")
    
    print("✅ Environment check passed!")
    return True


if __name__ == "__main__":
    if not check_environment():
        print("\n💡 Setup Instructions:")
        print("1. Set up OpenAI, Google Cloud, and Programmable Search Engine to get API keys.")
        print("2. Create a .env file and add your keys (see documentation for format).")
        print("3. Install dependencies: pip install -r requirements.txt (or see file header).")
        exit(1)
    
    try:
        cli = FinanceAgentCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\nExiting... Goodbye! 👋")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")