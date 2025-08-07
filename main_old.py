
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
from typing import Dict, List, Any, Optional
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
            self.fetch_webpage_content_tool,
            self.rag_search_tool,
            self.formula_analysis_tool,
            self.key_terms_search_tool
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
        - For financial calculation questions, start by using 'formula_analysis_tool' to understand what formula and data points are needed
        - Use 'key_terms_search_tool' to find specific financial data in documents after identifying key terms
        - Always cite sources for financial data from web searches
        - Use 'finnhub_search_tool' for stock prices and comprehensive financial data
        - Use 'web_search_tool' for news and general info
        - Perform calculations step-by-step using 'financial_calculator_tool'
        - Use 'rag_search_tool' to find relevant information from the FinanceQA dataset
        - Be transparent about limitations and uncertainty

        WORKFLOW FOR FINANCIAL CALCULATIONS:
        1. Use 'formula_analysis_tool' to extract the required formula and key terms
        2. Use 'key_terms_search_tool' or 'rag_search_tool' to find the specific data points
        3. Use 'financial_calculator_tool' to perform the calculation
        4. Provide the final answer with explanation

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
    def finnhub_search_tool(query: str) -> str:
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
        Do NOT use for getting current stock prices; use 'finnhub_search_tool' for that.
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
    def fetch_webpage_content_tool(url: str) -> str:
        """
        Fetch and extract content from a webpage.
        Use for: Getting detailed information from specific web pages.
        Input: A valid URL (e.g., 'https://example.com')
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
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
            
            return text[:1500] + "..." if len(text) > 1500 else text
        except Exception as e:
            return f"Could not fetch content from {url}: {str(e)}"
            
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
    def rag_search_tool(query: str) -> str:
        """
        Search for relevant information in the FinanceQA knowledge base using RAG.
        Use for: Finding relevant financial context and information from the FinanceQA dataset.
        Input: A search query (e.g., 'gross profit margin calculation', 'revenue analysis')
        """
        try:
            # Load the FinanceQA dataset
            import json
            import pickle
            import hashlib
            import signal
            import numpy as np
            from pathlib import Path
            
            def timeout_handler(signum, frame):
                """Handle timeout for API calls"""
                raise TimeoutError("API call timed out")
            
            def safe_embed_query(embeddings, text, timeout=30):
                """Safely embed text with timeout"""
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
                try:
                    result = embeddings.embed_query(text)
                    signal.alarm(0)  # Cancel timeout
                    return result
                except TimeoutError:
                    signal.alarm(0)  # Cancel timeout
                    raise TimeoutError(f"Embedding request timed out after {timeout} seconds")
                except Exception as e:
                    signal.alarm(0)  # Cancel timeout
                    raise e
            
            def split_context_into_chunks(context_text: str, chunk_size: int = 200, overlap: int = 100) -> List[str]:
                """Split context text into meaningful chunks with overlap."""
                if not context_text:
                    return []
                
                # Split by sentences first
                sentences = context_text.split('. ')
                
                chunks = []
                current_chunk = ""
                overlap_text = ""
                
                for sentence in sentences:
                    # Add period back if it was removed
                    if not sentence.endswith('.'):
                        sentence += '.'
                    
                    # If adding this sentence would make chunk too long, start new chunk
                    if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                        # Add current chunk
                        chunks.append(current_chunk.strip())
                        
                        # Calculate overlap: take the last portion of current chunk
                        if overlap > 0 and len(current_chunk) > overlap:
                            # Find the last sentence boundary within the overlap region
                            overlap_region = current_chunk[-overlap:]
                            last_sentence_start = overlap_region.find('. ')
                            if last_sentence_start != -1:
                                overlap_text = overlap_region[last_sentence_start + 2:]  # +2 to skip '. '
                            else:
                                overlap_text = overlap_region
                        else:
                            overlap_text = current_chunk
                        
                        # Start new chunk with overlap
                        current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                
                # Add the last chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                return chunks
            
            def get_cache_key(question_text: str, context_text: str) -> str:
                """Generate a unique cache key for a question and its context."""
                content = f"{question_text}|{context_text}"
                return hashlib.md5(content.encode('utf-8')).hexdigest()
            
            def get_cached_embeddings(question_text: str, context_text: str, cache_data: dict) -> Optional[dict]:
                """Get cached embeddings for a question and context."""
                cache_key = get_cache_key(question_text, context_text)
                return cache_data.get(cache_key)
            
            def cache_embeddings(question_text: str, context_text: str, question_embedding: List[float], 
                               chunk_embeddings: List[List[float]], cache_data: dict):
                """Cache embeddings for a question and context."""
                cache_key = get_cache_key(question_text, context_text)
                cache_data[cache_key] = {
                    'question_embedding': question_embedding,
                    'chunk_embeddings': chunk_embeddings,
                    'timestamp': os.path.getmtime('data/financeqa_test.jsonl')
                }
            
            data_path = Path("data/financeqa_test.jsonl")
            if not data_path.exists():
                return "FinanceQA dataset not found. Please ensure data/financeqa_test.jsonl exists."
            
            # Load all questions and contexts
            documents = []
            with open(data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        if doc.get('context'):
                            documents.append({
                                'question': doc.get('question', ''),
                                'context': doc.get('context', ''),
                                'answer': doc.get('answer', '')
                            })
            
            if not documents:
                return "No documents found in the FinanceQA dataset."
            
            # Load embeddings cache
            cache_path = Path("data/embeddings_cache.pkl")
            cache_data = {}
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                    print(f"📦 Loaded cache with {len(cache_data)} cached embeddings")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load cache: {e}")
            else:
                print("📝 No cache file found, will create new embeddings")
            
            # Initialize embeddings
            embeddings = OpenAIEmbeddings()
            
            # Find the most relevant question that matches our query
            query_embedding = safe_embed_query(embeddings, query)
            
            # Find the most similar question to our query
            best_question_similarity = -1
            best_question_doc = None
            
            print(f"🔍 Finding most relevant question for query: '{query[:50]}...'")
            
            for i, doc in enumerate(documents):
                try:
                    question_embedding = safe_embed_query(embeddings, doc['question'])
                    similarity = np.dot(query_embedding, question_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(question_embedding)
                    )
                    
                    if similarity > best_question_similarity:
                        best_question_similarity = similarity
                        best_question_doc = doc
                        
                except Exception as e:
                    print(f"⚠️  Error processing question {i+1}: {e}")
                    continue
            
            if not best_question_doc:
                return "No relevant questions found for the query."
            
            print(f"✅ Found best matching question (similarity: {best_question_similarity:.4f})")
            print(f"Question: {best_question_doc['question']}")
            
            # Now analyze the chunks of the best matching question using question-specific caching
            context_text = best_question_doc['context']
            question_text = best_question_doc['question']
            
            # Split context into chunks
            chunks = split_context_into_chunks(context_text)
            
            print(f"📊 Context Analysis:")
            print(f"   Total context length: {len(context_text)} characters")
            print(f"   Number of chunks: {len(chunks)}")
            print(f"   Average chunk length: {sum(len(chunk) for chunk in chunks)/len(chunks):.0f} characters")
            print(f"   Chunk size: 200 characters with 100 character overlap")
            
            # Check cache first using question-specific key
            cached_data = get_cached_embeddings(question_text, context_text, cache_data)
            
            if cached_data:
                print(f"⚡ Using cached embeddings for this question")
                question_embedding = cached_data['question_embedding']
                chunk_embeddings = cached_data['chunk_embeddings']
            else:
                print(f"🔄 Processing embeddings for this question...")
                
                # Get question embedding for similarity comparison
                question_embedding = safe_embed_query(embeddings, question_text)
                
                # Get chunk embeddings with timeout handling
                chunk_embeddings = []
                successful_chunks = 0
                for j, chunk in enumerate(chunks):
                    print(f"   Processing chunk {j+1}/{len(chunks)} (successful: {successful_chunks})...", end='\r')
                    try:
                        chunk_embedding = safe_embed_query(embeddings, chunk)
                        chunk_embeddings.append(chunk_embedding)
                        successful_chunks += 1
                    except Exception as e:
                        print(f"\n⚠️  Error processing chunk {j+1}: {e}")
                        continue
                
                # Only cache if we got all embeddings
                if len(chunk_embeddings) == len(chunks):
                    cache_embeddings(question_text, context_text, question_embedding, chunk_embeddings, cache_data)
                    print(f"\n💾 Cached embeddings for this question")
                else:
                    print(f"\n⚠️  Skipped caching due to incomplete embeddings ({len(chunk_embeddings)}/{len(chunks)})")
            
            # Calculate similarity for each chunk (question-to-chunk similarity)
            chunk_similarities = []
            for j, chunk_embedding in enumerate(chunk_embeddings):
                similarity = np.dot(question_embedding, chunk_embedding) / (
                    np.linalg.norm(question_embedding) * np.linalg.norm(chunk_embedding)
                )
                chunk_similarities.append({
                    'chunk_index': j,
                    'chunk_text': chunks[j],
                    'similarity': similarity
                })
            
            print(f"\n✅ Completed processing {len(chunks)} chunks")
            
            # Sort by similarity (highest first)
            chunk_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Save cache
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"💾 Saved cache with {len(cache_data)} embeddings")
            except Exception as e:
                print(f"⚠️  Warning: Could not save cache: {e}")
            
            # Format results
            results = []
            results.append(f"Query: {query}")
            results.append(f"Best Matching Question: {best_question_doc['question']}")
            results.append(f"Answer: {best_question_doc['answer']}")
            results.append(f"Question Similarity: {best_question_similarity:.4f}")
            results.append("")
            results.append("📋 Top 10 Most Relevant Context Chunks (by similarity):")
            
            for j, chunk_data in enumerate(chunk_similarities[:10], 1):
                chunk_text = chunk_data['chunk_text']
                similarity = chunk_data['similarity']
                results.append(f"   {j:2d}. Similarity: {similarity:.4f}")
                results.append(f"       {chunk_text[:200]}...")
                if len(chunk_text) > 200:
                    results.append(f"       (Length: {len(chunk_text)} characters)")
                results.append("")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error in RAG search: {str(e)}"

    @tool
    def formula_analysis_tool(question: str) -> str:
        """
        Extract the mathematical formula and key terms needed to answer a financial question.
        Use for: Understanding what calculation is needed and what data points to search for.
        Input: A financial question (e.g., 'What is the gross profit margin?', 'Calculate the P/E ratio')
        """
        try:
            # Get LLM from the agent instance
            llm = ChatOpenAI(
                temperature=0.1,
                model="gpt-3.5-turbo",
            )
            
            # Step 1: Get formula
            formula_prompt = f"""
            You are a financial analysis expert. Given this question:
            
            QUESTION: {question}
            
            Please provide ONLY the mathematical formula required to calculate the answer.
            Respond with just the formula, nothing else.
            
            Examples:
            - For "What is the gross profit margin?" → "Gross Profit Margin = (Revenue - Cost of Goods Sold) / Revenue"
            - For "What is EBITDA?" → "EBITDA = Net Income + Interest + Taxes + Depreciation + Amortization"
            - For "What is the current ratio?" → "Current Ratio = Current Assets / Current Liabilities"
            """
            
            formula_response = llm.invoke(formula_prompt)
            formula = formula_response.content.strip()
            
            # Step 2: Extract key terms and synonyms
            analysis_prompt = f"""
            You are a financial analysis expert. Given this formula:
            
            FORMULA: {formula}
            
            Please provide:
            1. The KEY TERMS needed from the formula (e.g., "revenue", "cost of goods sold")
            2. SYNONYMS for each key term (e.g., "revenue" → ["total revenue", "net sales", "sales", "gross sales", "total income"])
            
            Respond in this exact format:
            KEY_TERMS: [term1, term2, term3, ...]
            SYNONYMS: {{"term1": ["synonym1", "synonym2"], "term2": ["synonym1", "synonym2"], ...}}
            """
            
            analysis_response = llm.invoke(analysis_prompt)
            
            # Parse the response to extract key terms and synonyms
            key_terms = []
            synonyms = {}
            
            lines = analysis_response.content.split('\n')
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith('KEY_TERMS:'):
                    terms_str = line_stripped.replace('KEY_TERMS:', '').strip()
                    key_terms = [term.strip() for term in terms_str.strip('[]').split(',')]
                elif line_stripped.startswith('SYNONYMS:'):
                    # Try to parse the LLM's synonym response
                    synonyms_str = line_stripped.replace('SYNONYMS:', '').strip()
                    
                    # If this line doesn't contain the full JSON, look for the complete JSON structure
                    if not synonyms_str.startswith('{'):
                        # Find the complete JSON structure in the response
                        synonyms_start = analysis_response.content.find('SYNONYMS:')
                        if synonyms_start != -1:
                            synonyms_section = analysis_response.content[synonyms_start:]
                            # Find the opening brace
                            brace_start = synonyms_section.find('{')
                            if brace_start != -1:
                                # Find the matching closing brace
                                brace_count = 0
                                brace_end = -1
                                for i, char in enumerate(synonyms_section[brace_start:], brace_start):
                                    if char == '{':
                                        brace_count += 1
                                    elif char == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            brace_end = i + 1
                                            break
                                
                                if brace_end != -1:
                                    synonyms_str = synonyms_section[brace_start:brace_end]
                    
                    try:
                        # Try to parse as JSON
                        import json
                        synonyms = json.loads(synonyms_str)
                    except:
                        # If parsing fails, create synonyms based on key terms
                        synonyms = {}
                        for term in key_terms:
                            term_lower = term.lower()
                            if "revenue" in term_lower:
                                synonyms[term] = ["total revenue", "net sales", "sales", "revenue", "income", "net revenue"]
                            elif "cost" in term_lower and "goods" in term_lower:
                                synonyms[term] = ["cost of goods sold", "cogs", "merchandise costs", "cost of sales", "cost of merchandise sold", "merchandise costs"]
                            elif "gross" in term_lower and "profit" in term_lower:
                                synonyms[term] = ["gross profit", "gross margin dollars", "gross income", "gross profit margin"]
                            elif "operating" in term_lower and "income" in term_lower:
                                synonyms[term] = ["operating income", "operating profit", "ebit", "operating earnings"]
                            elif "net" in term_lower and "income" in term_lower:
                                synonyms[term] = ["net income", "net profit", "earnings", "net earnings", "net income attributable to costco"]
                            elif "ebitda" in term_lower:
                                synonyms[term] = ["ebitda", "earnings before interest taxes depreciation amortization"]
                            elif "depreciation" in term_lower:
                                synonyms[term] = ["depreciation", "depreciation and amortization", "d&a", "depreciation and amortization expense"]
                            elif "merchandise" in term_lower and "costs" in term_lower:
                                synonyms[term] = ["merchandise costs", "cost of goods sold", "cogs", "cost of sales", "merchandise expenses"]
                            elif "sales" in term_lower and "net" in term_lower:
                                synonyms[term] = ["net sales", "total revenue", "revenue", "sales", "net revenue"]
                            else:
                                # Default: just use the term itself
                                synonyms[term] = [term.lower()]
            
            # Format the response
            response = f"Formula: {formula}\n\nKey Terms: {key_terms}\n\nSynonyms: {synonyms}"
            
            return response
            
        except Exception as e:
            return f"Error in formula analysis: {str(e)}"

    @tool
    def key_terms_search_tool(key_terms: str, synonyms: str, context: str) -> str:
        """
        Search for specific key terms in a given context and return relevant sentences.
        Use for: Finding specific financial data points in documents after formula analysis.
        Input: key_terms (comma-separated), synonyms (JSON format), context (text to search in)
        """
        try:
            # Parse key terms and synonyms
            import json
            
            # Convert string representations back to lists/dicts
            key_terms_list = [term.strip().strip('"') for term in key_terms.strip('[]').split(',')]
            synonyms_dict = json.loads(synonyms.replace("'", '"'))
            
            # Create a comprehensive search query from key terms and synonyms
            all_search_terms = []
            for term in key_terms_list:
                # Clean up the term - remove quotes if present
                clean_term = term.strip().strip('"').strip("'")
                if clean_term in synonyms_dict:
                    all_search_terms.extend(synonyms_dict[clean_term])
                else:
                    all_search_terms.append(clean_term)
            
            # Simple direct search for terms in context
            extracted_info = {}
            context_lower = context.lower()
            
            for term in all_search_terms:
                term_lower = term.lower()
                
                if term_lower in context_lower:
                    # Find the position of the term
                    pos = context_lower.find(term_lower)
                    
                    # Extract surrounding text (approximately 50 characters before and after)
                    start = max(0, pos - 50)
                    end = min(len(context), pos + len(term) + 50)
                    
                    # Try to find sentence boundaries
                    start = context.rfind('.', start, pos) + 1 if context.rfind('.', start, pos) > start - 100 else start
                    end = context.find('.', pos) + 1 if context.find('.', pos) < end + 100 else end
                    
                    relevant_text = context[start:end].strip()
                    if relevant_text:
                        extracted_info[term] = relevant_text
                else:
                    # Try partial matches for multi-word terms
                    if ' ' in term_lower:
                        words = term_lower.split()
                        for word in words:
                            if len(word) > 3 and word in context_lower:  # Only search for words longer than 3 chars
                                # Find the position of the partial match
                                pos = context_lower.find(word)
                                
                                # Extract surrounding text (approximately 50 characters before and after)
                                start = max(0, pos - 50)
                                end = min(len(context), pos + len(word) + 50)
                                
                                # Try to find sentence boundaries
                                start = context.rfind('.', start, pos) + 1 if context.rfind('.', start, pos) > start - 100 else start
                                end = context.find('.', pos) + 1 if context.find('.', pos) < end + 100 else end
                                
                                relevant_text = context[start:end].strip()
                                if relevant_text:
                                    extracted_info[f"{term} (partial: {word})"] = relevant_text
                                    break
            
            if extracted_info:
                # Convert dictionary to a single string
                extracted_text = "\n\n".join([f"{term}: {text}" for term, text in extracted_info.items()])
                return f"Found relevant information:\n\n{extracted_text}"
            else:
                return "No relevant terms found in the provided context."
                
        except Exception as e:
            return f"Error in key terms search: {str(e)}"
    
    @tool
    def direct_rag_search_tool(question_num: int) -> str:
        """
        Direct RAG search for a specific question number from the FinanceQA dataset.
        Use for: Getting cached embeddings for a specific question without similarity search.
        Input: A question number (1-148)
        """
        try:
            # Load the FinanceQA dataset
            import json
            import pickle
            import hashlib
            import signal
            import numpy as np
            from pathlib import Path
            
            def timeout_handler(signum, frame):
                """Handle timeout for API calls"""
                raise TimeoutError("API call timed out")
            
            def safe_embed_query(embeddings, text, timeout=30):
                """Safely embed text with timeout"""
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
                try:
                    result = embeddings.embed_query(text)
                    signal.alarm(0)  # Cancel timeout
                    return result
                except TimeoutError:
                    signal.alarm(0)  # Cancel timeout
                    raise TimeoutError(f"Embedding request timed out after {timeout} seconds")
                except Exception as e:
                    signal.alarm(0)  # Cancel timeout
                    raise e
            
            def split_context_into_chunks(context_text: str, chunk_size: int = 200, overlap: int = 100) -> List[str]:
                """Split context text into meaningful chunks with overlap."""
                if not context_text:
                    return []
                
                # Split by sentences first
                sentences = context_text.split('. ')
                
                chunks = []
                current_chunk = ""
                overlap_text = ""
                
                for sentence in sentences:
                    # Add period back if it was removed
                    if not sentence.endswith('.'):
                        sentence += '.'
                    
                    # If adding this sentence would make chunk too long, start new chunk
                    if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                        # Add current chunk
                        chunks.append(current_chunk.strip())
                        
                        # Calculate overlap: take the last portion of current chunk
                        if overlap > 0 and len(current_chunk) > overlap:
                            # Find the last sentence boundary within the overlap region
                            overlap_region = current_chunk[-overlap:]
                            last_sentence_start = overlap_region.find('. ')
                            if last_sentence_start != -1:
                                overlap_text = overlap_region[last_sentence_start + 2:]  # +2 to skip '. '
                            else:
                                overlap_text = overlap_region
                        else:
                            overlap_text = current_chunk
                        
                        # Start new chunk with overlap
                        current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                
                # Add the last chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                return chunks
            
            def get_cache_key(question_text: str, context_text: str) -> str:
                """Generate a unique cache key for a question and its context."""
                content = f"{question_text}|{context_text}"
                return hashlib.md5(content.encode('utf-8')).hexdigest()
            
            def get_cached_embeddings(question_text: str, context_text: str, cache_data: dict) -> Optional[dict]:
                """Get cached embeddings for a question and context."""
                cache_key = get_cache_key(question_text, context_text)
                return cache_data.get(cache_key)
            
            def cache_embeddings(question_text: str, context_text: str, question_embedding: List[float], 
                               chunk_embeddings: List[List[float]], cache_data: dict):
                """Cache embeddings for a question and context."""
                cache_key = get_cache_key(question_text, context_text)
                cache_data[cache_key] = {
                    'question_embedding': question_embedding,
                    'chunk_embeddings': chunk_embeddings,
                    'timestamp': os.path.getmtime('data/financeqa_test.jsonl')
                }
            
            data_path = Path("data/financeqa_test.jsonl")
            if not data_path.exists():
                return "FinanceQA dataset not found. Please ensure data/financeqa_test.jsonl exists."
            
            # Load all questions and contexts
            documents = []
            with open(data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        if doc.get('context'):
                            documents.append({
                                'question': doc.get('question', ''),
                                'context': doc.get('context', ''),
                                'answer': doc.get('answer', '')
                            })
            
            if not documents:
                return "No documents found in the FinanceQA dataset."
            
            # Validate question number
            if question_num < 1 or question_num > len(documents):
                return f"Invalid question number. Please enter a number between 1 and {len(documents)}"
            
            # Get the specific question
            selected_doc = documents[question_num - 1]
            question_text = selected_doc['question']
            context_text = selected_doc['context']
            answer_text = selected_doc['answer']
            
            print(f"📋 Question {question_num}: {question_text}")
            print(f"📝 Answer: {answer_text}")
            
            # Load embeddings cache
            cache_path = Path("data/embeddings_cache.pkl")
            cache_data = {}
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                    print(f"📦 Loaded cache with {len(cache_data)} cached embeddings")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load cache: {e}")
            else:
                print("📝 No cache file found, will create new embeddings")
            
            # Initialize embeddings
            embeddings = OpenAIEmbeddings()
            
            # Split context into chunks
            chunks = split_context_into_chunks(context_text)
            
            print(f"📊 Context Analysis:")
            print(f"   Total context length: {len(context_text)} characters")
            print(f"   Number of chunks: {len(chunks)}")
            print(f"   Average chunk length: {sum(len(chunk) for chunk in chunks)/len(chunks):.0f} characters")
            print(f"   Chunk size: 200 characters with 100 character overlap")
            
            # Check cache first using question-specific key
            cached_data = get_cached_embeddings(question_text, context_text, cache_data)
            
            if cached_data:
                print(f"⚡ Using cached embeddings for question {question_num}")
                question_embedding = cached_data['question_embedding']
                chunk_embeddings = cached_data['chunk_embeddings']
            else:
                print(f"🔄 Processing embeddings for question {question_num}...")
                
                # Get question embedding for similarity comparison
                question_embedding = safe_embed_query(embeddings, question_text)
                
                # Get chunk embeddings with timeout handling
                chunk_embeddings = []
                successful_chunks = 0
                for j, chunk in enumerate(chunks):
                    print(f"   Processing chunk {j+1}/{len(chunks)} (successful: {successful_chunks})...", end='\r')
                    try:
                        chunk_embedding = safe_embed_query(embeddings, chunk)
                        chunk_embeddings.append(chunk_embedding)
                        successful_chunks += 1
                    except Exception as e:
                        print(f"\n⚠️  Error processing chunk {j+1}: {e}")
                        continue
                
                # Only cache if we got all embeddings
                if len(chunk_embeddings) == len(chunks):
                    cache_embeddings(question_text, context_text, question_embedding, chunk_embeddings, cache_data)
                    print(f"\n💾 Cached embeddings for question {question_num}")
                else:
                    print(f"\n⚠️  Skipped caching due to incomplete embeddings ({len(chunk_embeddings)}/{len(chunks)})")
            
            # Calculate similarity for each chunk (question-to-chunk similarity)
            chunk_similarities = []
            for j, chunk_embedding in enumerate(chunk_embeddings):
                similarity = np.dot(question_embedding, chunk_embedding) / (
                    np.linalg.norm(question_embedding) * np.linalg.norm(chunk_embedding)
                )
                chunk_similarities.append({
                    'chunk_index': j,
                    'chunk_text': chunks[j],
                    'similarity': similarity
                })
            
            print(f"\n✅ Completed processing {len(chunks)} chunks")
            
            # Sort by similarity (highest first)
            chunk_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Save cache
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"💾 Saved cache with {len(cache_data)} embeddings")
            except Exception as e:
                print(f"⚠️  Warning: Could not save cache: {e}")
            
            # Format results
            results = []
            results.append(f"Question {question_num}: {question_text}")
            results.append(f"Answer: {answer_text}")
            results.append("")
            results.append("📋 Top 10 Most Relevant Context Chunks (by similarity):")
            
            for j, chunk_data in enumerate(chunk_similarities[:10], 1):
                chunk_text = chunk_data['chunk_text']
                similarity = chunk_data['similarity']
                results.append(f"   {j:2d}. Similarity: {similarity:.4f}")
                results.append(f"       {chunk_text[:200]}...")
                if len(chunk_text) > 200:
                    results.append(f"       (Length: {len(chunk_text)} characters)")
                results.append("")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error in direct RAG search: {str(e)}"

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
        """Main method to process a financial question using the custom control flow."""
        start_time = datetime.now()
        try:
            # Import control flow functions
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from control import (
                analyze_question_requirements, 
                extract_formula_from_analysis,
                search_key_terms_in_context,
                extract_values_from_context,
                assess_information_completeness,
                iterative_tool_selection_loop,
                extract_formula_values_for_calculator
            )
            
            # Create question data structure
            question_data = {
                'question_num': 1,  # Default for single question processing
                'question': question,
                'context': '',  # Will be populated if available
                'expected_answer': 'N/A',
                'company': 'N/A',
                'question_type': 'N/A'
            }
            
            # Analyze question requirements
            analysis = analyze_question_requirements(question, self)
            formula = extract_formula_from_analysis(analysis)
            
            # Search for key terms in context (if available)
            search_result = ""
            extraction_result = {'values': {}, 'calculation': '', 'result': ''}
            
            # Assess information completeness
            assessment_result = assess_information_completeness(question_data, formula, extraction_result, self)
            
            # If assessment says "No", enter iterative tool selection loop
            if assessment_result['complete_answer'] == "No":
                print(f"\n🔄 Entering iterative tool selection loop...")
                final_result = iterative_tool_selection_loop(question_data, formula, self, assessment_result)
                answer = final_result.get('final_answer', 'No final answer generated')
            else:
                # Assessment says "Yes" - extract formula values and calculate
                print(f"\n🧮 Complete answer possible! Extracting formula values...")
                
                # Create working answer
                working_answer = {
                    'question_num': question_data['question_num'],
                    'question': question_data['question'],
                    'formula': formula,
                    'tool_results': [],
                    'extraction_result': extraction_result,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Extract formula values and prepare for calculator
                calculator_expression = extract_formula_values_for_calculator(
                    question_data, formula, working_answer, self
                )
                
                if calculator_expression and calculator_expression.get('calculator_expression'):
                    try:
                        calculator_result = self.financial_calculator_tool.invoke({
                            "expression": calculator_expression['calculator_expression']
                        })
                        answer = f"Final Result: {calculator_result}"
                    except Exception as e:
                        answer = f"Error in calculation: {str(e)}"
                else:
                    answer = "Unable to extract formula values for calculation"
            
            processing_time = (datetime.now() - start_time).total_seconds()
            return {"question": question, "answer": answer, "success": True, "processing_time": processing_time}
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return {"question": question, "answer": f"Error processing question: {str(e)}", "success": False, "processing_time": processing_time, "error": str(e)}

# (The FinanceAgentCLI class remains the same, no changes needed there)
class FinanceAgentCLI:
    def __init__(self): self.agent = FinanceQAAgent()
    def run(self):
        print("\nWelcome to FinanceQA AI Agent!")
        print("=" * 50)
        while True:
            print("\nSelect from the following options:")
            print("1. Pick Question")
            print("2. Run Demo of Agent Thought Process")
            print("3. Exit")
            choice = input("\n> ").strip()
            if choice == "1": asyncio.run(self.run_pick_question())
            elif choice == "2": asyncio.run(self.run_demo_thought_process())
            elif choice == "3": print("Goodbye!"); break
            else: print("Invalid option. Please select 1-3.")
    
    async def run_pick_question(self):
        """Allow user to pick a specific question from the dataset."""
        print("\nPick Question from FinanceQA Dataset")
        print("Type 'quit' to return to main menu.")
        
        # Load the FinanceQA dataset
        try:
            import json
            from pathlib import Path
            
            data_path = Path("data/financeqa_test.jsonl")
            if not data_path.exists():
                print("FinanceQA dataset not found. Please run download_financeqa.py first.")
                return
            
            # Load all questions
            questions = []
            with open(data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
            
            print(f"Loaded {len(questions)} questions from FinanceQA dataset")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
        
        while True:
            print(f"\nChoose a question from 1-{len(questions)} (or type 'quit' to exit):")
            choice = input("> ").strip()
            
            if choice.lower() in ['quit', 'exit', 'q']:
                break
            
            try:
                question_num = int(choice)
                if 1 <= question_num <= len(questions):
                    # Get the selected question
                    selected_question = questions[question_num - 1]
                    
                    # Create proper question data structure
                    question_data = {
                        'question_num': question_num,
                        'question': selected_question.get('question', 'N/A'),
                        'context': selected_question.get('context', ''),
                        'expected_answer': selected_question.get('answer', 'N/A'),
                        'company': selected_question.get('company', 'N/A'),
                        'question_type': selected_question.get('question_type', 'N/A')
                    }
                    
                    print(f"\nQuestion #{question_num}:")
                    print(f"Type: {question_data['question_type']}")
                    print(f"Company: {question_data['company']}")
                    print(f"Question: {question_data['question']}")
                    print(f"Expected Answer: {question_data['expected_answer']}")
                    
                    # Import and use the control flow logic
                    import sys
                    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                    from control import display_question_info
                    
                    # Use the control flow logic
                    display_question_info(question_data)
                    
                else:
                    print(f"Please enter a number between 1 and {len(questions)}")
                    
            except ValueError:
                print("Please enter a valid number or 'quit'")
            except Exception as e:
                print(f"Error processing question: {e}")
    
    async def run_demo_thought_process(self):
        """Run a demo showing the agent's thought process."""
        print("\nDemo of Agent Thought Process")
        print("=" * 40)
        
        # Load the FinanceQA dataset
        try:
            import json
            from pathlib import Path
            
            data_path = Path("data/financeqa_test.jsonl")
            if not data_path.exists():
                print("FinanceQA dataset not found. Please run download_financeqa.py first.")
                return
            
            # Load all questions
            questions = []
            with open(data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
            
            # Take the first 4 questions
            demo_questions = questions[:4]
            print(f"Loaded first 4 questions from FinanceQA dataset")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
        
        for i, question_data in enumerate(demo_questions, 1):
            print(f"\n--- Demo {i}/{len(demo_questions)} ---")
            print(f"Type: {question_data.get('question_type', 'N/A')}")
            print(f"Company: {question_data.get('company', 'N/A')}")
            print(f"Question: {question_data.get('question', 'N/A')}")
            print(f"Expected Answer: {question_data.get('answer', 'N/A')}")
            
            # Create proper question data structure with context
            processed_question_data = {
                'question_num': i,
                'question': question_data.get('question', 'N/A'),
                'context': question_data.get('context', ''),
                'expected_answer': question_data.get('answer', 'N/A'),
                'company': question_data.get('company', 'N/A'),
                'question_type': question_data.get('question_type', 'N/A')
            }
            
            print("\nAgent thinking...")
            
            # Import and use the control flow logic
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from control import display_question_info
            
            # Use the control flow logic
            display_question_info(processed_question_data)
            
            if i < len(demo_questions): 
                input("\nPress Enter to continue...")
    
    async def run_prebuilt_demo(self):
        print("\nPre-Built Demo Starting...")
        
        # Load the FinanceQA dataset
        try:
            import json
            from pathlib import Path
            
            data_path = Path("data/financeqa_test.jsonl")
            if not data_path.exists():
                print("FinanceQA dataset not found. Please run download_financeqa.py first.")
                return
            
            # Load all questions
            questions = []
            with open(data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
            
            # Take the first 4 questions
            demo_questions = questions[:4]
            print(f"Loaded first 4 questions from FinanceQA dataset")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
        
        for i, question_data in enumerate(demo_questions, 1):
            print(f"\n--- Demo {i}/{len(demo_questions)} ---")
            print(f"Type: {question_data.get('question_type', 'N/A')}")
            print(f"Company: {question_data.get('company', 'N/A')}")
            print(f"Question: {question_data.get('question', 'N/A')}")
            print(f"Expected Answer: {question_data.get('answer', 'N/A')}")
            
            # Create proper question data structure with context
            processed_question_data = {
                'question_num': i,
                'question': question_data.get('question', 'N/A'),
                'context': question_data.get('context', ''),
                'expected_answer': question_data.get('answer', 'N/A'),
                'company': question_data.get('company', 'N/A'),
                'question_type': question_data.get('question_type', 'N/A')
            }
            
            print("\nAgent thinking...")
            
            # Import and use the control flow logic
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from control import display_question_info
            
            # Use the control flow logic
            display_question_info(processed_question_data)
            
            # For compatibility, create a result object
            result = {
                'answer': 'Processed using control flow logic',
                'processing_time': 0.0
            }
            
            print(f"\nAI Answer: {result['answer']}")
            print(f"Processing time: {result['processing_time']:.2f}s")
            
            # Show comparison
            print(f"\nComparison:")
            print(f"Expected: {question_data.get('answer', 'N/A')}")
            print(f"AI Answer: {result['answer']}")
            
            if i < len(demo_questions): 
                input("\nPress Enter to continue...")
    async def run_live_demo(self):
        print("\nLive Demo - Choose from FinanceQA Dataset!")
        print("Type 'quit' to return to main menu.")
        
        # Load the FinanceQA dataset
        try:
            import json
            from pathlib import Path
            
            data_path = Path("data/financeqa_test.jsonl")
            if not data_path.exists():
                print("FinanceQA dataset not found. Please run download_financeqa.py first.")
                return
            
            # Load all questions
            questions = []
            with open(data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
            
            print(f"Loaded {len(questions)} questions from FinanceQA dataset")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
        
        while True:
            print(f"\nChoose a question from 1-{len(questions)} (or type 'quit' to exit):")
            choice = input("> ").strip()
            
            if choice.lower() in ['quit', 'exit', 'q']:
                break
            
            try:
                question_num = int(choice)
                if 1 <= question_num <= len(questions):
                    # Get the selected question
                    selected_question = questions[question_num - 1]
                    
                    print(f"\nQuestion #{question_num}:")
                    print(f"Type: {selected_question.get('question_type', 'N/A')}")
                    print(f"Company: {selected_question.get('company', 'N/A')}")
                    print(f"Question: {selected_question.get('question', 'N/A')}")
                    print(f"Expected Answer: {selected_question.get('answer', 'N/A')}")
                    
                    print("\nProcessing with AI Agent...")
                    result = await self.agent.process_question(selected_question['question'])
                    
                    print(f"\nAI Answer: {result['answer']}")
                    print(f"Processing time: {result['processing_time']:.2f}s")
                    
                    # Show comparison
                    print(f"\nComparison:")
                    print(f"Expected: {selected_question.get('answer', 'N/A')}")
                    print(f"AI Answer: {result['answer']}")
                    
                else:
                    print(f"Please enter a number between 1 and {len(questions)}")
                    
            except ValueError:
                print("Please enter a valid number or 'quit'")
            except Exception as e:
                print(f"Error processing question: {e}")

    def run_step_by_step_control(self):
        """Simple question selection and context checking."""
        print("\n🤖 Step-by-Step Agent Control")
        print("=" * 40)
        print("Select a question from the FinanceQA dataset.")
        print("Type 'quit' to return to main menu.\n")
        
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
        
        # Get question selection
        print(f"\n📊 Choose a question from 1-{len(questions)}:")
        choice = input("> ").strip()
        
        if choice.lower() in ['quit', 'exit', 'q']:
            return
        
        try:
            question_num = int(choice)
            if 1 <= question_num <= len(questions):
                # Get the selected question
                selected_question = questions[question_num - 1]
                question = selected_question.get('question', 'N/A')
                context = selected_question.get('context', '')
                expected_answer = selected_question.get('answer', 'N/A')
                
                # Print the question
                print(f"\n📋 Question #{question_num}: {question}")
                
                # Check if context is available and print status
                if context and context.strip():
                    print("✅ CONTEXT FOUND")
                else:
                    print("❌ NO CONTEXT")
                
                print(f"Expected Answer: {expected_answer}")
                
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