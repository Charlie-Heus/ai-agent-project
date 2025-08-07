#!/usr/bin/env python3
"""
FinanceQA AI Agent - Clean Version
==================================

This is a clean version with only the three requested menu options:
1. Pick Question
2. Run Demo of Agent Thought Process  
3. Exit
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
                            earnings_data = earnings_response_data['earnings'][:4]  # Last 4 quarters
                    
                    # Compile comprehensive financial data
                    financial_summary = {
                        'company': profile_data,
                        'metrics': metrics_data,
                        'earnings': earnings_data
                    }
                    
                    return f"Finnhub Financial Data for {ticker}: {json.dumps(financial_summary, indent=2)}"
                else:
                    return f"Error fetching metrics for {ticker}: {metrics_response.status_code}"
            else:
                return f"Error fetching profile for {ticker}: {profile_response.status_code}"
                
        except Exception as e:
            return f"Error accessing Finnhub data: {str(e)}"

    @tool 
    def web_search_tool(query: str) -> str:
        """
        Search the web for current financial information and news.
        Use for: Recent news, market updates, company announcements, financial analysis.
        Input: Search query (e.g., 'Apple Q4 2023 earnings', 'Tesla stock price today')
        """
        try:
            # Use Google Custom Search API
            google_api_key = os.getenv('GOOGLE_API_KEY')
            search_engine_id = os.getenv('CUSTOM_SEARCH_ENGINE_ID')
            
            if not google_api_key or not search_engine_id:
                return "Google API key or Search Engine ID not configured. Cannot perform web search."
            
            # Create the service
            service = build("customsearch", "v1", developerKey=google_api_key)
            
            # Perform the search
            result = service.list(
                q=query,
                cx=search_engine_id,
                num=5  # Get top 5 results
            ).execute()
            
            if 'items' in result:
                search_results = []
                for item in result['items']:
                    search_results.append({
                        'title': item.get('title', 'No title'),
                        'snippet': item.get('snippet', 'No snippet'),
                        'link': item.get('link', 'No link')
                    })
                
                return f"Web Search Results for '{query}': {json.dumps(search_results, indent=2)}"
            else:
                return f"No web search results found for '{query}'"
                
        except Exception as e:
            return f"Error performing web search: {str(e)}"

    @tool
    def fetch_webpage_content_tool(url: str) -> str:
        """
        Fetch and extract text content from a webpage.
        Use for: Getting detailed information from financial websites, SEC filings, company pages.
        Input: URL to fetch (e.g., 'https://www.sec.gov/Archives/edgar/data/...')
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit to first 1500 characters to avoid token limits
            return text[:1500] + "..." if len(text) > 1500 else text
            
        except Exception as e:
            return f"Could not fetch content from {url}: {str(e)}"

    @tool
    def financial_calculator_tool(expression: str) -> str:
        """
        Perform financial calculations safely.
        Use for: Ratios, percentages, basic arithmetic for financial metrics.
        Input: Mathematical expression (e.g., '1000 * 1.05', '50000 / 1000000 * 100')
        """
        try:
            # Use numexpr for safe evaluation
            result = numexpr.evaluate(expression)
            return f"Calculation result: {expression} = {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}. Please provide a valid mathematical expression."

    @tool
    def knowledge_base_tool(query: str) -> str:
        """
        Search the financial knowledge base for definitions and concepts.
        Use for: Understanding financial terms, ratios, and concepts.
        Input: Financial term or concept (e.g., 'P/E ratio', 'EBITDA', 'debt to equity')
        """
        # Simple knowledge base - can be expanded
        knowledge_base = {
            'pe ratio': 'Price-to-Earnings ratio = Market Price per Share / Earnings per Share. Measures how much investors are willing to pay for each dollar of earnings.',
            'ebitda': 'Earnings Before Interest, Taxes, Depreciation, and Amortization. A measure of a company\'s operating performance.',
            'debt to equity': 'Total Debt / Total Shareholders\' Equity. Measures financial leverage and risk.',
            'market cap': 'Market Capitalization = Current Stock Price × Total Shares Outstanding. Total value of a company\'s shares.',
            'roe': 'Return on Equity = Net Income / Shareholders\' Equity. Measures profitability relative to equity.',
            'roa': 'Return on Assets = Net Income / Total Assets. Measures how efficiently assets generate earnings.',
            'current ratio': 'Current Assets / Current Liabilities. Measures short-term liquidity.',
            'quick ratio': '(Current Assets - Inventory) / Current Liabilities. More conservative liquidity measure.',
            'gross margin': 'Gross Profit / Revenue. Measures profitability after direct costs.',
            'operating margin': 'Operating Income / Revenue. Measures profitability from core operations.',
            'net margin': 'Net Income / Revenue. Overall profitability measure.',
            'asset turnover': 'Revenue / Average Total Assets. Measures asset efficiency.',
            'inventory turnover': 'Cost of Goods Sold / Average Inventory. Measures inventory efficiency.',
            'days sales outstanding': 'Accounts Receivable / (Revenue / 365). Average collection period.',
            'days payable outstanding': 'Accounts Payable / (Cost of Goods Sold / 365). Average payment period.',
            'cash conversion cycle': 'Days Sales Outstanding + Days Inventory Outstanding - Days Payable Outstanding.',
            'free cash flow': 'Operating Cash Flow - Capital Expenditures. Cash available for investors.',
            'working capital': 'Current Assets - Current Liabilities. Short-term financial health.',
            'book value': 'Total Assets - Total Liabilities. Net asset value per accounting.',
            'tangible book value': 'Book Value - Intangible Assets. Net asset value excluding intangibles.'
        }
        
        query_lower = query.lower()
        for term, definition in knowledge_base.items():
            if term in query_lower:
                return f"Definition of '{term}': {definition}"
        
        return f"No definition found for '{query}'. Available terms: {list(knowledge_base.keys())}"

    @tool
    def rag_search_tool(query: str) -> str:
        """
        Search the FinanceQA dataset using RAG (Retrieval-Augmented Generation).
        Use for: Finding relevant financial data and context from the dataset.
        Input: Search query (e.g., 'Costco revenue 2024', 'gross profit calculation')
        """
        try:
            # Load the FinanceQA dataset
            data_path = Path("data/financeqa_test.jsonl")
            if not data_path.exists():
                return "FinanceQA dataset not found. Please run download_financeqa.py first."
            
            # Load questions and contexts
            questions = []
            with open(data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
            
            # Simple keyword-based search (can be enhanced with embeddings)
            relevant_results = []
            
            # Handle None or empty query
            if not query or query.strip() == "":
                return "Error: No search query provided for RAG search."
            
            query_lower = query.lower()
            
            for q in questions:
                question_text = q.get('question', '')
                context_text = q.get('context', '')
                
                # Handle None values safely
                if question_text is None:
                    question_text = ""
                if context_text is None:
                    context_text = ""
                
                question_text = question_text.lower()
                context_text = context_text.lower()
                
                # Check if query terms appear in question or context
                query_terms = query_lower.split()
                if any(term in question_text or term in context_text for term in query_terms):
                    # Handle None context safely
                    context = q.get('context', '')
                    if context is None:
                        context = ""
                    
                    context_preview = context[:200] + "..." if len(context) > 200 else context
                    
                    relevant_results.append({
                        'question': q.get('question', ''),
                        'context_preview': context_preview,
                        'answer': q.get('answer', '')
                    })
            
            if relevant_results:
                return f"RAG Search Results for '{query}': {json.dumps(relevant_results[:3], indent=2)}"  # Top 3 results
            else:
                return f"No relevant results found in FinanceQA dataset for '{query}'"
                
        except Exception as e:
            return f"Error performing RAG search: {str(e)}"

    @tool
    def formula_analysis_tool(question: str) -> str:
        """
        Analyze a financial question to extract the required formula and key terms.
        Use for: Understanding what calculation is needed and what data points to find.
        Input: Financial question (e.g., 'What is the P/E ratio?', 'Calculate gross profit margin')
        """
        try:
            # Create LLM instance for this tool
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                temperature=0.1,
                model="gpt-3.5-turbo",
            )
            
            # Use the LLM to analyze the question
            analysis_prompt = f"""
            Analyze this financial question and extract:
            1. The formula needed for calculation
            2. Key terms that need to be found
            3. Synonyms for those key terms
            
            Question: {question}
            
            IMPORTANT GUIDELINES:
            - For EBITDA questions: Include "Net Income", "Interest", "Taxes", "Depreciation", "Amortization" as key terms
            - For ADJUSTED EBITDA questions: Include "EBITDA", "Adjustments", "Net Income", "Interest", "Taxes", "Depreciation", "Amortization" as key terms
            - For ADJUSTED EBIT questions: Include "EBIT", "Adjustments", "Revenue", "Cost of Goods Sold", "Operating Expenses" as key terms
            - For EBIT questions: Include "Revenue", "Cost of Goods Sold", "Operating Expenses" as key terms
            - For Gross Profit questions: Include "Revenue", "Cost of Goods Sold" as key terms
            - For ratio questions: Include both numerator and denominator terms
            - Include the main metric being asked for (e.g., "EBITDA", "Adjusted EBITDA", "EBIT", "Adjusted EBIT", "Gross Profit", "P/E Ratio")
            - Include year/time period if mentioned (e.g., "2024", "year ending 2024")
            
            FORMULA EXAMPLES:
            - EBITDA = Net Income + Interest + Taxes + Depreciation + Amortization
            - Adjusted EBITDA = EBITDA + Adjustments
            - EBIT = Revenue - Cost of Goods Sold - Operating Expenses
            - Adjusted EBIT = EBIT + Adjustments
            - Gross Profit = Revenue - Cost of Goods Sold
            
            IMPORTANT: Keep formulas simple and consistent. For "Adjusted" metrics, use the base metric + adjustments.
            
            Provide your analysis in this exact format:
            Formula: [the mathematical formula]
            Key Terms: [list of terms needed]
            Synonyms: [dictionary of synonyms for each key term]
            """
            
            response = llm.invoke(analysis_prompt)
            return response.content
            
        except Exception as e:
            return f"Error analyzing question: {str(e)}"

    @tool
    def key_terms_search_tool(key_terms: str, synonyms: str, context: str) -> str:
        """
        Search for specific key terms in a given context.
        Use for: Finding specific financial data points in documents or contexts.
        Input: key_terms (comma-separated), synonyms (JSON format), context (text to search in)
        """
        try:
            # Parse key terms and synonyms
            terms = [term.strip() for term in key_terms.split(',')]
            synonyms_dict = json.loads(synonyms) if synonyms else {}
            
            # Create search patterns
            search_patterns = []
            for term in terms:
                search_patterns.append(term)
                if term in synonyms_dict:
                    search_patterns.extend(synonyms_dict[term])
            
            # Search in context
            context_lower = context.lower()
            found_matches = []
            
            for pattern in search_patterns:
                if pattern.lower() in context_lower:
                    # Find the surrounding text
                    start_idx = context_lower.find(pattern.lower())
                    end_idx = start_idx + len(pattern)
                    
                    # Extract surrounding context (50 chars before and after)
                    context_start = max(0, start_idx - 50)
                    context_end = min(len(context), end_idx + 50)
                    
                    match_text = context[context_start:context_end]
                    found_matches.append({
                        'term': pattern,
                        'context': match_text,
                        'position': start_idx
                    })
            
            if found_matches:
                return f"Found relevant information:\n\n" + "\n\n".join([
                    f"{match['term']}: {match['context']}" for match in found_matches
                ])
            else:
                return f"No matches found for terms: {', '.join(terms)}"
                
        except Exception as e:
            return f"Error searching for key terms: {str(e)}"

    @tool
    def direct_rag_search_tool(question_num: int) -> str:
        """
        Directly search for a specific question number in the FinanceQA dataset.
        Use for: Getting the exact context and data for a specific question.
        Input: Question number (integer)
        """
        try:
            # Load the FinanceQA dataset
            data_path = Path("data/financeqa_test.jsonl")
            if not data_path.exists():
                return "FinanceQA dataset not found. Please run download_financeqa.py first."
            
            # Load questions
            questions = []
            with open(data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
            
            # Get the specific question
            if 1 <= question_num <= len(questions):
                question_data = questions[question_num - 1]
                return f"Question #{question_num}:\nQuestion: {question_data.get('question', '')}\nContext: {question_data.get('context', '')[:500]}...\nAnswer: {question_data.get('answer', '')}"
            else:
                return f"Question number {question_num} not found. Available questions: 1-{len(questions)}"
                
        except Exception as e:
            return f"Error accessing question {question_num}: {str(e)}"

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
                        answer = f"📊 Final Result: {calculator_result}\n🎯 Expected Answer: {question_data.get('expected_answer', 'N/A')}"
                    except Exception as e:
                        answer = f"Error in calculation: {str(e)}"
                else:
                    answer = "Unable to extract formula values for calculation"
            
            processing_time = (datetime.now() - start_time).total_seconds()
            return {"question": question, "answer": answer, "success": True, "processing_time": processing_time}
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return {"question": question, "answer": f"Error processing question: {str(e)}", "success": False, "processing_time": processing_time, "error": str(e)}


class FinanceAgentCLI:
    def __init__(self): 
        self.agent = FinanceQAAgent()
    
    def run(self):
        print("\nWelcome to FinanceQA AI Agent!")
        print("=" * 50)
        while True:
            print("\nSelect from the following options:")
            print("1. Pick Question")
            print("2. Run Demo of Agent Thought Process")
            print("3. Exit")
            choice = input("\n> ").strip()
            if choice == "1": 
                asyncio.run(self.run_pick_question())
            elif choice == "2": 
                asyncio.run(self.run_demo_thought_process())
            elif choice == "3": 
                print("Goodbye!"); 
                break
            else: 
                print("Invalid option. Please select 1-3.")
    
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
        
        print(f"\nChoose a question from 1-{len(questions)} (or type 'quit' to exit):")
        choice = input("> ").strip()
        
        if choice.lower() in ['quit', 'exit', 'q']:
            return
        
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
                
                # Use the control flow logic (not demo mode for pick question)
                display_question_info(question_data, demo_mode=False)
                
                # Return to main menu after processing the question
                print(f"\n✅ Question processing completed. Returning to main menu...")
                return
                
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
            
            # Take only the first question
            demo_questions = questions[:1]
            print(f"Loaded first question from FinanceQA dataset")
            
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
            
            # Use the control flow logic (demo mode for demo)
            display_question_info(processed_question_data, demo_mode=True)
            
        print("\n✅ Demo completed! Question processed successfully.")
        print("Exiting...")


def check_environment():
    """Check if required environment variables are available."""
    print("Checking environment...")
    
    required_keys = ['OPENAI_API_KEY', 'GOOGLE_API_KEY', 'CUSTOM_SEARCH_ENGINE_ID']
    optional_keys = ['SEC_API_KEY', 'FINNHUB_API_KEY']
    
    missing_required = [key for key in required_keys if not os.getenv(key)]
    missing_optional = [key for key in optional_keys if not os.getenv(key)]
    
    if missing_required:
        print(f"Missing required API keys: {missing_required}")
        print("Please set these environment variables in your .env file before running.")
        return False
    
    if missing_optional:
        print(f"Missing optional API keys: {missing_optional}. Some features may be limited.")
    
    print("Environment check passed!")
    return True


if __name__ == "__main__":
    if not check_environment():
        print("\nSetup Instructions:")
        print("1. Set up OpenAI, Google Cloud, and Programmable Search Engine to get API keys.")
        print("2. Create a .env file and add your keys (see documentation for format).")
        print("3. Install dependencies: pip install -r requirements.txt (or see file header).")
        exit(1)
    
    # Clear previous working answers at startup
    try:
        from control import clear_working_answers
        clear_working_answers()
    except Exception as e:
        print(f"⚠️  Warning: Could not clear working answers: {str(e)}")
    
    try:
        cli = FinanceAgentCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\nExiting... Goodbye!")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}") 