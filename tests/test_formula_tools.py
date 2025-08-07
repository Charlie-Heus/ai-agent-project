#!/usr/bin/env python3
"""
Test script for the formula analysis tools.
"""

import os
import json
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import FinanceQAAgent

# Load environment variables
load_dotenv()

def load_question_1_from_jsonl():
    """Load the first question from financeqa_test.jsonl."""
    jsonl_path = "data/financeqa_test.jsonl"
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()
            if first_line:
                question_data = json.loads(first_line)
                return {
                    "question": question_data.get("question", ""),
                    "context": question_data.get("context", ""),
                    "answer": question_data.get("answer", ""),
                    "chain_of_thought": question_data.get("chain_of_thought", ""),
                    "company": question_data.get("company", ""),
                    "question_type": question_data.get("question_type", "")
                }
    except FileNotFoundError:
        print(f"❌ Error: {jsonl_path} not found")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in {jsonl_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading question: {e}")
        return None

def test_formula_tools():
    """Test the formula analysis tools with question 1 from financeqa_test.jsonl."""
    
    # Load question 1 from JSONL
    question_data = load_question_1_from_jsonl()
    if not question_data:
        print("❌ Failed to load question data. Using fallback question.")
        test_question = "What is the gross profit margin for the year ending in 2024?"
        test_context = ""
    else:
        test_question = question_data["question"]
        test_context = question_data["context"]
        print(f"📋 Loaded Question 1 from financeqa_test.jsonl:")
        print(f"   Question: {test_question}")
        print(f"   Company: {question_data['company']}")
        print(f"   Type: {question_data['question_type']}")
        print(f"   Expected Answer: {question_data['answer']}")
        print(f"   Context Length: {len(test_context)} characters")
    
    # Initialize FinanceQAAgent
    agent = FinanceQAAgent()
    
    print("\n🧪 Testing Formula Analysis Tools")
    print("=" * 50)
    
    # Test 1: Formula analysis tool
    print("\n🔧 Test 1: Formula Analysis Tool")
    print("-" * 30)
    formula_result = agent.formula_analysis_tool.invoke({"question": test_question})
    print(f"✅ Formula Analysis Result:")
    print(formula_result)
    
    # Test 2: Key terms search tool (if context is available)
    if test_context:
        print("\n🔧 Test 2: Key Terms Search Tool")
        print("-" * 30)
        # We need to extract key terms and synonyms from the formula analysis result
        # For this test, we'll use some sample key terms and synonyms
        sample_key_terms = ["revenue", "cost of goods sold", "gross profit"]
        sample_synonyms = {
            "revenue": ["total revenue", "net sales", "sales", "revenue"],
            "cost of goods sold": ["cogs", "merchandise costs", "cost of sales"],
            "gross profit": ["gross margin dollars", "gross income"]
        }
        
        # Convert to the format expected by the key_terms_search_tool
        key_terms_str = ", ".join(sample_key_terms)
        synonyms_str = json.dumps(sample_synonyms)
        
        search_result = agent.key_terms_search_tool.invoke({
            "key_terms": key_terms_str,
            "synonyms": synonyms_str,
            "context": test_context
        })
        print(f"✅ Key Terms Search Result:")
        print(search_result)
    
    # Test 3: Test both tools together
    print("\n🔧 Test 3: Complete Pipeline (Formula Analysis + Key Terms Search)")
    print("-" * 30)
    
    # First, get the formula analysis
    formula_analysis = agent.formula_analysis_tool.invoke({"question": test_question})
    print(f"✅ Formula Analysis completed")
    
    # Then, if we have context, search for key terms
    if test_context:
        # Extract key terms from the formula analysis result
        # This is a simplified approach - in practice, you'd parse the result more carefully
        key_terms = ["revenue", "cost of goods sold"]  # Simplified for testing
        synonyms = {
            "revenue": ["total revenue", "net sales", "sales"],
            "cost of goods sold": ["cogs", "merchandise costs"]
        }
        
        # Convert to the format expected by the key_terms_search_tool
        key_terms_str = ", ".join(key_terms)
        synonyms_str = json.dumps(synonyms)
        
        search_result = agent.key_terms_search_tool.invoke({
            "key_terms": key_terms_str,
            "synonyms": synonyms_str,
            "context": test_context
        })
        print(f"✅ Key Terms Search completed")
        print(f"✅ Complete pipeline executed successfully!")
    else:
        print(f"✅ Formula Analysis completed (no context for search)")

if __name__ == "__main__":
    test_formula_tools() 