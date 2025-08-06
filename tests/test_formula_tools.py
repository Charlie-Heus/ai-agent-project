#!/usr/bin/env python3
"""
Test script for the formula analysis tools.
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formula_tools import get_formula_from_question, extract_key_terms_and_synonyms, run_formula_analysis_pipeline, search_key_terms_in_context

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
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    print("\n🧪 Testing Formula Analysis Tools")
    print("=" * 50)
    
    # Test 1: Get formula only
    print("\n🔧 Test 1: Get Formula Only")
    print("-" * 30)
    formula_result = get_formula_from_question(test_question, llm)
    print(f"✅ Formula: {formula_result['formula']}")
    
    # Test 2: Extract key terms and synonyms from formula
    print("\n🔧 Test 2: Extract Key Terms and Synonyms")
    print("-" * 30)
    analysis_result = extract_key_terms_and_synonyms(formula_result['formula'], llm)
    print(f"✅ Key Terms: {analysis_result['key_terms']}")
    print(f"✅ Synonyms: {analysis_result['synonyms']}")
    
    # Test 3: Search for key terms in context (if context is available)
    if test_context:
        print("\n🔧 Test 3: Search Key Terms in Context")
        print("-" * 30)
        search_result = search_key_terms_in_context(
            analysis_result['key_terms'], 
            analysis_result['synonyms'], 
            test_context
        )
        print(f"✅ Search completed. Found {len(search_result['extracted_info'])} relevant terms.")
        if search_result['extracted_text']:
            print(f"✅ Extracted text length: {len(search_result['extracted_text'])} characters")
        else:
            print("❌ No relevant information found in context")
    
    # Test 4: Run complete pipeline
    print("\n🔧 Test 4: Complete Pipeline")
    print("-" * 30)
    pipeline_result = run_formula_analysis_pipeline(test_question, llm, test_context)
    print(f"✅ Complete pipeline executed successfully!")

if __name__ == "__main__":
    test_formula_tools() 