#!/usr/bin/env python3
"""
Test script to check value extraction for Question 5
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import FinanceQAAgent

def test_value_extraction():
    """Test the value extraction for Question 5."""
    
    # Create agent
    agent = FinanceQAAgent()
    
    # Question 5 data
    question_data = {
        'question_num': 5,
        'question': 'What is adjusted EBIT for the year ending in 2024?',
        'company': 'Costco',
        'question_type': 'assumption',
        'expected_answer': '$9,396 (in millions)',
        'context': 'COSTCO WHOLESALE CORPORATION...'  # Simplified for test
    }
    
    # Formula
    formula = "Adjusted EBIT = EBIT + Adjustments"
    
    # Context chunks from key terms search
    context_chunks = """
Found relevant information:

Adjustments:  noncontrolling interests $ 7,367 $ 6,292 $ 5,915
Adjustments to reconcile net income including noncontrolling 

adjustments:  noncontrolling interests $ 7,367 $ 6,292 $ 5,915
Adjustments to reconcile net income including noncontrolling 

Revenue: ptember 1,
2024
September 3,
2023
August 28,
2022
REVENUE
Net sales $ 249,625 $ 237,710 $ 222,730
Membershi

revenue: ptember 1,
2024
September 3,
2023
August 28,
2022
REVENUE
Net sales $ 249,625 $ 237,710 $ 222,730
Membershi

Operating Expenses: 4,580 4,224
Total revenue 254,453 242,290 226,954
OPERATING EXPENSES
Merchandise costs 222,358 212,586 199,382
Selling

operating expenses: 4,580 4,224
Total revenue 254,453 242,290 226,954
OPERATING EXPENSES
Merchandise costs 222,358 212,586 199,382
Selling
"""
    
    # Import the function
    from control import extract_values_from_context
    
    # Test value extraction
    print("Testing value extraction for Question 5...")
    print(f"Formula: {formula}")
    print(f"Context chunks: {context_chunks[:200]}...")
    
    result = extract_values_from_context(question_data, formula, context_chunks, agent)
    
    print(f"\nExtraction result:")
    print(f"Values: {result['values']}")
    print(f"Calculation: {result['calculation']}")
    print(f"Result: {result['result']}")
    
    return result

if __name__ == "__main__":
    test_value_extraction() 