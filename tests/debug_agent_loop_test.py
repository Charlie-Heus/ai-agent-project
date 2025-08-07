#!/usr/bin/env python3
"""
Debug script to test one question and see the full output.
"""

import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import FinanceQAAgent
from control import load_financeqa_dataset

def debug_single_question():
    """Debug a single question to see what's happening."""
    print("🐛 DEBUG TEST - Single Question")
    print("=" * 40)
    
    # Load dataset
    questions = load_financeqa_dataset()
    if not questions:
        return
    
    # Get first question
    question_data = questions[0]
    question_data['question_num'] = 1
    
    print(f"Question: {question_data['question']}")
    print(f"Expected Answer: {question_data['answer']}")
    
    # Create the proper question_data structure
    processed_question_data = {
        'question_num': 1,
        'question': question_data['question'],
        'context': question_data['context'],
        'expected_answer': question_data['answer'],
        'company': question_data.get('company', 'N/A'),
        'question_type': question_data.get('question_type', 'N/A')
    }
    
    # Initialize agent
    print("\n🤖 Initializing FinanceQAAgent...")
    agent = FinanceQAAgent()
    
    # Run the question
    print("\n🔄 Running question through control flow...")
    
    import io
    import contextlib
    
    # Capture the output
    output = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(output):
            from control import display_question_info
            display_question_info(processed_question_data)
        
        full_output = output.getvalue()
        print("\n📋 FULL OUTPUT:")
        print("=" * 40)
        print(full_output)
        print("=" * 40)
        
        # Try to extract result
        print("\n🔍 EXTRACTING RESULT:")
        lines = full_output.split('\n')
        
        # Look for different result patterns
        result_patterns = [
            "Final Result:",
            "📊 Final Result:",
            "🎯 Final Answer:",
            "Result:",
            "Calculator Result:",
            "🔢 Calculator Result:"
        ]
        
        for pattern in result_patterns:
            for line in lines:
                if pattern in line:
                    print(f"Found '{pattern}' in line: {line}")
        
        # Look for any numerical results
        import re
        for line in lines:
            if re.search(r'\d+', line) and any(keyword in line.lower() for keyword in ['result', 'answer', 'calculated']):
                print(f"Potential result line: {line}")
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    debug_single_question() 