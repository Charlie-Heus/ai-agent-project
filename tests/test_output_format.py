#!/usr/bin/env python3
"""
Test script to verify the new output format
"""

import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_output_format():
    """Test the new output format."""
    print("Testing new output format...")
    print()
    
    # Test the helper function
    from control import display_final_result
    
    # Test case 1: Result with equals sign
    print("Test 1: Result with equals sign")
    display_final_result("result = 32095", {"expected_answer": "32095 million"})
    print()
    
    # Test case 2: Result without equals sign
    print("Test 2: Result without equals sign")
    display_final_result("32095", {"expected_answer": "32095 million"})
    print()
    
    # Test case 3: Missing expected answer
    print("Test 3: Missing expected answer")
    display_final_result("result = 32095", {})
    print()
    
    print("✅ All tests completed!")

if __name__ == "__main__":
    test_output_format() 