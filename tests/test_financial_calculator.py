#!/usr/bin/env python3
"""
Financial Calculator Tool Test Script
Tests the financial_calculator_tool function in isolation
"""

import os
import re
import numexpr
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)

def financial_calculator_tool(expression: str) -> str:
    """
    Copy of your financial_calculator_tool function for isolated testing
    """
    try:
        safe_expr = "".join(re.findall(r'[0-9\.\+\-\*\/\(\)\s]', expression))
        if not safe_expr:
            return "Invalid expression. No calculable content found."
        result = numexpr.evaluate(safe_expr).item()
        return f"Calculation result: {safe_expr.strip()} = {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}. Please provide a valid mathematical expression."

def test_environment():
    """Test if the calculator tool is properly configured"""
    print("🔍 Testing Financial Calculator Environment...")
    
    # The calculator tool doesn't require any API keys
    print("✅ Calculator tool is self-contained - no external dependencies required")
    return True

def test_basic_functionality():
    """Test basic calculator functionality"""
    print("\n" + "="*60)
    print("TESTING BASIC CALCULATOR FUNCTIONALITY")
    print("="*60)
    
    test_cases = [
        {
            "name": "Basic Addition",
            "expression": "10 + 20",
            "expected": "Should calculate 30"
        },
        {
            "name": "Basic Multiplication",
            "expression": "5 * 6",
            "expected": "Should calculate 30"
        },
        {
            "name": "Division",
            "expression": "100 / 4",
            "expected": "Should calculate 25"
        },
        {
            "name": "Complex Expression",
            "expression": "10 + 5 * 2",
            "expected": "Should follow order of operations"
        },
        {
            "name": "Decimal Math",
            "expression": "3.5 + 2.5",
            "expected": "Should handle decimals"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print(f"Expression: '{test_case['expression']}'")
        print(f"Expected: {test_case['expected']}")
        print("-" * 40)
        
        try:
            result = financial_calculator_tool(test_case['expression'])
            
            # Print the actual data returned
            print(f"📊 DATA RETURNED:")
            print(f"   {result}")
            print()
            
            # Analyze the result
            if "Calculation error" in result:
                print(f"❌ CALCULATION ERROR: {result}")
            elif "Invalid expression" in result:
                print(f"❌ INVALID EXPRESSION: {result}")
            elif "Calculation result:" in result:
                print(f"✅ SUCCESS: {result}")
                # Extract the calculated result
                if "=" in result:
                    calculated_result = result.split("=")[-1].strip()
                    print(f"   ✅ Calculated result: {calculated_result}")
            else:
                print(f"⚠️  UNEXPECTED RESPONSE: {result}")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
        
        if i < len(test_cases):
            input("\nPress Enter to continue to next test...")

def test_financial_calculations():
    """Test financial-specific calculations"""
    print("\n" + "="*60)
    print("TESTING FINANCIAL CALCULATIONS")
    print("="*60)
    
    financial_test_cases = [
        {
            "name": "Percentage Calculation",
            "expression": "150 * 0.15",
            "expected": "Should calculate 15% of 150"
        },
        {
            "name": "Growth Rate",
            "expression": "(120 - 100) / 100",
            "expected": "Should calculate 20% growth rate"
        },
        {
            "name": "P/E Ratio Calculation",
            "expression": "100 / 5",
            "expected": "Should calculate P/E ratio of 20"
        },
        {
            "name": "P/E Ratio with Decimals",
            "expression": "50.25 / 2.5",
            "expected": "Should calculate P/E ratio with decimals"
        },
        {
            "name": "Complex Financial Expression",
            "expression": "(1000 * 0.08) + (500 * 0.12)",
            "expected": "Should calculate weighted average"
        }
    ]
    
    for i, test_case in enumerate(financial_test_cases, 1):
        print(f"\n🧪 Financial Test {i}: {test_case['name']}")
        print(f"Expression: '{test_case['expression']}'")
        print(f"Expected: {test_case['expected']}")
        print("-" * 40)
        
        try:
            result = financial_calculator_tool(test_case['expression'])
            
            # Print the actual data returned
            print(f"📊 DATA RETURNED:")
            print(f"   {result}")
            print()
            
            # Analyze the result
            if "Calculation result:" in result:
                print(f"✅ SUCCESS: {result}")
                # Extract the calculated result
                if "=" in result:
                    calculated_result = result.split("=")[-1].strip()
                    print(f"   ✅ Calculated result: {calculated_result}")
            elif "Invalid expression" in result:
                print(f"✅ HANDLED GRACEFULLY: {result}")
            elif "Calculation error" in result:
                print(f"❌ CALCULATION ERROR: {result}")
            else:
                print(f"⚠️  UNEXPECTED RESPONSE: {result}")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
        
        if i < len(financial_test_cases):
            input("\nPress Enter to continue to next test...")

def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60)
    
    error_test_cases = [
        {
            "name": "Invalid Expression",
            "expression": "10 + + 5",
            "expected": "Should handle invalid syntax gracefully"
        },
        {
            "name": "Empty Expression",
            "expression": "",
            "expected": "Should handle empty input gracefully"
        },
        {
            "name": "Non-numeric Characters",
            "expression": "abc + def",
            "expected": "Should handle non-numeric input gracefully"
        },
        {
            "name": "Division by Zero",
            "expression": "10 / 0",
            "expected": "Should handle division by zero gracefully"
        },
        {
            "name": "Invalid Syntax",
            "expression": "1.25(10)",
            "expected": "Should handle invalid syntax gracefully"
        }
    ]
    
    for i, test_case in enumerate(error_test_cases, 1):
        print(f"\n🧪 Error Test {i}: {test_case['name']}")
        print(f"Expression: '{test_case['expression']}'")
        print(f"Expected: {test_case['expected']}")
        print("-" * 40)
        
        try:
            result = financial_calculator_tool(test_case['expression'])
            
            # Print the actual data returned
            print(f"📊 DATA RETURNED:")
            print(f"   {result}")
            print()
            
            # For error cases, we expect either graceful handling or appropriate error messages
            if "Calculation error" in result:
                print(f"✅ ERROR HANDLED GRACEFULLY: {result}")
            elif "Invalid expression" in result:
                print(f"✅ INVALID EXPRESSION HANDLED: {result}")
            elif "Calculation result:" in result:
                print(f"⚠️  UNEXPECTED SUCCESS: {result}")
            else:
                print(f"⚠️  UNKNOWN RESPONSE: {result}")
                
        except Exception as e:
            print(f"❌ UNHANDLED EXCEPTION: {str(e)}")
        
        if i < len(error_test_cases):
            input("\nPress Enter to continue to next test...")

def test_custom_calculation():
    """Test with a custom calculation provided by user"""
    print("\n" + "="*60)
    print("CUSTOM CALCULATION TEST")
    print("="*60)
    
    print("Enter a mathematical expression to test:")
    custom_expression = input("Expression: ").strip()
    
    if not custom_expression:
        print("No expression provided, skipping custom test.")
        return
        
    print(f"\n🔍 Testing calculation: {custom_expression}")
    print("-" * 40)
    
    try:
        result = financial_calculator_tool(custom_expression)
        
        print(f"📋 Result: {result}")
        
        # Additional analysis
        if "Calculation result:" in result:
            print("🎯 SUCCESS: Calculation completed successfully")
        elif "Calculation error" in result:
            print("❌ FAILED: Calculation error occurred")
        elif "Invalid expression" in result:
            print("❌ FAILED: Invalid expression provided")
        else:
            print("❓ UNCLEAR: Unexpected response format")
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")

if __name__ == "__main__":
    print("🧮 Financial Calculator Tool Test Suite")
    print("=" * 50)
    
    # Step 1: Check environment
    if not test_environment():
        print("\n❌ Environment check failed.")
        exit(1)
    else:
        print("\n✅ Environment check passed!")
    
    print("\nWhat would you like to test?")
    print("1. Basic functionality")
    print("2. Financial calculations")
    print("3. Error handling")
    print("4. Custom calculation test")
    print("5. Run all tests")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        test_basic_functionality()
    elif choice == "2":
        test_financial_calculations()
    elif choice == "3":
        test_error_handling()
    elif choice == "4":
        test_custom_calculation()
    elif choice == "5":
        print("Running all tests...")
        test_basic_functionality()
        test_financial_calculations()
        test_error_handling()
        test_custom_calculation()
    else:
        print("Invalid choice. Running basic functionality test...")
        test_basic_functionality()
    
    print("\n✅ Financial calculator tool testing complete!") 