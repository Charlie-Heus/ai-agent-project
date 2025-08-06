"""
Webpage Content Tool Test Script
Tests the fetch_webpage_content_tool function in isolation
"""

import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)

def _fetch_full_page_content_static(url: str, max_chars: int = 1500) -> str:
    """Static method to fetch full content from a webpage - copied from your main.py"""
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
        
        return text[:max_chars] + "..." if len(text) > max_chars else text
        
    except Exception as e:
        return f"Could not fetch content from {url}: {str(e)}"

def fetch_webpage_content_tool(url: str) -> str:
    """
    Copy of your fetch_webpage_content_tool for isolated testing
    """
    return _fetch_full_page_content_static(url)

def test_basic_functionality():
    """Test basic webpage fetching functionality"""
    print("\n" + "="*60)
    print("TESTING BASIC WEBPAGE CONTENT FETCHING")
    print("="*60)
    
    test_cases = [
        {
            "name": "Financial News Site",
            "url": "https://finance.yahoo.com/news/",
            "expected": "Should return Yahoo Finance news content"
        },
        {
            "name": "Simple HTML Page",
            "url": "https://httpbin.org/html",
            "expected": "Should return basic HTML content"
        },
        {
            "name": "SEC Website",
            "url": "https://www.sec.gov/",
            "expected": "Should return SEC homepage content"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print(f"URL: {test_case['url']}")
        print(f"Expected: {test_case['expected']}")
        print("-" * 40)
        
        try:
            result = fetch_webpage_content_tool(test_case['url'])
            
            # Basic validation
            if result.startswith("Could not fetch content"):
                print(f"❌ ERROR: {result}")
            elif len(result) < 50:
                print(f"⚠️  VERY SHORT CONTENT: {result}")
            elif len(result) >= 1500:
                print("✅ SUCCESS: Good length content retrieved")
                print(f"📊 Content length: {len(result)} characters")
                print(f"📋 Content preview: {result[:200]}...")
            else:
                print("✅ SUCCESS: Content retrieved")
                print(f"📊 Content length: {len(result)} characters")
                print(f"📋 Content preview: {result[:200]}...")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
        
        if i < len(test_cases):
            input("\nPress Enter to continue to next test...")

def test_error_handling():
    """Test error handling with invalid URLs"""
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60)
    
    error_test_cases = [
        {
            "name": "Invalid URL",
            "url": "https://this-domain-definitely-does-not-exist.com",
            "expected": "Should return error message about DNS/connection failure"
        },
        {
            "name": "Empty URL", 
            "url": "",
            "expected": "Should return error about invalid URL"
        },
        {
            "name": "Invalid Protocol",
            "url": "not-a-url",
            "expected": "Should return error about invalid URL format"
        },
        {
            "name": "404 Page",
            "url": "https://httpbin.org/status/404",
            "expected": "Should return error about 404 status"
        }
    ]
    
    for i, test_case in enumerate(error_test_cases, 1):
        print(f"\n🧪 Error Test {i}: {test_case['name']}")
        print(f"URL: '{test_case['url']}'")
        print(f"Expected: {test_case['expected']}")
        print("-" * 40)
        
        try:
            result = fetch_webpage_content_tool(test_case['url'])
            
            if "Could not fetch content" in result:
                print(f"✅ ERROR HANDLED CORRECTLY: {result}")
            else:
                print(f"⚠️  UNEXPECTED SUCCESS: {result[:100]}...")
                
        except Exception as e:
            print(f"❌ UNHANDLED EXCEPTION: {str(e)}")
        
        if i < len(error_test_cases):
            input("\nPress Enter to continue to next test...")

def test_content_limits():
    """Test content length limiting functionality"""
    print("\n" + "="*60)
    print("TESTING CONTENT LENGTH LIMITS")
    print("="*60)
    
    # Test with different max_chars limits
    test_url = "https://en.wikipedia.org/wiki/Financial_analysis"
    limits = [100, 500, 1500, 3000]
    
    for i, limit in enumerate(limits, 1):
        print(f"\n🧪 Limit Test {i}: Max {limit} characters")
        print(f"URL: {test_url}")
        print("-" * 40)
        
        try:
            result = _fetch_full_page_content_static(test_url, max_chars=limit)
            
            actual_length = len(result)
            print(f"📊 Requested limit: {limit} chars")
            print(f"📊 Actual length: {actual_length} chars")
            
            if actual_length <= limit + 3:  # +3 for "..." if truncated
                print("✅ SUCCESS: Content properly limited")
            else:
                print(f"❌ FAILED: Content exceeds limit by {actual_length - limit} chars")
                
            # Show preview
            print(f"📋 Content preview: {result[:100]}...")
            
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
        
        if i < len(limits):
            input("\nPress Enter to continue to next test...")

def test_financial_content_parsing():
    """Test with actual financial content URLs"""
    print("\n" + "="*60)
    print("TESTING FINANCIAL CONTENT PARSING")
    print("="*60)
    
    financial_urls = [
        {
            "name": "SEC Filing Example",
            "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20241228.htm",
            "keywords": ["Apple", "financial", "revenue", "assets"]
        },
        {
            "name": "Financial News Article",
            "url": "https://finance.yahoo.com/quote/AAPL/",
            "keywords": ["Apple", "stock", "price", "$"]
        },
        {
            "name": "Investopedia Article",
            "url": "https://www.investopedia.com/terms/f/financial-analysis.asp",
            "keywords": ["financial", "analysis", "investment", "evaluation"]
        }
    ]
    
    for i, test_case in enumerate(financial_urls, 1):
        print(f"\n🧪 Financial Test {i}: {test_case['name']}")
        print(f"URL: {test_case['url']}")
        print(f"Looking for keywords: {test_case['keywords']}")
        print("-" * 40)
        
        try:
            result = fetch_webpage_content_tool(test_case['url'])
            
            if result.startswith("Could not fetch content"):
                print(f"❌ ERROR: {result}")
                continue
                
            # Check for financial keywords
            result_lower = result.lower()
            found_keywords = [kw for kw in test_case['keywords'] if kw.lower() in result_lower]
            
            print(f"✅ SUCCESS: Content retrieved ({len(result)} chars)")
            print(f"📊 Keywords found: {found_keywords}")
            print(f"📋 Content preview: {result[:200]}...")
            
            if len(found_keywords) >= 2:
                print("🎯 GOOD: Multiple relevant keywords found")
            elif len(found_keywords) == 1:
                print("⚠️  OK: Some relevant keywords found") 
            else:
                print("❓ UNCLEAR: No obvious financial keywords found")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
        
        if i < len(financial_urls):
            input("\nPress Enter to continue to next test...")

def custom_url_test():
    """Test with a custom URL provided by user"""
    print("\n" + "="*60)
    print("CUSTOM URL TEST")
    print("="*60)
    
    print("Enter a URL to test the webpage content fetching:")
    custom_url = input("URL: ").strip()
    
    if not custom_url:
        print("No URL provided, skipping custom test.")
        return
        
    print(f"\n🔍 Testing custom URL: {custom_url}")
    print("-" * 40)
    
    try:
        result = fetch_webpage_content_tool(custom_url)
        
        print(f"📊 Result length: {len(result)} characters")
        print(f"📋 Full result:")
        print("-" * 40)
        print(result)
        
    except Exception as e:
        print(f"❌ Exception: {str(e)}")

if __name__ == "__main__":
    print("🔧 Webpage Content Tool Test Suite")
    print("=" * 50)
    
    print("\nWhat would you like to test?")
    print("1. Basic functionality")
    print("2. Error handling") 
    print("3. Content length limits")
    print("4. Financial content parsing")
    print("5. Custom URL test")
    print("6. Run all tests")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice == "1":
        test_basic_functionality()
    elif choice == "2":
        test_error_handling()
    elif choice == "3":
        test_content_limits()
    elif choice == "4":
        test_financial_content_parsing()
    elif choice == "5":
        custom_url_test()
    elif choice == "6":
        print("Running all tests...")
        test_basic_functionality()
        test_error_handling()
        test_content_limits()
        test_financial_content_parsing()
        custom_url_test()
    else:
        print("Invalid choice. Running basic functionality test...")
        test_basic_functionality()
    
    print("\n✅ Webpage content tool testing complete!")