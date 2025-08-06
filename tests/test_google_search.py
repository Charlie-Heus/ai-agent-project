"""
Google Search Tool Test Script
Tests the web_search_tool function in isolation
"""

import os
from dotenv import load_dotenv, find_dotenv
from googleapiclient.discovery import build
from google.api_core.exceptions import GoogleAPIError

# Load environment variables
load_dotenv(find_dotenv(), override=True)

def web_search_tool(query: str) -> str:
    """
    Copy of your web_search_tool function for isolated testing
    """

    # Add this check at the beginning of web_search_tool
    if not query or not query.strip():
        return "Please provide a search query. Empty queries are not supported."
    
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

def test_environment():
    """Test if environment variables are properly set"""
    print("🔍 Testing Environment Variables...")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("CUSTOM_SEARCH_ENGINE_ID")
    
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment")
        return False
    else:
        print(f"✅ GOOGLE_API_KEY found (length: {len(api_key)} chars)")
    
    if not cse_id:
        print("❌ CUSTOM_SEARCH_ENGINE_ID not found in environment")
        return False
    else:
        print(f"✅ CUSTOM_SEARCH_ENGINE_ID found: {cse_id}")
    
    return True

def run_test_cases():
    """Run various test cases for the Google Search Tool"""
    
    test_cases = [
        {
            "name": "Basic Financial Query",
            "query": "Tesla quarterly earnings 2024",
            "expected": "Should return financial news about Tesla"
        },
        {
            "name": "Stock Analysis Query", 
            "query": "Apple stock price analysis",
            "expected": "Should return Apple stock analysis content"
        },
        {
            "name": "Empty Query",
            "query": "",
            "expected": "Should handle empty query gracefully"
        },
        {
            "name": "Very Short Query",
            "query": "AAPL",
            "expected": "Should return results for Apple ticker"
        }
    ]
    
    print("\n" + "="*60)
    print("RUNNING GOOGLE SEARCH TOOL TESTS")
    print("="*60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print(f"Query: '{test_case['query']}'")
        print(f"Expected: {test_case['expected']}")
        print("-" * 40)
        
        try:
            result = web_search_tool(test_case['query'])
            
            # Basic validation
            if "error" in result.lower():
                print(f"❌ ERROR RESULT: {result}")
            elif "no web search results found" in result.lower():
                print(f"⚠️  NO RESULTS: {result}")
            elif "Title:" in result and "Snippet:" in result:
                print("✅ SUCCESS: Properly formatted results returned")
                
                # Count results
                result_count = result.count("Title:")
                print(f"📊 Found {result_count} search results")
                
                # Show first result preview
                first_result = result.split("---")[0]
                lines = first_result.split("\n")
                for line in lines[:3]:  # Show first 3 lines
                    if line.strip():
                        print(f"   {line}")
                if len(lines) > 3:
                    print("   ...")
            else:
                print(f"⚠️  UNEXPECTED FORMAT: {result[:200]}...")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
        
        if i < len(test_cases):
            input("\nPress Enter to continue to next test...")

def detailed_result_inspection():
    """Inspect a single search result in detail"""
    print("\n" + "="*60)
    print("DETAILED RESULT INSPECTION")
    print("="*60)
    
    query = "Microsoft financial results Q4 2024"
    print(f"Testing detailed search for: '{query}'")
    
    result = web_search_tool(query)
    
    print(f"\n📋 Raw Result:")
    print("-" * 40)
    print(result)
    
    print(f"\n📊 Result Analysis:")
    print("-" * 40)
    print(f"Character count: {len(result)}")
    print(f"Number of results: {result.count('Title:')}")
    print(f"Contains sources: {'Source:' in result}")
    print(f"Contains snippets: {'Snippet:' in result}")

if __name__ == "__main__":
    print("🏦 Google Search Tool Test Suite")
    print("=" * 50)
    
    # Step 1: Check environment
    if not test_environment():
        print("\n❌ Environment check failed. Please check your .env file.")
        exit(1)
    
    print("\n✅ Environment check passed!")
    
    # Ask user what they want to test
    print("\nWhat would you like to test?")
    print("1. Run all test cases")
    print("2. Run detailed result inspection")
    print("3. Custom query test")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        run_test_cases()
    elif choice == "2":
        detailed_result_inspection()
    elif choice == "3":
        custom_query = input("Enter your custom query: ").strip()
        print(f"\n🔍 Testing custom query: '{custom_query}'")
        result = web_search_tool(custom_query)
        print(f"\n📋 Result:\n{result}")
    else:
        print("Invalid choice. Running all test cases...")
        run_test_cases()
    
    print("\n✅ Google Search Tool testing complete!")