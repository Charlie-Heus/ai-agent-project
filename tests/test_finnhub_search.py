"""
Finnhub Financial Data Tool Test Script
Tests the finnhub_search_tool function in isolation
"""

import os
import requests
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)

def finnhub_search_tool(query: str) -> str:
    """
    Copy of your Finnhub financial data tool function for isolated testing
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

def test_environment():
    """Test if Finnhub API key is properly configured"""
    print("🔍 Testing Finnhub API Environment...")
    
    finnhub_api_key = os.getenv('FINNHUB_API_KEY')
    
    if not finnhub_api_key:
        print("❌ FINNHUB_API_KEY not found in environment")
        print("💡 Note: Finnhub API is optional, but needed for this tool to work")
        print("💡 Get a free API key at: https://finnhub.io/register")
        return False
    else:
        print(f"✅ FINNHUB_API_KEY found (length: {len(finnhub_api_key)} chars)")
        
        # Basic format check (Finnhub API keys are typically long strings)
        if len(finnhub_api_key) < 10:
            print("⚠️  Warning: Finnhub API key seems unusually short")
        
        return True

def test_basic_functionality():
    """Test basic Finnhub financial data functionality"""
    print("\n" + "="*60)
    print("TESTING BASIC FINNHUB FINANCIAL DATA FUNCTIONALITY")
    print("="*60)
    
    test_cases = [
        {
            "name": "Apple Inc (AAPL)",
            "query": "AAPL",
            "expected": "Should find Apple financial data"
        },
        {
            "name": "Microsoft (MSFT)",
            "query": "MSFT", 
            "expected": "Should find Microsoft financial data"
        },
        {
            "name": "Tesla (TSLA)",
            "query": "TSLA",
            "expected": "Should find Tesla financial data"
        },
        {
            "name": "Lowercase ticker",
            "query": "aapl",
            "expected": "Should work with lowercase (converted to uppercase)"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print(f"Query: '{test_case['query']}'")
        print(f"Expected: {test_case['expected']}")
        print("-" * 40)
        
        try:
            result = finnhub_search_tool(test_case['query'])
            
            # Print the actual data returned
            print(f"📊 DATA RETURNED:")
            print(f"   {result}")
            print()
            
            # Analyze the result
            if "Finnhub API key not configured" in result:
                print(f"❌ API KEY ISSUE: {result}")
            elif "Finnhub search error" in result:
                print(f"❌ SEARCH ERROR: {result}")
            elif "Finnhub profile failed" in result or "Finnhub metrics failed" in result:
                print(f"❌ API FAILURE: {result}")
                # Show more details about the failure
                if "Status:" in result:
                    status_code = result.split("Status: ")[1]
                    print(f"   HTTP Status Code: {status_code}")
                    if status_code == "401":
                        print("   💡 This suggests authentication issues (invalid API key)")
                    elif status_code == "403": 
                        print("   💡 This suggests authorization issues (API key lacks permissions)")
                    elif status_code == "429":
                        print("   💡 This suggests rate limiting (too many requests)")
                    elif status_code == "500":
                        print("   💡 This suggests server issues on Finnhub API side")
            elif "Finnhub Financial Data" in result:
                print(f"✅ SUCCESS: {result}")
                # Additional validation for Finnhub data
                if "'company':" in result and "'quote':" in result:
                    print("   ✅ Contains company profile and quote data")
                if "'currentPrice':" in result:
                    print("   ✅ Contains current stock price")
                if "'marketCapitalization':" in result:
                    print("   ✅ Contains market capitalization data")
            else:
                print(f"⚠️  UNEXPECTED RESPONSE: {result}")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
        
        if i < len(test_cases):
            input("\nPress Enter to continue to next test...")

def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60)
    
    error_test_cases = [
        {
            "name": "Invalid Ticker",
            "query": "NOTAREALTICKER123", 
            "expected": "Should handle invalid ticker gracefully"
        },
        {
            "name": "Empty Query",
            "query": "",
            "expected": "Should handle empty query gracefully"
        },
        {
            "name": "Very Long Query",
            "query": "A" * 100,
            "expected": "Should handle overly long query gracefully"
        },
        {
            "name": "Special Characters",
            "query": "@@##$$",
            "expected": "Should handle special characters gracefully"
        }
    ]
    
    for i, test_case in enumerate(error_test_cases, 1):
        print(f"\n🧪 Error Test {i}: {test_case['name']}")
        print(f"Query: '{test_case['query']}'")
        print(f"Expected: {test_case['expected']}")
        print("-" * 40)
        
        try:
            result = finnhub_search_tool(test_case['query'])
            
            # For error cases, we expect either graceful handling or appropriate error messages
            if "Finnhub API key not configured" in result:
                print(f"❌ API KEY MISSING: {result}")
            elif "Finnhub search error" in result or "Finnhub profile failed" in result or "Finnhub metrics failed" in result:
                print(f"✅ ERROR HANDLED GRACEFULLY: {result}")
            elif "Finnhub Financial Data" in result:
                print(f"✅ SUCCESS: {result}")
            else:
                print(f"⚠️  UNKNOWN RESPONSE: {result}")
                
        except Exception as e:
            print(f"❌ UNHANDLED EXCEPTION: {str(e)}")
        
        if i < len(error_test_cases):
            input("\nPress Enter to continue to next test...")

def test_api_connectivity():
    """Test raw Finnhub API connectivity without using the tool"""
    print("\n" + "="*60)
    print("TESTING RAW FINNHUB API CONNECTIVITY")
    print("="*60)
    
    finnhub_api_key = os.getenv('FINNHUB_API_KEY')
    
    if not finnhub_api_key:
        print("❌ Cannot test API connectivity - no Finnhub API key configured")
        return
    
    print("Testing direct API connection...")
    
    try:
        # Test with a known good ticker
        url = "https://finnhub.io/api/v1/stock/profile2"
        params = {'symbol': 'AAPL', 'token': finnhub_api_key}
        
        print(f"🔗 Making request to: {url}")
        print(f"📋 Query parameters: symbol=AAPL")
        print(f"🔑 Using API key: {finnhub_api_key[:10]}...{finnhub_api_key[-5:]}")
        
        response = requests.get(url, params=params, timeout=10)
        
        print(f"📊 Response status code: {response.status_code}")
        print(f"📊 Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ API CONNECTIVITY SUCCESS")
            
            # Try to parse the response
            try:
                data = response.json()
                print(f"📋 Response data type: {type(data)}")
                if isinstance(data, dict):
                    print(f"📋 Response keys: {list(data.keys())}")
                    if 'name' in data:
                        print(f"📋 Company name: {data['name']}")
                    if 'ticker' in data:
                        print(f"📋 Ticker: {data['ticker']}")
                    if 'marketCapitalization' in data:
                        print(f"📋 Market Cap: {data['marketCapitalization']}")
                elif isinstance(data, list):
                    print(f"📋 Response list length: {len(data)}")
                    
                # Show a small sample of the response
                response_text = response.text[:500]
                print(f"📋 Response preview: {response_text}...")
                
            except Exception as e:
                print(f"⚠️  Could not parse JSON response: {e}")
                print(f"📋 Raw response: {response.text[:200]}...")
                
        else:
            print(f"❌ API CONNECTIVITY FAILED")
            print(f"📋 Response text: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ API REQUEST TIMEOUT")
    except requests.exceptions.ConnectionError:
        print("❌ API CONNECTION ERROR") 
    except Exception as e:
        print(f"❌ API REQUEST EXCEPTION: {str(e)}")

def test_custom_query():
    """Test with a custom ticker provided by user"""
    print("\n" + "="*60)
    print("CUSTOM TICKER TEST")
    print("="*60)
    
    print("Enter a stock ticker symbol to test Finnhub search:")
    custom_ticker = input("Ticker Symbol: ").strip().upper()
    
    if not custom_ticker:
        print("No ticker provided, skipping custom test.")
        return
        
    print(f"\n🔍 Testing Finnhub search for: {custom_ticker}")
    print("-" * 40)
    
    try:
        result = finnhub_search_tool(custom_ticker)
        
        print(f"📋 Result: {result}")
        
        # Additional analysis
        if "Finnhub Financial Data" in result:
            print("🎯 SUCCESS: Finnhub data found for this ticker")
        elif "Finnhub profile failed" in result or "Finnhub metrics failed" in result:
            print("❌ FAILED: No Finnhub data found - may be invalid ticker or API issue")
        elif "Finnhub API key not configured" in result:
            print("🔑 API KEY MISSING: Cannot test without Finnhub API key")
        else:
            print("❓ UNCLEAR: Unexpected response format")
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")

if __name__ == "__main__":
    print("📊 Finnhub Financial Data Tool Test Suite")
    print("=" * 50)
    
    # Step 1: Check environment
    if not test_environment():
        print("\n❌ Finnhub API key not configured.")
        print("\n💡 To get a Finnhub API key:")
        print("1. Visit https://finnhub.io/register")
        print("2. Sign up for a free account")
        print("3. Get your API key from the dashboard")
        print("4. Add FINNHUB_API_KEY=your_key_here to your .env file")
        print("\n⚠️  Note: Finnhub API is optional for your main agent, but required for this tool to work.")
        
        # Ask if user wants to continue anyway
        choice = input("\nContinue testing anyway? (y/n): ").lower().strip()
        if choice != 'y':
            exit(1)
    else:
        print("\n✅ Finnhub API key found!")
    
    print("\nWhat would you like to test?")
    print("1. Basic functionality")
    print("2. Error handling")
    print("3. API connectivity")
    print("4. Custom ticker test")
    print("5. Run all tests")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        test_basic_functionality()
    elif choice == "2":
        test_error_handling()
    elif choice == "3":
        test_api_connectivity()
    elif choice == "4":
        test_custom_query()
    elif choice == "5":
        print("Running all tests...")
        test_basic_functionality()
        test_error_handling() 
        test_api_connectivity()
        test_custom_query()
    else:
        print("Invalid choice. Running basic functionality test...")
        test_basic_functionality()
    
    print("\n✅ Finnhub financial data tool testing complete!")