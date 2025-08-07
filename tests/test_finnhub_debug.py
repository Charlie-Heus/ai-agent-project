#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from main import FinanceQAAgent

# Load environment variables
load_dotenv()

def test_finnhub_tool():
    """Test the Finnhub tool directly to see what's happening."""
    print("🔍 Testing Finnhub Tool")
    print("=" * 50)
    
    # Check if API key is set
    finnhub_api_key = os.getenv('FINNHUB_API_KEY')
    print(f"FINNHUB_API_KEY set: {'Yes' if finnhub_api_key else 'No'}")
    if finnhub_api_key:
        print(f"API Key length: {len(finnhub_api_key)}")
        print(f"API Key starts with: {finnhub_api_key[:10]}...")
    
    # Create agent
    print("\n🏗️  Creating FinanceQAAgent...")
    agent = FinanceQAAgent()
    
    # Test Finnhub tool
    print("\n🔍 Testing Finnhub search for AAPL...")
    try:
        result = agent.finnhub_search_tool.invoke({"query": "AAPL"})
        print(f"✅ Tool call completed")
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(str(result))}")
        print(f"Raw result: {repr(result)}")
        
        if result and str(result).strip():
            print("\n📊 Finnhub Results:")
            print("=" * 50)
            print(result)
            print("=" * 50)
        else:
            print("\n❌ Empty result returned")
            
    except Exception as e:
        print(f"❌ Error calling Finnhub tool: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_finnhub_tool() 