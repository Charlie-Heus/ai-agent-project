def get_formula_from_question(question: str, llm) -> dict:
    """Tool 1: Get the mathematical formula required to answer a financial question."""
    print(f"\n🔍 Step 1: Getting formula for question...")
    
    formula_prompt = f"""
    You are a financial analysis expert. Given this question:
    
    QUESTION: {question}
    
    Please provide ONLY the mathematical formula required to calculate the answer.
    Respond with just the formula, nothing else.
    
    Examples:
    - For "What is the gross profit margin?" → "Gross Profit Margin = (Revenue - Cost of Goods Sold) / Revenue"
    - For "What is EBITDA?" → "EBITDA = Net Income + Interest + Taxes + Depreciation + Amortization"
    - For "What is the current ratio?" → "Current Ratio = Current Assets / Current Liabilities"
    """
    
    formula_response = llm.invoke(formula_prompt)
    formula = formula_response.content.strip()
    
    print(f"📋 Formula: {formula}")
    
    return {
        "formula": formula,
        "question": question,
        "llm_response": formula_response.content
    }

def extract_key_terms_and_synonyms(formula: str, llm) -> dict:
    """Tool 2: Extract key terms and synonyms from a financial formula."""
    print(f"\n🔍 Step 2: Extracting key terms and synonyms from formula...")
    
    analysis_prompt = f"""
    You are a financial analysis expert. Given this formula:
    
    FORMULA: {formula}
    
    Please provide:
    1. The KEY TERMS needed from the formula (e.g., "revenue", "cost of goods sold")
    2. SYNONYMS for each key term (e.g., "revenue" → ["total revenue", "net sales", "sales", "gross sales", "total income"])
    
    Respond in this exact format:
    KEY_TERMS: [term1, term2, term3, ...]
    SYNONYMS: {{"term1": ["synonym1", "synonym2"], "term2": ["synonym1", "synonym2"], ...}}
    """
    
    analysis_response = llm.invoke(analysis_prompt)
    print(f"📋 Analysis Response:\n{analysis_response.content}\n")
    
    # Parse the response to extract key terms and synonyms
    key_terms = []
    synonyms = {}
    
    lines = analysis_response.content.split('\n')
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith('KEY_TERMS:'):
            terms_str = line_stripped.replace('KEY_TERMS:', '').strip()
            key_terms = [term.strip() for term in terms_str.strip('[]').split(',')]
        elif line_stripped.startswith('SYNONYMS:'):
            # Try to parse the LLM's synonym response
            synonyms_str = line_stripped.replace('SYNONYMS:', '').strip()
            
            # If this line doesn't contain the full JSON, look for the complete JSON structure
            if not synonyms_str.startswith('{'):
                # Find the complete JSON structure in the response
                synonyms_start = analysis_response.content.find('SYNONYMS:')
                if synonyms_start != -1:
                    synonyms_section = analysis_response.content[synonyms_start:]
                    # Find the opening brace
                    brace_start = synonyms_section.find('{')
                    if brace_start != -1:
                        # Find the matching closing brace
                        brace_count = 0
                        brace_end = -1
                        for i, char in enumerate(synonyms_section[brace_start:], brace_start):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    brace_end = i + 1
                                    break
                        
                        if brace_end != -1:
                            synonyms_str = synonyms_section[brace_start:brace_end]
            
            try:
                # Try to parse as JSON
                import json
                synonyms = json.loads(synonyms_str)
            except:
                # If parsing fails, create synonyms based on key terms
                synonyms = {}
                for term in key_terms:
                    term_lower = term.lower()
                    if "revenue" in term_lower:
                        synonyms[term] = ["total revenue", "net sales", "sales", "revenue", "income", "net revenue"]
                    elif "cost" in term_lower and "goods" in term_lower:
                        synonyms[term] = ["cost of goods sold", "cogs", "merchandise costs", "cost of sales", "cost of merchandise sold", "merchandise costs"]
                    elif "gross" in term_lower and "profit" in term_lower:
                        synonyms[term] = ["gross profit", "gross margin dollars", "gross income", "gross profit margin"]
                    elif "operating" in term_lower and "income" in term_lower:
                        synonyms[term] = ["operating income", "operating profit", "ebit", "operating earnings"]
                    elif "net" in term_lower and "income" in term_lower:
                        synonyms[term] = ["net income", "net profit", "earnings", "net earnings", "net income attributable to costco"]
                    elif "ebitda" in term_lower:
                        synonyms[term] = ["ebitda", "earnings before interest taxes depreciation amortization"]
                    elif "depreciation" in term_lower:
                        synonyms[term] = ["depreciation", "depreciation and amortization", "d&a", "depreciation and amortization expense"]
                    elif "merchandise" in term_lower and "costs" in term_lower:
                        synonyms[term] = ["merchandise costs", "cost of goods sold", "cogs", "cost of sales", "merchandise expenses"]
                    elif "sales" in term_lower and "net" in term_lower:
                        synonyms[term] = ["net sales", "total revenue", "revenue", "sales", "net revenue"]
                    else:
                        # Default: just use the term itself
                        synonyms[term] = [term.lower()]
    
    print(f"🎯 Key Terms: {key_terms}")
    print(f"🎯 Synonyms: {synonyms}")
    
    return {
        "formula": formula,
        "key_terms": key_terms,
        "synonyms": synonyms,
        "llm_response": analysis_response.content
    }

def search_key_terms_in_context(key_terms: list, synonyms: dict, context: str) -> dict:
    """Search for key terms in context and return the most relevant sentences."""
    print(f"\n🔍 Searching for key terms in context...")
    
    # Create a comprehensive search query from key terms and synonyms
    all_search_terms = []
    for term in key_terms:
        # Clean up the term - remove quotes if present
        clean_term = term.strip().strip('"').strip("'")
        if clean_term in synonyms:
            all_search_terms.extend(synonyms[clean_term])
        else:
            all_search_terms.append(clean_term)
    
    print(f"🔍 Searching for terms: {all_search_terms}")
    
    # Debug: Show a sample of the context to see what we're searching in
    print(f"🔍 Context sample (first 500 chars): {context[:500]}...")
    
    # Simple direct search for terms in context
    extracted_info = {}
    context_lower = context.lower()
    
    for term in all_search_terms:
        term_lower = term.lower()
        print(f"🔍 Looking for: '{term}' (lowercase: '{term_lower}')")
        
        if term_lower in context_lower:
            # Find the position of the term
            pos = context_lower.find(term_lower)
            
            # Extract surrounding text (approximately 50 characters before and after)
            start = max(0, pos - 50)
            end = min(len(context), pos + len(term) + 50)
            
            # Try to find sentence boundaries
            start = context.rfind('.', start, pos) + 1 if context.rfind('.', start, pos) > start - 100 else start
            end = context.find('.', pos) + 1 if context.find('.', pos) < end + 100 else end
            
            relevant_text = context[start:end].strip()
            if relevant_text:
                extracted_info[term] = relevant_text
                print(f"✅ Found '{term}': {relevant_text[:100]}...")
        else:
            print(f"❌ Not found: '{term}'")
            # Try partial matches for multi-word terms
            if ' ' in term_lower:
                words = term_lower.split()
                for word in words:
                    if len(word) > 3 and word in context_lower:  # Only search for words longer than 3 chars
                        print(f"🔍 Found partial match: '{word}' in '{term}'")
                        # Find the position of the partial match
                        pos = context_lower.find(word)
                        
                        # Extract surrounding text (approximately 50 characters before and after)
                        start = max(0, pos - 50)
                        end = min(len(context), pos + len(word) + 50)
                        
                        # Try to find sentence boundaries
                        start = context.rfind('.', start, pos) + 1 if context.rfind('.', start, pos) > start - 100 else start
                        end = context.find('.', pos) + 1 if context.find('.', pos) < end + 100 else end
                        
                        relevant_text = context[start:end].strip()
                        if relevant_text:
                            extracted_info[f"{term} (partial: {word})"] = relevant_text
                            print(f"✅ Found partial '{word}' from '{term}': {relevant_text[:100]}...")
                            break
    
    if extracted_info:
        print(f"\n📊 Extracted Information:")
        for term, text in extracted_info.items():
            print(f"  • {term}: {text}")
            print()  # Add newline between each match
        
        # Convert dictionary to a single string
        extracted_text = "\n\n".join([f"{term}: {text}" for term, text in extracted_info.items()])
    else:
        print(f"\n❌ No relevant information found for any of the search terms")
        extracted_text = ""
    
    return {
        "extracted_text": extracted_text,
        "extracted_info": extracted_info,
        "search_terms": all_search_terms,
        "context_length": len(context)
    }

# Example usage function
def run_formula_analysis_pipeline(question: str, llm, context: str = "") -> dict:
    """Run the complete pipeline: get formula, then extract key terms and synonyms, then search in context."""
    print(f"\n🚀 Starting Formula Analysis Pipeline")
    print(f"Question: {question}")
    
    # Step 1: Get formula
    formula_result = get_formula_from_question(question, llm)
    
    # Step 2: Extract key terms and synonyms from formula
    analysis_result = extract_key_terms_and_synonyms(formula_result["formula"], llm)
    
    # Step 3: Search for key terms in context (if context is provided)
    search_result = None
    if context:
        print(f"\n🔍 Step 3: Searching for key terms in context...")
        search_result = search_key_terms_in_context(
            analysis_result['key_terms'], 
            analysis_result['synonyms'], 
            context
        )
    
    # Combine results
    final_result = {
        "question": question,
        "formula": formula_result["formula"],
        "key_terms": analysis_result["key_terms"],
        "synonyms": analysis_result["synonyms"],
        "step1_response": formula_result["llm_response"],
        "step2_response": analysis_result["llm_response"],
        "search_result": search_result
    }
    
    print(f"\n✅ Pipeline Complete!")
    print(f"📊 Final Results:")
    print(f"   Formula: {final_result['formula']}")
    print(f"   Key Terms: {final_result['key_terms']}")
    print(f"   Synonyms: {final_result['synonyms']}")
    if search_result:
        print(f"   Search Results: Found {len(search_result['extracted_info'])} relevant terms")
    
    return final_result 