#!/usr/bin/env python3
"""
Test Control Flow Script
========================

This script tests the exact control flow path we've been testing:
1. Generate formula and keywords
2. Keyword search
3. Assess information completeness
4. If not enough info, enter max 5 iteration loop
5. Choose one of 5 tools in each iteration

This follows the same path as the main control.py but in a more controlled test environment.
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from main import FinanceQAAgent

# Load environment variables
load_dotenv()

def display_final_result(calc_result_str, question_data):
    """Display the final result with expected answer."""
    if "=" in calc_result_str:
        result_part = calc_result_str.split("=")[-1].strip()
        print(f"\n📊 Final Result: {result_part}")
        print(f"🎯 Expected Answer: {question_data.get('expected_answer', 'N/A')}")
    else:
        print(f"\n📊 Final Result: {calc_result_str}")
        print(f"🎯 Expected Answer: {question_data.get('expected_answer', 'N/A')}")

def load_financeqa_dataset():
    """Load the FinanceQA dataset from JSONL file."""
    try:
        # Use relative path from current working directory (root of project)
        data_path = Path("data/financeqa_test.jsonl")
        if not data_path.exists():
            print("❌ FinanceQA dataset not found. Please run download_financeqa.py first.")
            return None
        
        # Load all questions
        questions = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        
        print(f"✅ Loaded {len(questions)} questions from FinanceQA dataset")
        return questions
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def test_control_flow():
    """Test the complete control flow path."""
    print("\n🧪 TESTING CONTROL FLOW PATH")
    print("=" * 50)
    
    # Load dataset
    questions = load_financeqa_dataset()
    if not questions:
        return
    
    # Use question #1 for testing (Costco question)
    question_data = {
        'question_num': 1,
        'question': questions[0].get('question', 'N/A'),
        'context': questions[0].get('context', ''),
        'expected_answer': questions[0].get('answer', 'N/A'),
        'company': questions[0].get('company', 'N/A'),
        'question_type': questions[0].get('question_type', 'N/A')
    }
    
    print(f"\n📋 Testing with Question #{question_data['question_num']}")
    print(f"Question: {question_data['question']}")
    print(f"Company: {question_data['company']}")
    print(f"Type: {question_data['question_type']}")
    
    # Initialize agent
    agent = FinanceQAAgent()
    
    # Step 1: Generate formula and keywords
    print(f"\n🔧 Step 1: Generating formula and keywords...")
    analysis = analyze_question_requirements(question_data['question'], agent)
    print(f"Formula Analysis: {analysis}")
    
    # Extract formula
    formula = extract_formula_from_analysis(analysis)
    print(f"Extracted Formula: {formula}")
    
    # Step 2: Keyword search
    print(f"\n🔍 Step 2: Keyword search in context...")
    if question_data['context'] and question_data['context'].strip():
        search_result = search_key_terms_in_context(question_data, agent, analysis)
        print(f"Keyword Search Results: {search_result[:200]}...")
        
        # Extract values and calculate result
        extraction_result = extract_values_from_context(question_data, formula, search_result, agent)
        print(f"Extracted Values: {extraction_result['values']}")
        print(f"Calculation: {extraction_result['calculation']}")
        print(f"Result: {extraction_result['result']}")
    else:
        print("⚠️ No context available for keyword search")
        extraction_result = {
            'values': {},
            'calculation': "No calculation possible without context",
            'result': "No result available"
        }
    
    # Step 3: Assess information completeness
    print(f"\n🤔 Step 3: Assessing information completeness...")
    assessment_result = assess_information_completeness(question_data, formula, extraction_result, agent)
    
    print(f"Complete Answer Possible: {assessment_result['complete_answer']}")
    print(f"Missing Information: {assessment_result['missing_info']}")
    print(f"Additional Help Needed: {assessment_result['additional_help']}")
    print(f"Confidence Level: {assessment_result['confidence']}")
    
    
    # Step 4: If not enough info, enter iterative loop
    if assessment_result['complete_answer'] == "No":
        print(f"\n🔄 Step 4: Entering iterative tool selection loop...")
        loop_result = test_iterative_tool_selection_loop(question_data, formula, agent, assessment_result)
        
        print(f"\n📊 Final Loop Results:")
        print(f"Tool calls made: {loop_result['tool_calls_made']}")
        print(f"Reason for ending: {loop_result['reason']}")
        print(f"Final assessment: Complete Answer Possible = {loop_result['final_assessment']['complete_answer']}")
        print(f"Final confidence: {loop_result['final_assessment']['confidence']}")
        
        # Show calculator expression if available
        if 'calculator_expression' in loop_result:
            calc_result = loop_result['calculator_expression']
            print(f"\n🧮 Calculator Expression:")
            print(f"   Variables found: {calc_result['variables_found']}")
            print(f"   Values extracted: {calc_result['values_extracted']}")
            print(f"   Expression: {calc_result['calculator_expression']}")
            print(f"   Explanation: {calc_result['explanation']}")
            
            # Execute the calculator expression
            if calc_result['calculator_expression'] and calc_result['calculator_expression'] != "Error in extraction":
                try:
                    calculator_result = agent.financial_calculator_tool.invoke({
                        "expression": calc_result['calculator_expression']
                    })
                    print(f"\n🔢 Calculator Result:")
                    print(f"   {calculator_result}")
                    
                    # Extract just the numerical result
                    calc_result_str = str(calculator_result)
                    display_final_result(calc_result_str, question_data)
                except Exception as e:
                    print(f"❌ Error executing calculator: {str(e)}")
        

        
        # Show simple final result if calculator was used
        if 'calculator_expression' in loop_result:
            calc_result = loop_result['calculator_expression']
            if calc_result['calculator_expression'] and calc_result['calculator_expression'] != "Error in extraction":
                try:
                    calculator_result = agent.financial_calculator_tool.invoke({
                        "expression": calc_result['calculator_expression']
                    })
                    calc_result_str = str(calculator_result)
                    display_final_result(calc_result_str, question_data)
                except Exception as e:
                    print(f"❌ Error executing calculator: {str(e)}")
    else:
        print(f"\n✅ Step 4: Skipping loop - complete answer possible!")
    
    print(f"\n✅ Control flow test completed!")

def test_iterative_tool_selection_loop(question_data, formula, agent, initial_assessment_result):
    """Test the iterative tool selection loop with controlled inputs."""
    print(f"\n🔄 Starting iterative tool selection loop...")
    print(f"📊 Initial assessment: Complete Answer Possible = {initial_assessment_result['complete_answer']}")
    
    # Initialize loop state
    tool_calls_made = 0
    max_tool_calls = 5
    current_working_answer = {
        'question_num': question_data['question_num'],
        'question': question_data['question'],
        'formula': formula,
        'tool_results': [],
        'timestamp': __import__('datetime').datetime.now().isoformat()
    }
    
    # Predefined tool sequence for testing
    test_tool_sequence = [
        {
            'tool_choice': 'finnhub_search_tool',
            'tool_input': 'COST',
            'reasoning': 'Need current financial data for Costco'
        },
        {
            'tool_choice': 'web_search_tool', 
            'tool_input': 'Costco financial news and analysis 2024',
            'reasoning': 'Need recent news and market analysis'
        },
        {
            'tool_choice': 'knowledge_base_tool',
            'tool_input': 'gross profit margin calculation',
            'reasoning': 'Need financial concept definitions'
        },
        {
            'tool_choice': 'direct_rag_search_tool',
            'tool_input': '1',
            'reasoning': 'Need historical financial data from dataset'
        },
        {
            'tool_choice': 'fetch_webpage_content_tool',
            'tool_input': 'https://en.wikipedia.org/wiki/Costco',
            'reasoning': 'Need detailed company information'
        }
    ]
    
    while tool_calls_made < max_tool_calls:
        print(f"\n🔄 Tool call {tool_calls_made + 1}/{max_tool_calls}")
        print(f"📋 Current working answer has {len(current_working_answer['tool_results'])} tool results")
        
        # Use predefined tool sequence for testing
        if tool_calls_made < len(test_tool_sequence):
            tool_info = test_tool_sequence[tool_calls_made]
            tool_choice = tool_info['tool_choice']
            tool_input = tool_info['tool_input']
            reasoning = tool_info['reasoning']
        else:
            # If we run out of predefined tools, use the first one
            tool_info = test_tool_sequence[0]
            tool_choice = tool_info['tool_choice']
            tool_input = tool_info['tool_input']
            reasoning = tool_info['reasoning']
        
        print(f"\n🤖 Test Tool Selection:")
        print(f"   Tool: {tool_choice}")
        print(f"   Input: {tool_input}")
        print(f"   Reasoning: {reasoning}")
        print(f"   Iteration: {tool_calls_made + 1}/5")
        
        # Execute the chosen tool
        try:
            print(f"\n🔧 Executing {tool_choice}...")
            
            # Handle different parameter names for different tools
            if tool_choice == 'fetch_webpage_content_tool':
                # Always use hardcoded URL for testing
                hardcoded_url = "https://en.wikipedia.org/wiki/Costco"
                tool_result = agent.fetch_webpage_content_tool.invoke({"url": hardcoded_url})
            elif tool_choice == 'direct_rag_search_tool':
                # Convert string input to integer for question_num
                try:
                    question_num = int(tool_input)
                    tool_result = agent.direct_rag_search_tool.invoke({"question_num": question_num})
                except ValueError:
                    print(f"❌ Invalid question number: {tool_input}")
                    tool_calls_made += 1
                    continue
            else:
                tool_result = getattr(agent, tool_choice).invoke({"query": tool_input})
            
            print(f"✅ Tool execution successful")
            print(f"📊 Result preview: {str(tool_result)[:200]}...")
            
            # Add tool result to working answer
            current_working_answer['tool_results'].append({
                'tool_name': tool_choice,
                'tool_input': tool_input,
                'tool_result': tool_result,
                'reasoning': reasoning,
                'timestamp': __import__('datetime').datetime.now().isoformat()
            })
            
            # Update working answer file
            update_working_answer_with_tool_results(question_data, current_working_answer)
            
            # Reassess information completeness
            print(f"\n🤔 Reassessing information completeness...")
            updated_assessment = assess_information_completeness_with_tool_results(
                question_data, formula, current_working_answer, agent
            )
            
            # For testing: force "Yes" after 3 iterations to test calculator functionality
            if tool_calls_made >= 2:  # After 3 tool calls (0, 1, 2)
                print(f"\n🧪 TESTING: Forcing assessment to 'Yes' to test calculator functionality")
                updated_assessment['complete_answer'] = "Yes"
                updated_assessment['confidence'] = "High"
                updated_assessment['missing_info'] = "None - testing calculator"
                updated_assessment['final_answer'] = "Testing calculator expression extraction"
            
            print(f"📊 Updated Assessment:")
            print(f"   Complete Answer Possible: {updated_assessment['complete_answer']}")
            print(f"   Missing Information: {updated_assessment['missing_info']}")
            print(f"   Confidence Level: {updated_assessment['confidence']}")
            
            # Check if we can break the loop
            if updated_assessment['complete_answer'] == "Yes":
                print(f"\n🎯 Complete answer possible! Breaking loop.")
                
                # Extract formula values and prepare for calculator
                calculator_expression = extract_formula_values_for_calculator(
                    question_data, formula, current_working_answer, agent
                )
                
                return {
                    'final_assessment': updated_assessment,
                    'working_answer': current_working_answer,
                    'tool_calls_made': tool_calls_made + 1,
                    'reason': 'Complete answer achieved',
                    'calculator_expression': calculator_expression
                }
            
            tool_calls_made += 1
            
        except Exception as e:
            print(f"❌ Error executing {tool_choice}: {str(e)}")
            tool_calls_made += 1
            continue
    
    # If we reach here, we've hit the maximum tool calls
    print(f"\n⏰ Maximum tool calls ({max_tool_calls}) reached. Final assessment:")
    final_assessment = assess_information_completeness_with_tool_results(
        question_data, formula, current_working_answer, agent
    )
    
    return {
        'final_assessment': final_assessment,
        'working_answer': current_working_answer,
        'tool_calls_made': tool_calls_made,
        'reason': 'Maximum tool calls reached'
    }

# Import the helper functions from control.py
def analyze_question_requirements(question, agent):
    """Analyze what information is needed to answer the question using the formula analysis tool."""
    try:
        result = agent.formula_analysis_tool.invoke({"question": question})
        return result
    except Exception as e:
        return f"Error analyzing question requirements: {str(e)}"

def extract_formula_from_analysis(formula_analysis):
    """Extract the formula from the formula analysis output."""
    try:
        lines = formula_analysis.split('\n')
        for line in lines:
            if line.startswith('Formula:'):
                formula = line.replace('Formula:', '').strip()
                return formula
        return "Formula not found in analysis"
    except Exception as e:
        return f"Error extracting formula: {str(e)}"

def search_key_terms_in_context(question_data, agent, formula_analysis):
    """Search for key terms in the context using the key terms search tool."""
    try:
        # Parse the formula analysis to extract key terms and synonyms
        lines = formula_analysis.split('\n')
        key_terms = []
        synonyms = {}
        
        for line in lines:
            if line.startswith('Key Terms:'):
                # Extract key terms from the line
                terms_str = line.replace('Key Terms:', '').strip()
                key_terms = [term.strip().strip('[]\'\"') for term in terms_str.strip('[]').split(',')]
            elif line.startswith('Synonyms:'):
                # Extract synonyms from the line
                synonyms_str = line.replace('Synonyms:', '').strip()
                # This is a simplified approach - you might want more robust JSON parsing
                try:
                    import ast
                    synonyms = ast.literal_eval(synonyms_str)
                except:
                    # Fallback: create simple synonyms
                    synonyms = {}
                    for term in key_terms:
                        synonyms[term] = [term.lower()]
        
        if not key_terms:
            return "Could not extract key terms from formula analysis."
        
        # Convert to string format for the tool
        key_terms_str = str(key_terms)
        synonyms_str = str(synonyms)
        
        # Search for key terms in context
        search_result = agent.key_terms_search_tool.invoke({
            "key_terms": key_terms_str,
            "synonyms": synonyms_str,
            "context": question_data['context']
        })
        
        return search_result
        
    except Exception as e:
        return f"Error searching for key terms: {str(e)}"

def extract_values_from_context(question_data, formula, context_chunks, agent):
    """Extract the necessary values from context chunks and calculate the result."""
    try:
        # Create a prompt for the LLM to extract values
        extraction_prompt = f"""
        You are a financial analysis expert. Given this formula and context chunks, extract the necessary values.
        
        FORMULA: {formula}
        
        CONTEXT CHUNKS:
        {context_chunks}
        
        Please:
        1. Identify the specific values needed for the formula
        2. Extract those values from the context chunks
        3. Provide the values in a clear format
        4. Calculate the result using the formula
        
        Respond in this exact format:
        VALUES: {{"variable1": value1, "variable2": value2, ...}}
        CALCULATION: Show the calculation step by step
        RESULT: The final calculated result
        """
        
        # Use the LLM to extract values
        llm = agent.llm
        response = llm.invoke(extraction_prompt)
        
        # Parse the response
        lines = response.content.split('\n')
        values = {}
        calculation = ""
        result = ""
        
        for line in lines:
            if line.startswith('VALUES:'):
                values_str = line.replace('VALUES:', '').strip()
                try:
                    import ast
                    values = ast.literal_eval(values_str)
                except:
                    values = {}
            elif line.startswith('CALCULATION:'):
                calculation = line.replace('CALCULATION:', '').strip()
            elif line.startswith('RESULT:'):
                result = line.replace('RESULT:', '').strip()
        
        return {
            'values': values,
            'calculation': calculation,
            'result': result,
            'full_response': response.content
        }
        
    except Exception as e:
        return {
            'values': {},
            'calculation': f"Error: {str(e)}",
            'result': "Error in calculation",
            'full_response': f"Error extracting values: {str(e)}"
        }

def assess_information_completeness(question_data, formula, extraction_result, agent):
    """Ask the LLM if it needs more information to answer the question completely."""
    try:
        # Create a prompt to assess if more information is needed
        assessment_prompt = f"""
        You are a financial analysis expert. Assess whether you have enough information to provide a complete answer.
        
        ORIGINAL QUESTION: {question_data['question']}
        
        FORMULA IDENTIFIED: {formula}
        
        VALUES EXTRACTED: {extraction_result['values']}
        
        CALCULATION PERFORMED: {extraction_result['calculation']}
        
        RESULT: {extraction_result['result']}
        
        CONTEXT AVAILABLE: {question_data['context'][:500] if question_data['context'] else 'No context available'}
        
        Please assess:
        1. Do you have all the necessary information to answer the question completely?
        2. Are there any missing pieces of information?
        3. What additional information would be helpful?
        4. Can you provide a complete answer with the current information?
        
        Respond in this exact format:
        COMPLETE_ANSWER: Yes/No
        MISSING_INFO: List any missing information or "None"
        ADDITIONAL_HELP: What additional information would be helpful or "None"
        FINAL_ANSWER: Your complete answer to the original question
        CONFIDENCE: High/Medium/Low
        """
        
        # Use the LLM to assess completeness
        llm = agent.llm
        response = llm.invoke(assessment_prompt)
        
        # Parse the response
        lines = response.content.split('\n')
        complete_answer = "Unknown"
        missing_info = "Unknown"
        additional_help = "Unknown"
        final_answer = "Unknown"
        confidence = "Unknown"
        
        for line in lines:
            if line.startswith('COMPLETE_ANSWER:'):
                complete_answer = line.replace('COMPLETE_ANSWER:', '').strip()
            elif line.startswith('MISSING_INFO:'):
                missing_info = line.replace('MISSING_INFO:', '').strip()
            elif line.startswith('ADDITIONAL_HELP:'):
                additional_help = line.replace('ADDITIONAL_HELP:', '').strip()
            elif line.startswith('FINAL_ANSWER:'):
                final_answer = line.replace('FINAL_ANSWER:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                confidence = line.replace('CONFIDENCE:', '').strip()
        
        # TEMPORARY: Hardcode to "No" for testing
        complete_answer = "No"
        missing_info = "Current stock price and financial data"
        additional_help = "Real-time stock data and financial statements"
        
        return {
            'complete_answer': complete_answer,
            'missing_info': missing_info,
            'additional_help': additional_help,
            'final_answer': final_answer,
            'confidence': confidence,
            'full_response': response.content
        }
        
    except Exception as e:
        return {
            'complete_answer': "Error",
            'missing_info': f"Error: {str(e)}",
            'additional_help': "Error in assessment",
            'final_answer': "Error in assessment",
            'confidence': "Error",
            'full_response': f"Error assessing completeness: {str(e)}"
        }

def update_working_answer_with_tool_results(question_data, working_answer):
    """Update the working answer file with new tool results."""
    try:
        filename = f"working_answers/working_answer_{question_data['question_num']}.json"
        
        # Create working_answers directory if it doesn't exist
        import os
        os.makedirs("working_answers", exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(working_answer, f, indent=2)
        
        return f"✅ Working answer updated: {filename}"
    except Exception as e:
        return f"❌ Error updating working answer: {str(e)}"

def assess_information_completeness_with_tool_results(question_data, formula, working_answer, agent):
    """Assess if we have enough information to answer the question completely, considering tool results."""
    try:
        # Create a comprehensive prompt that includes all tool results
        tool_results_summary = ""
        for i, result in enumerate(working_answer['tool_results']):
            tool_results_summary += f"\nTool {i+1}: {result['tool_name']}"
            tool_results_summary += f"\nInput: {result['tool_input']}"
            tool_results_summary += f"\nResult: {str(result['tool_result'])[:500]}..."
            tool_results_summary += f"\nReasoning: {result['reasoning']}\n"
        
        assessment_prompt = f"""
        You are a financial analysis expert. Assess whether you have enough information to provide a complete answer.
        
        ORIGINAL QUESTION: {question_data['question']}
        
        FORMULA IDENTIFIED: {formula}
        
        TOOL RESULTS GATHERED:
        {tool_results_summary}
        
        CONTEXT AVAILABLE: {question_data['context'][:500] if question_data['context'] else 'No context available'}
        
        Please assess:
        1. Do you have all the necessary information to answer the question completely?
        2. Are there any missing pieces of information?
        3. What additional information would be helpful?
        4. Can you provide a complete answer with the current information?
        
        Respond in this exact format:
        COMPLETE_ANSWER: Yes/No
        MISSING_INFO: List any missing information or "None"
        ADDITIONAL_HELP: What additional information would be helpful or "None"
        FINAL_ANSWER: Your complete answer to the original question
        CONFIDENCE: High/Medium/Low
        """
        
        # Use the LLM to assess completeness
        llm = agent.llm
        response = llm.invoke(assessment_prompt)
        
        # Parse the response
        lines = response.content.split('\n')
        complete_answer = "Unknown"
        missing_info = "Unknown"
        additional_help = "Unknown"
        final_answer = "Unknown"
        confidence = "Unknown"
        
        for line in lines:
            if line.startswith('COMPLETE_ANSWER:'):
                complete_answer = line.replace('COMPLETE_ANSWER:', '').strip()
            elif line.startswith('MISSING_INFO:'):
                missing_info = line.replace('MISSING_INFO:', '').strip()
            elif line.startswith('ADDITIONAL_HELP:'):
                additional_help = line.replace('ADDITIONAL_HELP:', '').strip()
            elif line.startswith('FINAL_ANSWER:'):
                final_answer = line.replace('FINAL_ANSWER:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                confidence = line.replace('CONFIDENCE:', '').strip()
        
        # Let the LLM make the natural assessment
        # (removed hardcoded override)
        
        return {
            'complete_answer': complete_answer,
            'missing_info': missing_info,
            'additional_help': additional_help,
            'final_answer': final_answer,
            'confidence': confidence,
            'full_response': response.content
        }
        
    except Exception as e:
        return {
            'complete_answer': "Error",
            'missing_info': f"Error: {str(e)}",
            'additional_help': "Error in assessment",
            'final_answer': "Error in assessment",
            'confidence': "Error",
            'full_response': f"Error assessing completeness: {str(e)}"
        }

def extract_formula_values_for_calculator(question_data, formula, working_answer, agent):
    """
    Extract values from tool results and prepare a calculator expression.
    
    This function:
    1. Analyzes the formula to identify required variables
    2. Searches through tool results to find relevant values
    3. Creates a calculator-ready expression with actual values
    """
    try:
        # Create a comprehensive summary of all tool results
        tool_results_summary = ""
        for i, result in enumerate(working_answer['tool_results']):
            tool_results_summary += f"\nTool {i+1}: {result['tool_name']}"
            tool_results_summary += f"\nInput: {result['tool_input']}"
            tool_results_summary += f"\nResult: {str(result['tool_result'])[:500]}..."
            tool_results_summary += f"\nReasoning: {result['reasoning']}\n"
        
        # Prompt the LLM to extract values and create calculator expression
        extraction_prompt = f"""
        You are a financial analysis expert. Given the formula and all the tool results gathered, 
        extract the necessary values and create a calculator-ready expression.
        
        ORIGINAL QUESTION: {question_data['question']}
        
        FORMULA: {formula}
        
        TOOL RESULTS GATHERED:
        {tool_results_summary}
        
        CONTEXT AVAILABLE: {question_data['context'][:500] if question_data['context'] else 'No context available'}
        
        Please:
        1. Identify the variables needed for the formula
        2. Extract the actual numerical values from the tool results
        3. Create a calculator expression with the actual values substituted
        4. Ensure the expression uses only basic mathematical operators (+, -, *, /, (, ))
        5. Make sure all values are numerical (no text, no currency symbols)
        
        Respond in this exact format:
        VARIABLES_FOUND: [list of variables found]
        VALUES_EXTRACTED: {{"variable1": value1, "variable2": value2, ...}}
        CALCULATOR_EXPRESSION: [expression with actual values, e.g., "1000 - 500"]
        EXPLANATION: [brief explanation of what the calculation does]
        """
        
        # Use the LLM to extract values and create expression
        llm = agent.llm
        response = llm.invoke(extraction_prompt)
        
        # Parse the response
        lines = response.content.split('\n')
        variables_found = []
        values_extracted = {}
        calculator_expression = ""
        explanation = ""
        
        for line in lines:
            if line.startswith('VARIABLES_FOUND:'):
                variables_str = line.replace('VARIABLES_FOUND:', '').strip()
                try:
                    import ast
                    variables_found = ast.literal_eval(variables_str)
                except:
                    variables_found = []
            elif line.startswith('VALUES_EXTRACTED:'):
                values_str = line.replace('VALUES_EXTRACTED:', '').strip()
                try:
                    import ast
                    values_extracted = ast.literal_eval(values_str)
                except:
                    values_extracted = {}
            elif line.startswith('CALCULATOR_EXPRESSION:'):
                calculator_expression = line.replace('CALCULATOR_EXPRESSION:', '').strip()
            elif line.startswith('EXPLANATION:'):
                explanation = line.replace('EXPLANATION:', '').strip()
        
        print(f"\n🧮 Formula Value Extraction:")
        print(f"   Variables found: {variables_found}")
        print(f"   Values extracted: {values_extracted}")
        print(f"   Calculator expression: {calculator_expression}")
        print(f"   Explanation: {explanation}")
        
        return {
            'variables_found': variables_found,
            'values_extracted': values_extracted,
            'calculator_expression': calculator_expression,
            'explanation': explanation,
            'full_response': response.content
        }
        
    except Exception as e:
        print(f"❌ Error extracting formula values: {str(e)}")
        return {
            'variables_found': [],
            'values_extracted': {},
            'calculator_expression': "Error in extraction",
            'explanation': f"Error: {str(e)}",
            'full_response': f"Error extracting formula values: {str(e)}"
        }

if __name__ == "__main__":
    test_control_flow()