#!/usr/bin/env python3
"""
Script to organize FinanceQA questions by type.

This script reads the financeqa_test.jsonl file and creates separate text files
for each question type: basic, assumption, and conceptual.
"""

import json
from pathlib import Path
from typing import Dict, List


def load_dataset(data_path: str = "data/financeqa_test.jsonl") -> List[Dict]:
    """
    Load the FinanceQA dataset from the JSONL file.
    
    Args:
        data_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the dataset
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    print(f"Loading dataset from {data_path}")
    
    data = []
    with open(data_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Error parsing line {line_num}: {e}")
                    continue
    
    return data


def organize_questions_by_type(data: List[Dict], output_dir: str = "data/question_types"):
    """
    Organize questions by type and save to separate files.
    
    Args:
        data: List of question dictionaries
        output_dir: Directory to save the organized files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize question type collections
    question_types = {
        'basic': [],
        'assumption': [],
        'conceptual': []
    }
    
    # Organize questions by type
    for i, question_data in enumerate(data, 1):
        question_type = question_data.get('question_type', 'unknown')
        
        if question_type in question_types:
            # Format the question with metadata
            formatted_question = format_question(question_data, i)
            question_types[question_type].append(formatted_question)
        else:
            print(f"Warning: Unknown question type '{question_type}' for question {i}")
    
    # Write each type to its own file
    for q_type, questions in question_types.items():
        file_path = output_path / f"{q_type}.txt"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# {q_type.upper()} QUESTIONS\n")
            f.write(f"# Total: {len(questions)} questions\n")
            f.write("=" * 80 + "\n\n")
            
            for question in questions:
                f.write(question)
                f.write("\n" + "-" * 80 + "\n\n")
        
        print(f"✅ Saved {len(questions)} {q_type} questions to {file_path}")
    
    # Print summary
    print(f"\n📊 Summary:")
    for q_type, questions in question_types.items():
        print(f"  {q_type}: {len(questions)} questions")


def format_question(question_data: Dict, question_number: int) -> str:
    """
    Format a question with all its details.
    
    Args:
        question_data: Dictionary containing question data
        question_number: The question number
        
    Returns:
        Formatted string with question details
    """
    lines = []
    lines.append(f"QUESTION #{question_number}")
    lines.append(f"Type: {question_data.get('question_type') or 'N/A'}")
    lines.append(f"Company: {question_data.get('company') or 'N/A'}")
    lines.append("")
    lines.append("QUESTION:")
    lines.append(question_data.get('question') or 'N/A')
    lines.append("")
    lines.append("EXPECTED ANSWER:")
    lines.append(question_data.get('answer') or 'No answer provided')
    
    return "\n".join(lines)


def main():
    """Main function to organize questions by type."""
    print("FinanceQA Question Organizer")
    print("=" * 40)
    
    try:
        # Load dataset
        data = load_dataset()
        print(f"✅ Dataset loaded successfully! Total questions: {len(data)}")
        
        # Organize questions by type
        organize_questions_by_type(data)
        
        print(f"\n✅ Organization completed!")
        print(f"📁 Check the 'data/question_types/' folder for the organized files")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the dataset is downloaded to data/financeqa_test.jsonl")


if __name__ == "__main__":
    main() 