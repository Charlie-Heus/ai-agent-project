#!/usr/bin/env python3
"""
FinanceQA Dataset Explorer

This script provides utilities for exploring and analyzing the FinanceQA dataset
that has already been downloaded to data/financeqa_test.jsonl.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def load_dataset(data_path: str = "data") -> List[Dict[str, Any]]:
    """
    Load the FinanceQA dataset from the JSONL file.
    
    Args:
        data_path: Path to the dataset directory
        
    Returns:
        List of dictionaries containing the dataset
    """
    data_dir = Path(data_path)
    jsonl_path = data_dir / "financeqa_test.jsonl"
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"No dataset found at {jsonl_path}")
    
    print(f"Loading dataset from {jsonl_path}")
    
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    
    return data


def write_dataset_summary(data: List[Dict[str, Any]], output_file: str = "data/dataset_summary.txt"):
    """
    Write dataset summary to a text file.
    
    Args:
        data: List of dictionaries containing the dataset
        output_file: Path to output file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Total number of questions
        f.write(f"Total number of questions: {len(data)}\n\n")
        
        # Number of questions of each question type
        question_types = {}
        for item in data:
            q_type = item.get('question_type', 'unknown')
            question_types[q_type] = question_types.get(q_type, 0) + 1
        
        f.write("Number of questions of each question type:\n")
        for q_type, count in question_types.items():
            f.write(f"  {q_type}: {count}\n")
        f.write("\n")
        
        # Columns (from first item)
        if data:
            columns = list(data[0].keys())
            f.write(f"Columns: {columns}\n\n")
        
        # Each question with answer, company, and question type
        f.write("All questions in the dataset:\n")
        f.write("=" * 80 + "\n\n")
        
        for i, item in enumerate(data, 1):
            f.write(f"{i}. Question: {item.get('question', 'N/A')}\n")
            f.write(f"   Answer: {item.get('answer', 'N/A')}\n")
            f.write(f"   Company: {item.get('company', 'N/A')}\n")
            f.write(f"   Question Type: {item.get('question_type', 'N/A')}\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"Dataset summary written to {output_path}")


def main():
    """Main function to explore the FinanceQA dataset."""
    print("FinanceQA Dataset Explorer")
    print("=" * 40)
    
    try:
        # Load dataset
        data = load_dataset()
        print(f"✅ Dataset loaded successfully! Total questions: {len(data)}")
        
        # Write summary to file
        write_dataset_summary(data)
        
        print("✅ Exploration completed!")
        
    except Exception as e:
        print(f"❌ Error exploring dataset: {e}")
        print("Make sure the dataset is downloaded to data/financeqa_test.jsonl")


if __name__ == "__main__":
    main() 