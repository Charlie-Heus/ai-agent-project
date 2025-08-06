#!/usr/bin/env python3
"""
Script to download the FinanceQA dataset from Hugging Face.

This script downloads the FinanceQA dataset and saves it as JSONL format.
"""

import json
from pathlib import Path
from typing import Dict, Any

from datasets import load_dataset


def download_financeqa_dataset(output_dir: str = "data") -> Dict[str, Any]:
    """
    Download the FinanceQA dataset from Hugging Face.
    
    Args:
        output_dir: Directory to save the dataset
        
    Returns:
        Dictionary containing dataset information
    """
    print("Downloading FinanceQA dataset from Hugging Face...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load the dataset
        dataset = load_dataset("AfterQuery/FinanceQA")
        
        print(f"Dataset loaded successfully!")
        print(f"Dataset info: {dataset}")
        
        # Save the dataset as JSONL
        test_data = dataset['test']
        jsonl_path = output_path / "financeqa_test.jsonl"
        
        with open(jsonl_path, "w") as f:
            for row in test_data:
                f.write(json.dumps(row) + "\n")
        
        print(f"Saved test data to {jsonl_path}")
        
        return {
            "dataset": dataset,
            "test_data": test_data,
            "output_path": str(output_path)
        }
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return {}


def main():
    """Main function to download the FinanceQA dataset."""
    print("FinanceQA Dataset Downloader")
    print("="*40)
    
    # Download the dataset
    dataset_info = download_financeqa_dataset()
    
    if dataset_info:
        print(f"\nDataset downloaded successfully to: {dataset_info['output_path']}")
        print("You can now use this dataset to test your FinanceQA AI Agent!")
    else:
        print("Failed to download dataset. Please check your internet connection and try again.")


if __name__ == "__main__":
    main()