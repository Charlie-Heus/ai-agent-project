#!/usr/bin/env python3
"""
Simplified RAG Demo for FinanceQA Dataset
Only shows the 2 requested demo options
"""

import os
import json
import pickle
import hashlib
import signal
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# LangChain imports
from langchain_openai import OpenAIEmbeddings

def timeout_handler(signum, frame):
    """Handle timeout for API calls"""
    raise TimeoutError("API call timed out")

def safe_embed_query(embeddings, text, timeout=30):
    """Safely embed text with timeout"""
    # Set timeout signal
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        result = embeddings.embed_query(text)
        signal.alarm(0)  # Cancel timeout
        return result
    except TimeoutError:
        signal.alarm(0)  # Cancel timeout
        raise TimeoutError(f"Embedding request timed out after {timeout} seconds")
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        raise e

def get_cache_key(question_text: str, context_text: str) -> str:
    """Generate a unique cache key for a question and its context."""
    # Create a hash of the question and context
    content = f"{question_text}|{context_text}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def load_embeddings_cache(cache_file: str = "data/embeddings_cache.pkl") -> Dict:
    """Load embeddings from cache file."""
    cache_path = Path(cache_file)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load cache: {e}")
    return {}

def save_embeddings_cache(cache_data: Dict, cache_file: str = "data/embeddings_cache.pkl"):
    """Save embeddings to cache file."""
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(exist_ok=True)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"⚠️  Warning: Could not save cache: {e}")

def get_cached_embeddings(question_text: str, context_text: str, cache_data: Dict) -> Optional[Dict]:
    """Get cached embeddings for a question and context."""
    cache_key = get_cache_key(question_text, context_text)
    return cache_data.get(cache_key)

def cache_embeddings(question_text: str, context_text: str, question_embedding: List[float], 
                     chunk_embeddings: List[List[float]], cache_data: Dict):
    """Cache embeddings for a question and context."""
    cache_key = get_cache_key(question_text, context_text)
    cache_data[cache_key] = {
        'question_embedding': question_embedding,
        'chunk_embeddings': chunk_embeddings,
        'timestamp': os.path.getmtime('data/financeqa_test.jsonl')  # Use dataset modification time
    }

def load_financeqa_test_questions(file_path: str = "data/financeqa_test.jsonl") -> List[Dict]:
    """Load test questions from FinanceQA dataset."""
    
    if not Path(file_path).exists():
        print(f"❌ FinanceQA test file not found: {file_path}")
        return []
    
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                questions.append(data)
            except json.JSONDecodeError:
                continue
    
    return questions

def split_context_into_chunks(context_text: str, chunk_size: int = 200, overlap: int = 100) -> List[str]:
    """Split context text into meaningful chunks with overlap."""
    if not context_text:
        return []
    
    # Split by sentences first
    sentences = context_text.split('. ')
    
    chunks = []
    current_chunk = ""
    overlap_text = ""
    
    for sentence in sentences:
        # Add period back if it was removed
        if not sentence.endswith('.'):
            sentence += '.'
        
        # If adding this sentence would make chunk too long, start new chunk
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            # Add current chunk
            chunks.append(current_chunk.strip())
            
            # Calculate overlap: take the last portion of current chunk
            if overlap > 0 and len(current_chunk) > overlap:
                # Find the last sentence boundary within the overlap region
                overlap_region = current_chunk[-overlap:]
                last_sentence_start = overlap_region.find('. ')
                if last_sentence_start != -1:
                    overlap_text = overlap_region[last_sentence_start + 2:]  # +2 to skip '. '
                else:
                    overlap_text = overlap_region
            else:
                overlap_text = current_chunk
            
            # Start new chunk with overlap
            current_chunk = overlap_text + " " + sentence if overlap_text else sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def run_demo_first_4_questions():
    """Run demo with first 4 questions and their 10 most relevant context chunks"""
    print("\n" + "="*60)
    print("DEMO: FIRST 4 QUESTIONS WITH TOP 10 CONTEXT CHUNKS")
    print("="*60)
    
    try:
        # Load FinanceQA data
        test_questions = load_financeqa_test_questions()
        if not test_questions:
            print("❌ No test questions found")
            return
        
        # Load embeddings cache
        cache_data = load_embeddings_cache()
        print(f"📦 Loaded cache with {len(cache_data)} cached embeddings")
        
        # Get first 4 questions
        first_4_docs = test_questions[:4]
        
        for i, doc in enumerate(first_4_docs, 1):
            print(f"\n🧪 QUESTION {i}")
            print("-" * 40)
            print(f"Question ID: {i}")
            print(f"Question: {doc.get('question', '')}")
            print(f"Answer: {doc.get('answer', '')}")
            
            context_text = doc.get('context', '')
            if context_text and context_text.strip():
                print(f"\n📊 Context Analysis:")
                
                # Split context into chunks
                chunks = split_context_into_chunks(context_text)
                
                print(f"   Total context length: {len(context_text)} characters")
                print(f"   Number of chunks: {len(chunks)}")
                print(f"   Average chunk length: {sum(len(chunk) for chunk in chunks)/len(chunks):.0f} characters")
                print(f"   Chunk size: 200 characters with 100 character overlap")
                
                # Show overlap statistics
                if len(chunks) > 1:
                    overlap_analysis = []
                    for i in range(len(chunks) - 1):
                        chunk1 = chunks[i]
                        chunk2 = chunks[i + 1]
                        # Find common text at the end of chunk1 and start of chunk2
                        common_length = 0
                        for j in range(min(len(chunk1), len(chunk2))):
                            if chunk1[-(j+1):] == chunk2[:j+1]:
                                common_length = j + 1
                        overlap_analysis.append(common_length)
                    
                    avg_overlap = sum(overlap_analysis) / len(overlap_analysis) if overlap_analysis else 0
                    print(f"   Average actual overlap: {avg_overlap:.0f} characters")
                
                # Check cache first
                cached_data = get_cached_embeddings(doc.get('question', ''), context_text, cache_data)
                
                if cached_data:
                    print(f"\n⚡ Using cached embeddings for question {i}")
                    question_embedding = cached_data['question_embedding']
                    chunk_embeddings = cached_data['chunk_embeddings']
                else:
                    print(f"\n🔄 Processing embeddings for question {i}...")
                    
                    # Initialize embeddings for this specific question
                    embeddings = OpenAIEmbeddings()
                    
                    # Get question embedding for similarity comparison
                    question_embedding = safe_embed_query(embeddings, doc.get('question', ''))
                    
                    # Get chunk embeddings with timeout handling
                    chunk_embeddings = []
                    successful_chunks = 0
                    for j, chunk in enumerate(chunks):
                        print(f"   Processing chunk {j+1}/{len(chunks)} (successful: {successful_chunks})...", end='\r')
                        try:
                            chunk_embedding = safe_embed_query(embeddings, chunk)
                            chunk_embeddings.append(chunk_embedding)
                            successful_chunks += 1
                        except Exception as e:
                            print(f"\n⚠️  Error processing chunk {j+1}: {e}")
                            # Continue with remaining chunks
                            continue
                    
                    # Only cache if we got all embeddings
                    if len(chunk_embeddings) == len(chunks):
                        cache_embeddings(doc.get('question', ''), context_text, question_embedding, chunk_embeddings, cache_data)
                        print(f"\n💾 Cached embeddings for question {i}")
                    else:
                        print(f"\n⚠️  Skipped caching due to incomplete embeddings ({len(chunk_embeddings)}/{len(chunks)})")
                
                # Calculate similarity for each chunk
                chunk_similarities = []
                for j, chunk_embedding in enumerate(chunk_embeddings):
                    similarity = np.dot(question_embedding, chunk_embedding) / (
                        np.linalg.norm(question_embedding) * np.linalg.norm(chunk_embedding)
                    )
                    chunk_similarities.append({
                        'chunk_index': j,
                        'chunk_text': chunks[j],
                        'similarity': similarity
                    })
                
                print(f"\n✅ Completed processing {len(chunks)} chunks")
                
                # Sort by similarity (highest first)
                chunk_similarities.sort(key=lambda x: x['similarity'], reverse=True)
                
                print(f"\n📋 Top 10 Most Relevant Context Chunks (by similarity):")
                for j, chunk_data in enumerate(chunk_similarities[:10], 1):
                    chunk_text = chunk_data['chunk_text']
                    similarity = chunk_data['similarity']
                    print(f"   {j:2d}. Similarity: {similarity:.4f}")
                    print(f"       {chunk_text[:200]}...")
                    if len(chunk_text) > 200:
                        print(f"       (Length: {len(chunk_text)} characters)")
                    print()
                
            else:
                print(f"\n⚠️  No context available for this question")
            
            print("\n" + "="*60)
        
        # Save cache after processing
        save_embeddings_cache(cache_data)
        print(f"\n💾 Saved cache with {len(cache_data)} embeddings")
            
    except Exception as e:
        print(f"❌ Error running demo: {e}")

def pick_question_demo():
    """Let user pick a question and show its 10 most relevant context chunks"""
    print("\n" + "="*60)
    print("PICK A QUESTION DEMO")
    print("="*60)
    
    try:
        # Load FinanceQA data
        test_questions = load_financeqa_test_questions()
        if not test_questions:
            print("❌ No test questions found")
            return
        
        # Load embeddings cache
        cache_data = load_embeddings_cache()
        print(f"📦 Loaded cache with {len(cache_data)} cached embeddings")
        
        total_questions = len(test_questions)
        print(f"Available questions: 1-{total_questions}")
        
        while True:
            try:
                choice = input(f"\nEnter question number (1-{total_questions}): ").strip()
                question_num = int(choice)
                
                if 1 <= question_num <= total_questions:
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {total_questions}")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Get the selected question
        selected_doc = test_questions[question_num - 1]
        
        print(f"\n🧪 SELECTED QUESTION {question_num}")
        print("-" * 40)
        print(f"Question ID: {question_num}")
        print(f"Question: {selected_doc.get('question', '')}")
        print(f"Answer: {selected_doc.get('answer', '')}")
        
        context_text = selected_doc.get('context', '')
        if context_text and context_text.strip():
            print(f"\n📊 Context Analysis:")
            
            # Split context into chunks
            chunks = split_context_into_chunks(context_text)
            
            print(f"   Total context length: {len(context_text)} characters")
            print(f"   Number of chunks: {len(chunks)}")
            print(f"   Average chunk length: {sum(len(chunk) for chunk in chunks)/len(chunks):.0f} characters")
            print(f"   Chunk size: 200 characters with 100 character overlap")
            
            # Show overlap statistics
            if len(chunks) > 1:
                overlap_analysis = []
                for i in range(len(chunks) - 1):
                    chunk1 = chunks[i]
                    chunk2 = chunks[i + 1]
                    # Find common text at the end of chunk1 and start of chunk2
                    common_length = 0
                    for j in range(min(len(chunk1), len(chunk2))):
                        if chunk1[-(j+1):] == chunk2[:j+1]:
                            common_length = j + 1
                    overlap_analysis.append(common_length)
                
                avg_overlap = sum(overlap_analysis) / len(overlap_analysis) if overlap_analysis else 0
                print(f"   Average actual overlap: {avg_overlap:.0f} characters")
            
            # Check cache first
            cached_data = get_cached_embeddings(selected_doc.get('question', ''), context_text, cache_data)
            
            if cached_data:
                print(f"\n⚡ Using cached embeddings for question {question_num}")
                question_embedding = cached_data['question_embedding']
                chunk_embeddings = cached_data['chunk_embeddings']
            else:
                print(f"\n🔄 Processing embeddings for question {question_num}...")
                
                # Initialize embeddings for this specific question
                embeddings = OpenAIEmbeddings()
                
                # Get question embedding for similarity comparison
                question_embedding = safe_embed_query(embeddings, selected_doc.get('question', ''))
                
                # Get chunk embeddings with timeout handling
                chunk_embeddings = []
                successful_chunks = 0
                for j, chunk in enumerate(chunks):
                    print(f"   Processing chunk {j+1}/{len(chunks)} (successful: {successful_chunks})...", end='\r')
                    try:
                        chunk_embedding = safe_embed_query(embeddings, chunk)
                        chunk_embeddings.append(chunk_embedding)
                        successful_chunks += 1
                    except Exception as e:
                        print(f"\n⚠️  Error processing chunk {j+1}: {e}")
                        # Continue with remaining chunks
                        continue
                
                # Only cache if we got all embeddings
                if len(chunk_embeddings) == len(chunks):
                    cache_embeddings(selected_doc.get('question', ''), context_text, question_embedding, chunk_embeddings, cache_data)
                    print(f"\n💾 Cached embeddings for question {question_num}")
                else:
                    print(f"\n⚠️  Skipped caching due to incomplete embeddings ({len(chunk_embeddings)}/{len(chunks)})")
            
            # Calculate similarity for each chunk
            chunk_similarities = []
            for j, chunk_embedding in enumerate(chunk_embeddings):
                similarity = np.dot(question_embedding, chunk_embedding) / (
                    np.linalg.norm(question_embedding) * np.linalg.norm(chunk_embedding)
                )
                chunk_similarities.append({
                    'chunk_index': j,
                    'chunk_text': chunks[j],
                    'similarity': similarity
                })
            
            print(f"\n✅ Completed processing {len(chunks)} chunks")
            
            # Sort by similarity (highest first)
            chunk_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            print(f"\n📋 Top 10 Most Relevant Context Chunks (by similarity):")
            for j, chunk_data in enumerate(chunk_similarities[:10], 1):
                chunk_text = chunk_data['chunk_text']
                similarity = chunk_data['similarity']
                print(f"   {j:2d}. Similarity: {similarity:.4f}")
                print(f"       {chunk_text[:200]}...")
                if len(chunk_text) > 200:
                    print(f"       (Length: {len(chunk_text)} characters)")
                print()
            
        else:
            print(f"\n⚠️  No context available for this question")
        
        # Save cache after processing
        save_embeddings_cache(cache_data)
        print(f"\n💾 Saved cache with {len(cache_data)} embeddings")
        
    except Exception as e:
        print(f"❌ Error in pick question demo: {e}")

def check_environment():
    """Check if the environment is properly configured"""
    print("🔍 Checking Environment...")
    
    # Check for OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("💡 Please add your OpenAI API key to your .env file")
        return False
    else:
        print(f"✅ OPENAI_API_KEY found (length: {len(openai_key)} chars)")
    
    # Check if FinanceQA dataset exists
    data_path = Path("data/financeqa_test.jsonl")
    if not data_path.exists():
        print("❌ FinanceQA dataset not found at data/financeqa_test.jsonl")
        print("💡 Please ensure the FinanceQA dataset is available")
        return False
    else:
        print("✅ FinanceQA dataset found")
        
        # Load and check dataset
        try:
            test_questions = load_financeqa_test_questions()
            if test_questions:
                print(f"✅ FinanceQA dataset loaded successfully ({len(test_questions)} questions)")
            else:
                print("❌ No valid questions found in dataset")
                return False
        except Exception as e:
            print(f"❌ Error loading FinanceQA dataset: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("🧠 RAG Demo for FinanceQA Dataset")
    print("=" * 50)
    
    # Step 1: Check environment
    if not check_environment():
        print("\n❌ Environment check failed.")
        exit(1)
    else:
        print("\n✅ Environment check passed!")
    
    print("\nWhat would you like to do?")
    print("1. Run Demo (first 4 questions)")
    print("2. Pick Question Demo")
    
    choice = input("\nSelect option (1-2): ").strip()
    
    if choice == "1":
        run_demo_first_4_questions()
    elif choice == "2":
        pick_question_demo()
    else:
        print("Invalid choice. Running first 4 questions demo...")
        run_demo_first_4_questions()
    
    print("\n✅ Demo complete!") 