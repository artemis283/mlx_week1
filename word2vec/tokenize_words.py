import sys
from datasets import load_dataset
import torch
import torch.nn.functional as F
import re
from collections import Counter
import random
import math
import json
import os

# Set random seed for reproducibility
random.seed(42)

print("Loading dataset...")
ds = load_dataset("artemisweb/test-dataset")
raw_text = ds["train"][0]["text"]  # obtaining raw text from database

# First, convert the raw text into a list format for the tokenize function
corpus = [raw_text]  # Wrap in a list as tokenize expects a list of texts

# Subsampling threshold (same as in your tokenizer)
t = 1e-5

def tokenize(corpus, min_count=8, phrase_min_count=100, phrase_threshold=15):
    """
    Preprocess the corpus and detect phrases in one function.
    
    Args:
        corpus: List of text strings (sentences or documents)
        min_count: Minimum count for individual words to keep
        phrase_min_count: Minimum count for phrases
        phrase_threshold: Threshold for phrase detection
        
    Returns:
        processed_corpus: List of tokenized texts with phrases joined
        phrases: Dictionary of detected phrases with their scores
    """
    print("Tokenizing corpus...")
    # Step 1: Initial tokenization of all texts
    tokenized_texts = []
    for text in corpus:
        # Convert to lowercase
        text = text.lower()
        
        # Handle contractions and special terms
        text = re.sub(r"([a-z])'([a-z])", r"\1\2", text)  # Replace don't with dont

         # Remove non-alphabetic characters (except spaces)
        text = re.sub(r"[^a-z\s]", "", text)
        
        # Split into tokens
        tokens = [token for token in text.split() if token]
        tokenized_texts.append(tokens)
    
    # Step 2: Count all tokens in the corpus
    print("Counting tokens...")
    all_tokens = [token for text in tokenized_texts for token in text]
    token_counts = Counter(all_tokens)
    
    # Step 3: Filter out rare words
    print("Filtering rare words...")
    filtered_texts = []
    for tokens in tokenized_texts:
        filtered = [token for token in tokens if token_counts[token] >= min_count]
        if filtered:  # Only add non-empty texts
            filtered_texts.append(filtered)
    
    # Step 4: Count bigrams for phrase detection
    print("Detecting phrases...")
    bigram_counts = Counter()
    for tokens_list in filtered_texts:  # Use filtered texts
        for i in range(len(tokens_list) - 1):
            bigram = (tokens_list[i], tokens_list[i+1])
            bigram_counts[bigram] += 1
    
    # Step 5: Calculate phrase scores
    phrases = {}
    for bigram, count in bigram_counts.items():
        if count < phrase_min_count:
            continue
            
        word1, word2 = bigram
            
        # Calculate score using the formula from the paper
        score = (count - phrase_min_count) / (token_counts[word1] * token_counts[word2])
        
        # Apply threshold filter
        if score > phrase_threshold:
            phrases[bigram] = score
    
    # Step 6: Replace phrases in the texts
    print("Processing phrases in text...")
    processed_texts = []
    for tokens in filtered_texts:
        i = 0
        new_tokens = []
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i+1]) in phrases:
                # Join phrase with underscore
                new_tokens.append(f"{tokens[i]}_{tokens[i+1]}")
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        
        # Don't forget the last token if it wasn't part of a phrase
        if i == len(tokens) - 1:
            new_tokens.append(tokens[i])
            
        processed_texts.append(new_tokens)
    
    # Flatten the list of lists for assign_ids
    all_processed_tokens = [token for tokens_list in processed_texts for token in tokens_list]
    
    return all_processed_tokens, phrases

# Adding sub sampling 
def subsample_words(corpus, word_freq, t):
    print("Subsampling words...")
    subsampled_corpus = []
    for word in corpus:
        f_w = word_freq.get(word, 0)  # Get the frequency of the word, default to 0 if not found
        if f_w == 0:
            continue
         
        f_w = word_freq[word]
        P_w = 1 - math.sqrt(t/f_w)
        if random.random() < P_w:
            subsampled_corpus.append(word)
    
    return subsampled_corpus
        
def assign_ids(tokens):
    print("Assigning word IDs...")
    # Create a set to track words we have seen so far
    seen_words = set()
    # Create word2id mapping based on order of appearance
    word2id = {}
    for word in tokens:
        if word not in seen_words:
            seen_words.add(word)
            word2id[word] = len(word2id)  # Assign ID based on order of appearance
    
    # Reverse the mapping to get id2word
    id2word = {i: word for word, i in word2id.items()}
    
    return word2id, id2word

def save_to_json(word2id, phrases, filename="tokenizer_data.json"):
    """
    Save the word2id mapping and phrases to a JSON file.
    
    Args:
        word2id: Dictionary mapping words to their IDs
        phrases: Dictionary of detected phrases with their scores
        filename: Name of the JSON file to save
    """
    # Convert phrases keys to strings for JSON serialization
    serializable_phrases = {f"{word1}_{word2}": score for (word1, word2), score in phrases.items()}
    
    # Create a dictionary with all the data
    data = {
        "word2id": word2id,
        "phrases": serializable_phrases,
        "vocab_size": len(word2id),
        "has_unk_token": "<unk>" in word2id,
    }
    
    if "<unk>" in word2id:
        data["unk_id"] = word2id["<unk>"]
    
    # Save to JSON file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"Tokenizer data saved to {filename}")
    print(f"Total vocabulary size: {len(word2id)}")
    return filename

# Execute the tokenization pipeline
print("Starting tokenization pipeline...")
tokenised_corpus, phrases = tokenize(corpus)
word_freq = Counter(tokenised_corpus)
subsampled_corpus = subsample_words(tokenised_corpus, word_freq, t)

# Assign IDs to words
word2id, id2word = assign_ids(subsampled_corpus)

# Add <unk> token if it doesn't exist
if "<unk>" not in word2id:
    word2id["<unk>"] = len(word2id)
    id2word[len(id2word)] = "<unk>"
    print("Added <unk> token to vocabulary")

# Print some statistics
print("\nVocabulary Statistics:")
print(f"Total vocabulary size: {len(word2id)}")
print(f"Total phrases detected: {len(phrases)}")

# Print first 10 word-ID pairs
print("\nFirst 10 word-ID pairs:")
for word, id_val in list(word2id.items())[:10]:
    print(f"Word: {word} ID: {id_val}")

# Print top 10 phrases by score
print("\nTop 10 phrases by score:")
sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
for (word1, word2), score in sorted_phrases[:10]:
    combined = f"{word1}_{word2}"
    print(f"Phrase: {combined}, Score: {score:.4f}")

# Save to JSON file
output_file = save_to_json(word2id, phrases)
print(f"\nTokenizer data successfully saved to {output_file}")

# Additional info for using the saved file
print("\nTo load this file in your regressor script:")
print("""
import json

# Load the tokenizer data
with open('tokenizer_data.json', 'r', encoding='utf-8') as f:
    tokenizer_data = json.load(f)

# Extract the word2id mapping
word2id = tokenizer_data['word2id']
vocab_size = tokenizer_data['vocab_size']

# Now you can use this word2id with your skip-gram model
""")




