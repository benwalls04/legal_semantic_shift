import json
import re
from typing import List, Iterator


def clean_text(text: str) -> str:
    """
    Clean text by:
    - Removing extra whitespace
    - Normalizing unicode characters
    - Removing special formatting characters
    """
    if not text:
        return ""
    
    text = re.sub(r'[\f\r]', ' ', text)
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words.
    - Converts to lowercase
    - Splits on whitespace and punctuation
    - Filters out empty tokens
    """
    if not text:
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Split on whitespace and punctuation, keeping alphanumeric sequences
    # This pattern matches words (alphanumeric sequences) and splits on punctuation
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    
    # Filter out very short tokens (less than 2 characters) and numbers only
    tokens = [token for token in tokens if len(token) >= 2 and not token.isdigit()]
    
    return tokens


def load_documents(jsonl_path: str) -> Iterator[dict]:
    """
    Load documents from a JSONL file.
    Yields dictionaries with document data.
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                yield doc
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue


def tokenize_documents(jsonl_path: str) -> List[List[str]]:
    """
    Load documents from JSONL file, extract plain_text, clean and tokenize.
    Returns a list of tokenized documents (each document is a list of tokens).
    """
    tokenized_docs = []
    
    for doc in load_documents(jsonl_path):
        plain_text = doc.get("plain_text", "")
        
        # Skip if empty
        if not plain_text or not plain_text.strip():
            continue
        
        # Clean the text
        cleaned_text = clean_text(plain_text)
        
        # Skip if cleaning resulted in empty text
        if not cleaned_text:
            continue
        
        # Tokenize
        tokens = tokenize_text(cleaned_text)
        
        # Only add documents with at least some tokens
        if len(tokens) > 0:
            tokenized_docs.append(tokens)
    
    return tokenized_docs


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tokenize.py <jsonl_file>")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    tokenized = tokenize_documents(jsonl_path)
    print(f"Tokenized {len(tokenized)} documents")
    print(f"First document has {len(tokenized[0]) if tokenized else 0} tokens")
    if tokenized:
        print(f"Sample tokens: {tokenized[0][:20]}")

