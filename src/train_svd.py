import os
import sys
import argparse
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle

# Add src directory to path to import tokenize
src_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(src_dir))
from preprocess.tokenize import tokenize_documents


def compute_ppmi(cooccurrence_matrix, alpha=0.0):
    """
    Compute Positive Pointwise Mutual Information (PPMI) from co-occurrence matrix.
    
    Args:
        cooccurrence_matrix: Sparse or dense co-occurrence matrix
        alpha: Smoothing parameter (default: 0.0 as recommended for SVD)
    
    Returns:
        PPMI matrix (same format as input)
    """
    # Convert to dense if sparse for easier computation
    if isinstance(cooccurrence_matrix, csr_matrix):
        cooccurrence = cooccurrence_matrix.toarray()
    else:
        cooccurrence = cooccurrence_matrix.copy()
    
    # Add smoothing
    cooccurrence = cooccurrence + alpha
    
    # Compute marginal probabilities
    total = cooccurrence.sum()
    if total == 0:
        return cooccurrence_matrix
    
    # Row and column sums (word and context word frequencies)
    row_sums = cooccurrence.sum(axis=1, keepdims=True)
    col_sums = cooccurrence.sum(axis=0, keepdims=True)
    
    # Avoid division by zero
    row_sums = np.where(row_sums == 0, 1, row_sums)
    col_sums = np.where(col_sums == 0, 1, col_sums)
    
    # Compute PMI: log(P(w,c) / (P(w) * P(c)))
    # P(w,c) = cooccurrence / total
    # P(w) = row_sums / total
    # P(c) = col_sums / total
    pmi = np.log((cooccurrence / total) / ((row_sums / total) * (col_sums / total)) + 1e-10)
    
    # PPMI: set negative values to 0
    ppmi = np.maximum(pmi, 0.0)
    
    # Convert back to sparse if input was sparse
    if isinstance(cooccurrence_matrix, csr_matrix):
        return csr_matrix(ppmi)
    return ppmi


def apply_context_distribution_smoothing(cooccurrence_matrix, smoothing=0.75):
    """
    Apply context distribution smoothing to co-occurrence matrix.
    
    Args:
        cooccurrence_matrix: Co-occurrence matrix
        smoothing: Smoothing parameter (default: 0.75)
    
    Returns:
        Smoothed co-occurrence matrix
    """
    if isinstance(cooccurrence_matrix, csr_matrix):
        cooccurrence = cooccurrence_matrix.toarray()
    else:
        cooccurrence = cooccurrence_matrix.copy()
    
    # Normalize rows (context distributions)
    row_sums = cooccurrence.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    
    # Apply smoothing: (1 - smoothing) * original + smoothing * uniform
    vocab_size = cooccurrence.shape[1]
    uniform_dist = np.ones((1, vocab_size)) / vocab_size
    
    smoothed = (1 - smoothing) * (cooccurrence / row_sums) + smoothing * uniform_dist
    
    # Scale back to original magnitude
    smoothed = smoothed * row_sums
    
    if isinstance(cooccurrence_matrix, csr_matrix):
        return csr_matrix(smoothed)
    return smoothed


class DummyVectorizer:
    """Dummy vectorizer for compatibility when using term-term co-occurrence matrices."""
    def __init__(self, vocab):
        self.vocab = vocab
    
    def get_feature_names_out(self):
        return np.array(self.vocab)


class SVDModel:
    """
    Wrapper class to make SVD embeddings compatible with Word2Vec-like interface
    for use with existing analysis code.
    """
    def __init__(self, svd_model, vectorizer, vocab, embeddings):
        self.svd_model = svd_model
        self.vectorizer = vectorizer
        self.vocab = vocab  # list of words in order
        self.embeddings = embeddings  # numpy array (vocab_size, embedding_dim)
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    class WordVectors:
        """Mock wv attribute to match Word2Vec interface"""
        def __init__(self, parent_model):
            self.parent = parent_model
        
        def __getitem__(self, word):
            if word not in self.parent.word_to_idx:
                raise KeyError(f"'{word}' not in vocabulary")
            idx = self.parent.word_to_idx[word]
            return self.parent.embeddings[idx]
        
        def __len__(self):
            return len(self.parent.vocab)
        
        @property
        def index_to_key(self):
            return self.parent.vocab
    
    @property
    def wv(self):
        return self.WordVectors(self)
    
    def save(self, path):
        """Save the SVD model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'svd_model': self.svd_model,
                'vectorizer': self.vectorizer,
                'vocab': self.vocab,
                'embeddings': self.embeddings
            }, f)
    
    @classmethod
    def load(cls, path):
        """Load the SVD model from disk"""
        # Register DummyVectorizer in __main__ so pickle can find it
        import sys
        if '__main__' in sys.modules:
            sys.modules['__main__'].DummyVectorizer = DummyVectorizer
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        return cls(
            svd_model=data['svd_model'],
            vectorizer=data['vectorizer'],
            vocab=data['vocab'],
            embeddings=data['embeddings']
        )


def train_svd(jsonl_path: str, output_path: str,
              n_components: int = 100,
              min_count: int = 5,
              max_features: int = None,
              use_tfidf: bool = True,
              matrix_type: str = 'term_doc',
              context_window: int = 4,
              ppmi_alpha: float = 0.0,
              context_smoothing: float = 0.75,
              eigenvalue_weight: float = 0.5):
    """
    Train SVD embeddings on tokenized legal documents.
    
    Args:
        jsonl_path: Path to JSONL file with legal opinions
        output_path: Path to save the trained model
        n_components: Dimensionality of word vectors (number of SVD components), recommended: 100-150
        min_count: Minimum word count to be included in vocabulary, recommended: 5-10
        max_features: Maximum vocabulary size (None = no limit)
        use_tfidf: If True, use TF-IDF weighting; if False, use raw counts
        matrix_type: 'term_doc' (term-document matrix) or 'term_term' (co-occurrence)
        context_window: Context window size (symmetric), recommended: 4
        ppmi_alpha: PPMI smoothing parameter (α), recommended: 0.0
        context_smoothing: Context distribution smoothing, recommended: 0.75
        eigenvalue_weight: Eigenvalue weighting (γ), recommended: 0.5-0.75
    """
    
    print(f"Loading and tokenizing documents from {jsonl_path}...")
    tokenized_docs = tokenize_documents(jsonl_path)
    
    if not tokenized_docs:
        print("Error: No documents were tokenized. Check your input file.")
        return
    
    print(f"Tokenized {len(tokenized_docs)} documents")
    total_tokens = sum(len(doc) for doc in tokenized_docs)
    print(f"Total tokens: {total_tokens:,}")
    
    # Convert tokenized documents to space-separated strings for vectorizer
    print("\nConverting documents to strings...")
    doc_strings = [' '.join(doc) for doc in tokenized_docs]
    
    if matrix_type == 'term_doc':
        # Build term-document matrix
        print(f"\nBuilding {'TF-IDF' if use_tfidf else 'count'} term-document matrix...")
        
        if use_tfidf:
            vectorizer = TfidfVectorizer(
                min_df=min_count,
                max_features=max_features,
                lowercase=False,  # Already lowercase from tokenization
                token_pattern=r'\S+',  # Match any non-whitespace (already tokenized)
                analyzer='word'
            )
        else:
            vectorizer = CountVectorizer(
                min_df=min_count,
                max_features=max_features,
                lowercase=False,
                token_pattern=r'\S+',
                analyzer='word'
            )
        
        # Fit and transform
        matrix = vectorizer.fit_transform(doc_strings)
        vocab = vectorizer.get_feature_names_out().tolist()
        
    elif matrix_type == 'term_term':
        # Build term-term co-occurrence matrix
        print(f"\nBuilding term-term co-occurrence matrix...")
        print("(This may take longer for large corpora)")
        
        # First, build vocabulary
        from collections import Counter
        word_counts = Counter()
        for doc in tokenized_docs:
            word_counts.update(doc)
        
        # Filter by min_count
        vocab = [word for word, count in word_counts.items() if count >= min_count]
        if max_features:
            vocab = sorted(vocab, key=lambda w: word_counts[w], reverse=True)[:max_features]
        vocab = sorted(vocab)  # Sort alphabetically for consistency
        
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        vocab_size = len(vocab)
        
        # Build co-occurrence matrix (window-based, symmetric)
        print(f"Building co-occurrence matrix with symmetric window size {context_window}...")
        cooccurrence = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        
        for doc in tokenized_docs:
            for i, word in enumerate(doc):
                if word not in word_to_idx:
                    continue
                word_idx = word_to_idx[word]
                
                # Look at words in symmetric window
                start = max(0, i - context_window)
                end = min(len(doc), i + context_window + 1)
                
                for j in range(start, end):
                    if i == j:
                        continue
                    context_word = doc[j]
                    if context_word in word_to_idx:
                        context_idx = word_to_idx[context_word]
                        # Distance-weighted co-occurrence
                        distance = abs(i - j)
                        weight = 1.0 / distance
                        cooccurrence[word_idx, context_idx] += weight
        
        # Convert to sparse matrix
        matrix = csr_matrix(cooccurrence)
        
        # Apply context distribution smoothing
        if context_smoothing > 0:
            print(f"Applying context distribution smoothing (smoothing={context_smoothing})...")
            matrix = apply_context_distribution_smoothing(matrix, smoothing=context_smoothing)
        
        # Apply PPMI transformation
        if ppmi_alpha >= 0:
            print(f"Computing PPMI with smoothing α={ppmi_alpha}...")
            matrix = compute_ppmi(matrix, alpha=ppmi_alpha)
        
        # Create a dummy vectorizer for compatibility
        vectorizer = DummyVectorizer(vocab)
        
    else:
        raise ValueError(f"Unknown matrix_type: {matrix_type}. Use 'term_doc' or 'term_term'")
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Apply SVD
    print(f"\nApplying TruncatedSVD with {n_components} components...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    
    if matrix_type == 'term_doc':
        # For term-document: transpose to get word embeddings
        # SVD on term-doc matrix: U @ S @ V^T
        # U gives document embeddings, V^T gives term embeddings
        # We want term embeddings, so we transpose first
        word_embeddings = svd.fit_transform(matrix.T)
    else:
        # For term-term: embeddings are directly from SVD
        word_embeddings = svd.fit_transform(matrix)
    
    # Apply eigenvalue weighting (γ)
    if eigenvalue_weight != 1.0:
        print(f"Applying eigenvalue weighting (γ={eigenvalue_weight})...")
        # Get singular values and weight them
        singular_values = svd.singular_values_
        # Weight the singular values: S^γ
        weighted_singular_values = np.power(singular_values, eigenvalue_weight)
        # Apply weighting to embeddings: U * S^γ (instead of U * S)
        # Since word_embeddings = U @ S, we need to scale by S^(γ-1)
        scaling = np.power(singular_values, eigenvalue_weight - 1.0)
        word_embeddings = word_embeddings * scaling[np.newaxis, :]
    
    print(f"Embedding shape: {word_embeddings.shape}")
    print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
    
    # Create model wrapper
    model = SVDModel(
        svd_model=svd,
        vectorizer=vectorizer,
        vocab=vocab,
        embeddings=word_embeddings
    )
    
    # Save the model
    model.save(output_path)
    
    print(f"\nModel saved to {output_path}")
    print(f"Vocabulary size: {len(model.vocab)}")
    print(f"Embedding dimension: {word_embeddings.shape[1]}")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SVD/LSA model on legal documents")
    parser.add_argument("--decade", type=int, required=True,
                       help="Decade to query (e.g., 1900, 1910, 1920)")
    parser.add_argument("--size", type=str, choices=["small", "medium", "large"], 
                       default="small",
                       help="Dataset size: small (200), medium (1k), large (50k). Default: small")
    parser.add_argument("--n-components", type=int, default=100,
                       help="Dimensionality of word vectors, recommended: 100-150 (default: 100)")
    parser.add_argument("--min-count", type=int, default=5,
                       help="Minimum word count to be included, recommended: 5-10 (default: 5)")
    parser.add_argument("--max-features", type=int, default=None,
                       help="Maximum vocabulary size (default: None = no limit)")
    parser.add_argument("--no-tfidf", action="store_true",
                       help="Use raw counts instead of TF-IDF (default: use TF-IDF)")
    parser.add_argument("--matrix-type", type=str, choices=["term_doc", "term_term"],
                       default="term_term",
                       help="Type of matrix: term_doc (faster) or term_term (co-occurrence, recommended)")
    parser.add_argument("--context-window", type=int, default=4,
                       help="Context window size (symmetric), recommended: 4 (default: 4)")
    parser.add_argument("--ppmi-alpha", type=float, default=0.0,
                       help="PPMI smoothing parameter (α), recommended: 0.0 (default: 0.0)")
    parser.add_argument("--context-smoothing", type=float, default=0.75,
                       help="Context distribution smoothing, recommended: 0.75 (default: 0.75)")
    parser.add_argument("--eigenvalue-weight", type=float, default=0.5,
                       help="Eigenvalue weighting (γ), recommended: 0.5-0.75 (default: 0.5)")
    
    args = parser.parse_args()

    input_path = f"../data/inputs/opinions-{args.decade}-{args.size}.jsonl"
    output_path = f"models/svd-{args.decade}-{args.size}.model"

    if not os.path.exists(input_path):
        print(f"Error: input path does not exist -> {input_path}")
        exit(1)
    
    train_svd(
        jsonl_path=input_path,
        output_path=output_path,
        n_components=args.n_components,
        min_count=args.min_count,
        max_features=args.max_features,
        use_tfidf=not args.no_tfidf,
        matrix_type=args.matrix_type,
        context_window=args.context_window,
        ppmi_alpha=args.ppmi_alpha,
        context_smoothing=args.context_smoothing,
        eigenvalue_weight=args.eigenvalue_weight
    )