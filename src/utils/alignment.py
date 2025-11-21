"""
Alignment utilities for comparing Word2Vec embeddings across different time periods.
Uses Procrustes orthogonal transformation to align embedding spaces.
"""
from gensim.models import Word2Vec
import numpy as np
from scipy.linalg import orthogonal_procrustes


def get_common_vocabulary(model1: Word2Vec, model2: Word2Vec, min_freq: int = None) -> list:
    """
    Extract words that exist in both models.
    
    Args:
        model1: First Word2Vec model
        model2: Second Word2Vec model
        min_freq: Optional minimum frequency filter (not used if None)
    
    Returns:
        List of words present in both models' vocabularies
    """
    vocab1 = set(model1.wv.index_to_key)
    vocab2 = set(model2.wv.index_to_key)
    common_words = sorted(list(vocab1 & vocab2))
    
    # If min_freq is specified, filter by frequency
    if min_freq is not None:
        common_words = [w for w in common_words 
                       if model1.wv.get_vecattr(w, 'count') >= min_freq 
                       and model2.wv.get_vecattr(w, 'count') >= min_freq]
    
    return common_words


def extract_embeddings_matrix(model: Word2Vec, words: list) -> np.ndarray:
    """
    Get embeddings for a list of words as a numpy array.
    
    Args:
        model: Word2Vec model
        words: List of words to extract embeddings for
    
    Returns:
        numpy array of shape (len(words), embedding_dim) with embeddings
    """
    embeddings = []
    for word in words:
        if word in model.wv:
            embeddings.append(model.wv[word])
        else:
            raise ValueError(f"Word '{word}' not found in model vocabulary")
    
    return np.array(embeddings)


def align_embeddings_models(model_before: Word2Vec, model_after: Word2Vec, 
                            common_words: list = None, min_freq: int = 5) -> tuple:
    """
    Compute Procrustes alignment between two Word2Vec models.
    
    Args:
        model_before: Word2Vec model from earlier time period
        model_after: Word2Vec model from later time period
        common_words: Optional list of words to use for alignment. If None, uses all words with min_freq
        min_freq: Minimum word frequency to include in alignment (used if common_words is None)
    
    Returns:
        R: rotation matrix of shape (embedding_dim, embedding_dim)
        common_words_used: list of words actually used for alignment
        alignment_mse: mean squared error after alignment (quality metric)
    """
    # Get common vocabulary if not provided
    if common_words is None:
        common_words = get_common_vocabulary(model_before, model_after, min_freq=min_freq)
    
    if len(common_words) < 10:
        raise ValueError(f"Too few common words ({len(common_words)}) for reliable alignment. Need at least 10.")
    
    # Extract embedding matrices for common words
    early_embs = extract_embeddings_matrix(model_before, common_words)
    later_embs = extract_embeddings_matrix(model_after, common_words)
    
    # Check embedding dimensions match
    if early_embs.shape[1] != later_embs.shape[1]:
        raise ValueError(f"Embedding dimensions must match. Got {early_embs.shape[1]} and {later_embs.shape[1]}")
    
    # Compute Procrustes alignment
    R, scale = orthogonal_procrustes(early_embs, later_embs)
    
    # Apply rotation to early embeddings
    early_embs_aligned = early_embs @ R
    
    # Calculate alignment quality (mean squared error)
    mse = np.mean((early_embs_aligned - later_embs) ** 2)
    
    return R, common_words, mse


def apply_alignment(model: Word2Vec, R: np.ndarray) -> np.ndarray:
    """
    Apply rotation matrix to all embeddings in a model.
    
    Args:
        model: Word2Vec model
        R: rotation matrix of shape (embedding_dim, embedding_dim)
    
    Returns:
        numpy array of shape (vocab_size, embedding_dim) with aligned embeddings
    """
    vocab = list(model.wv.index_to_key)
    embeddings = np.array([model.wv[word] for word in vocab])
    
    # Apply rotation
    aligned_embeddings = embeddings @ R
    
    return aligned_embeddings


