from scipy.linalg import orthogonal_procrustes
import numpy as np


def align_embeddings(early_embs: np.ndarray, later_embs: np.ndarray):
    """
    Align two embedding spaces using Procrustes orthogonal transformation.
    
    Args:
        early_embs: numpy array of shape (vocab_size, embedding_dim) for earlier decade
        later_embs: numpy array of shape (vocab_size, embedding_dim) for later decade
                   Must have same shape as early_embs
    
    Returns:
        R: rotation matrix of shape (embedding_dim, embedding_dim)
        early_embs_aligned: aligned early embeddings (early_embs @ R)
        later_embs: later embeddings (unchanged, used as target)
    """
    if early_embs.shape != later_embs.shape:
        raise ValueError(f"Embedding matrices must have same shape. Got {early_embs.shape} and {later_embs.shape}")
    
    # Compute Procrustes alignment: find R such that early_embs @ R best matches later_embs
    R, scale = orthogonal_procrustes(early_embs, later_embs)
    
    # Apply rotation to early embeddings
    early_embs_aligned = early_embs @ R
    
    return R, early_embs_aligned, later_embs


def cosine_similarity_single(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embedding vectors.
    
    Args:
        emb1: numpy array of shape (embedding_dim,)
        emb2: numpy array of shape (embedding_dim,)
    
    Returns:
        Cosine similarity value (range: -1 to 1)
    """
    # Normalize vectors
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Cosine similarity = dot product / (norm1 * norm2)
    cosine_sim = np.dot(emb1, emb2) / (norm1 * norm2)
    
    return float(cosine_sim)


def cosine_similarity_list(early_embs_aligned: np.ndarray, later_embs: np.ndarray,
                           word_to_idx_before: dict, word_to_idx_after: dict, 
                           words: list) -> list:
    """
    Calculate cosine similarities for a list of words between two aligned embedding spaces.
    
    Args:
        early_embs_aligned: aligned embeddings from earlier decade (vocab_size, embedding_dim)
        later_embs: embeddings from later decade (vocab_size, embedding_dim)
        word_to_idx_before: mapping from word to index in early_embs_aligned
        word_to_idx_after: mapping from word to index in later_embs
        words: list of words to calculate similarities for
    
    Returns:
        List of cosine similarity values (one per word, in same order as words list)
        Returns None for words not found in either model
    """
    similarities = []
    
    for word in words:
        if word not in word_to_idx_before or word not in word_to_idx_after:
            similarities.append(None)
            continue
        
        idx_before = word_to_idx_before[word]
        idx_after = word_to_idx_after[word]
        
        emb_before = early_embs_aligned[idx_before]
        emb_after = later_embs[idx_after]
        
        cosine_sim = cosine_similarity_single(emb_before, emb_after)
        similarities.append(cosine_sim)
    
    return similarities