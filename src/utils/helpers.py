from scipy.linalg import orthogonal_procrustes
import numpy as np
import os
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import euclidean_distances
import sys


def align_embeddings(early_embs: np.ndarray, later_embs: np.ndarray):
    if early_embs.shape != later_embs.shape:
        raise ValueError(f"Embedding matrices must have same shape. Got {early_embs.shape} and {later_embs.shape}")
    
    R, scale = orthogonal_procrustes(early_embs, later_embs)
    early_embs_aligned = early_embs @ R
    
    return R, early_embs_aligned, later_embs


def cosine_similarity_single(emb1: np.ndarray, emb2: np.ndarray) -> float:
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cosine_sim = np.dot(emb1, emb2) / (norm1 * norm2)
    
    return float(cosine_sim)


def cosine_similarity_list(early_embs_aligned: np.ndarray, later_embs: np.ndarray,
                           word_to_idx_before: dict, word_to_idx_after: dict, 
                           words: list) -> list:
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

def load_models(model_before_path: str, model_after_path: str):
    """Load both Word2Vec or SVD models."""
    import pickle
    from gensim.models import Word2Vec
    
    model_before_path = os.path.abspath(model_before_path)
    model_after_path = os.path.abspath(model_after_path)
    
    print(f"Loading model (before): {model_before_path}")
    print(f"  File exists: {os.path.exists(model_before_path)}")
    
    # Try to detect model type
    try:
        model_before = Word2Vec.load(model_before_path)
        print("  Model type: Word2Vec")
    except:
        # Try SVD model
        try:
            # Import from the correct path
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from train_svd import SVDModel
            model_before = SVDModel.load(model_before_path)
            print("  Model type: SVD")
        except Exception as e:
            raise ValueError(f"Could not load model from {model_before_path}: {e}")
    
    print(f"Loading model (after): {model_after_path}")
    print(f"  File exists: {os.path.exists(model_after_path)}")
    
    try:
        model_after = Word2Vec.load(model_after_path)
        print("  Model type: Word2Vec")
    except:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from train_svd import SVDModel
            model_after = SVDModel.load(model_after_path)
            print("  Model type: SVD")
        except Exception as e:
            raise ValueError(f"Could not load model from {model_after_path}: {e}")
    
    if model_before_path == model_after_path:
        raise ValueError(f"ERROR: Both models point to the same file: {model_before_path}")
    
    print(f"\nModel comparison:")
    print(f"  Before vocab size: {len(model_before.wv)}")
    print(f"  After vocab size: {len(model_after.wv)}")
    
    return model_before, model_after

def get_word_embedding(model, word: str):
    """Get embedding for a word, return None if word not in vocabulary."""
    try:
        return model.wv[word]
    except KeyError:
        return None


def extract_all_embeddings(model):
    """Extract all word embeddings from a model as a numpy matrix."""
    vocab = list(model.wv.index_to_key)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    embeddings = np.array([model.wv[word] for word in vocab])
    return embeddings, word_to_idx


def calculate_semantic_shift(early_embs_aligned: np.ndarray, later_embs: np.ndarray,
                            word_to_idx_before: dict, word_to_idx_after: dict, words: list,
                            model_before=None, model_after=None):
    """Calculate semantic shift metrics for a list of words using aligned embeddings."""
    results = {}
    
    for word in words:
        if word not in word_to_idx_before or word not in word_to_idx_after:
            print(f"Warning: '{word}' not found in one or both models. Skipping.")
            continue
        
        # Get word frequencies if models are provided
        if model_before is not None and model_after is not None:
            try:
                freq_before = model_before.wv.get_vecattr(word, 'count')
            except (KeyError, AttributeError):
                freq_before = 0
            try:
                freq_after = model_after.wv.get_vecattr(word, 'count')
            except (KeyError, AttributeError):
                freq_after = 0
            print(f"{word}: {freq_before} (before) vs {freq_after} (after)")
        
        idx_before = word_to_idx_before[word]
        idx_after = word_to_idx_after[word]
        
        emb_before = early_embs_aligned[idx_before]
        emb_after = later_embs[idx_after]
        
        # calculate semantic shift metrics 
        cosine_sim = cosine_similarity_single(emb_before, emb_after)
        euclidean_dist = euclidean_distances([emb_before], [emb_after])[0][0]
        shift_magnitude = 1 - cosine_sim 
        
        results[word] = {
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
            'shift_magnitude': shift_magnitude,
            'embedding_before': emb_before,
            'embedding_after': emb_after
        }
    
    return results


def print_shift_summary(results: dict):
    """Print a summary table of semantic shift results."""
    print("\n" + "="*70)
    print("SEMANTIC SHIFT SUMMARY")
    print("="*70)
    print(f"{'Word':<20} {'Cosine Sim':<15} {'Shift Magnitude':<15} {'Euclidean Dist':<15}")
    print("-"*70)
    
    for word, metrics in results.items():
        print(f"{word:<20} {metrics['cosine_similarity']:<15.4f} {metrics['shift_magnitude']:<15.4f} {metrics['euclidean_distance']:<15.4f}")
    
    print("="*70)
    print("\nInterpretation:")
    print("- Cosine Similarity: Range -1 to 1. Higher = more similar embeddings.")
    print("- Shift Magnitude: Range 0 to 2. Higher = more semantic shift.")
    print("- Euclidean Distance: Lower = embeddings are closer in vector space.")