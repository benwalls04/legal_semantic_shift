# visualize the shift in word embeddings
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import argparse

# Add utils directory to path to import helpers
utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, os.path.abspath(utils_dir))
from helpers import align_embeddings, cosine_similarity_single, cosine_similarity_list

SEMANTIC_SHIFT_WORDS = [
    "privacy",      # Privacy laws evolved significantly (digital age)
    "discrimination",  # Legal definitions expanded (protected classes, etc.)
    "technology",   # Legal implications of tech changed dramatically
    "surveillance", # Shifted from physical to digital surveillance
    "data"          # Data protection/privacy became major legal concern
]

# Control words (should have MINIMAL shift)
CONTROL_WORDS = [
    "court",        # Core legal institution, meaning stable
    "judge",        # Role definition unchanged
    "law",          # Fundamental concept, stable meaning
    "evidence",     # Legal concept with stable definition
    "trial"         # Legal procedure, meaning unchanged
]


def load_models(model_before_path: str, model_after_path: str):
    """Load both Word2Vec models."""
    # Convert to absolute paths for clarity
    model_before_path = os.path.abspath(model_before_path)
    model_after_path = os.path.abspath(model_after_path)
    
    print(f"Loading model (before): {model_before_path}")
    print(f"  File exists: {os.path.exists(model_before_path)}")
    model_before = Word2Vec.load(model_before_path)
    
    print(f"Loading model (after): {model_after_path}")
    print(f"  File exists: {os.path.exists(model_after_path)}")
    model_after = Word2Vec.load(model_after_path)
    
    # Verify they're different models
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
    """
    Extract all word embeddings from a Word2Vec model as a numpy matrix.
    
    Args:
        model: Word2Vec model
    
    Returns:
        embeddings: numpy array of shape (vocab_size, embedding_dim)
        word_to_idx: dictionary mapping word to row index in embeddings matrix
    """
    vocab = list(model.wv.index_to_key)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    embeddings = np.array([model.wv[word] for word in vocab])
    return embeddings, word_to_idx


def calculate_semantic_shift(early_embs_aligned: np.ndarray, later_embs: np.ndarray,
                            word_to_idx_before: dict, word_to_idx_after: dict, words: list,
                            model_before=None, model_after=None):
    """
    Calculate semantic shift metrics for a list of words using aligned embeddings.
    
    Args:
        early_embs_aligned: aligned embeddings from earlier decade (vocab_size, embedding_dim)
        later_embs: embeddings from later decade (vocab_size, embedding_dim)
        word_to_idx_before: mapping from word to index in early_embs_aligned
        word_to_idx_after: mapping from word to index in later_embs
        words: list of words to analyze
        model_before: Optional Word2Vec model from earlier decade (for frequency info)
        model_after: Optional Word2Vec model from later decade (for frequency info)
    
    Returns:
        Dictionary with cosine similarity, euclidean distance, and embeddings for each word.
    """
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
        
        # Calculate cosine similarity using helper function (higher = more similar, range: -1 to 1)
        cosine_sim = cosine_similarity_single(emb_before, emb_after)
        
        # Calculate euclidean distance (lower = more similar)
        euclidean_dist = euclidean_distances([emb_before], [emb_after])[0][0]
        
        # Calculate shift magnitude (1 - cosine similarity, normalized to 0-2 range)
        shift_magnitude = 1 - cosine_sim  # Range: 0 (identical) to 2 (opposite)
        
        results[word] = {
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
            'shift_magnitude': shift_magnitude,
            'embedding_before': emb_before,
            'embedding_after': emb_after
        }
    
    return results

def plot_normalized_cosine_similarities(results: dict, decade_before: int = None, 
                                       decade_after: int = None, output_path: str = None):
    """
    Create two separate plots: one for control words and one for shift words,
    showing normalized cosine similarities with average lines.
    """
    if not results:
        print("No results to visualize.")
        return
    
    # Separate words into control and shift groups
    control_words = [w for w in CONTROL_WORDS if w in results]
    shift_words = [w for w in SEMANTIC_SHIFT_WORDS if w in results]
    
    # Normalize cosine similarity from [-1, 1] to [0, 1]
    def normalize_cosine_sim(cos_sim):
        return (cos_sim + 1) / 2
    
    # Extract normalized cosine similarities
    control_sims = [normalize_cosine_sim(results[w]['cosine_similarity']) for w in control_words]
    shift_sims = [normalize_cosine_sim(results[w]['cosine_similarity']) for w in shift_words]
    
    # Calculate averages
    control_avg = np.mean(control_sims) if control_sims else 0
    shift_avg = np.mean(shift_sims) if shift_sims else 0
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create title with decade information
    if decade_before and decade_after:
        fig.suptitle(f'Normalized Cosine Similarities: {decade_before} vs {decade_after}', 
                    fontsize=16, fontweight='bold')
    else:
        fig.suptitle('Normalized Cosine Similarities', fontsize=16, fontweight='bold')
    
    # Plot 1: Control Words
    ax1 = axes[0]
    colors1 = plt.cm.Greens(np.linspace(0.4, 0.9, len(control_words)))
    bars1 = ax1.barh(control_words, control_sims, color=colors1)
    ax1.axvline(x=control_avg, color='red', linestyle='--', linewidth=2, 
                label=f'Average: {control_avg:.3f}')
    ax1.set_xlabel('Normalized Cosine Similarity (0-1)', fontsize=12)
    ax1.set_title('Control Words (Expected: High Similarity)', fontsize=13, fontweight='bold')
    ax1.set_xlim([0, 1])
    ax1.legend(loc='lower right')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, control_sims)):
        ax1.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10)
    
    # Plot 2: Shift Words
    ax2 = axes[1]
    colors2 = plt.cm.Reds(np.linspace(0.4, 0.9, len(shift_words)))
    bars2 = ax2.barh(shift_words, shift_sims, color=colors2)
    ax2.axvline(x=shift_avg, color='blue', linestyle='--', linewidth=2, 
                label=f'Average: {shift_avg:.3f}')
    ax2.set_xlabel('Normalized Cosine Similarity (0-1)', fontsize=12)
    ax2.set_title('Semantic Shift Words (Expected: Lower Similarity)', fontsize=13, fontweight='bold')
    ax2.set_xlim([0, 1])
    ax2.legend(loc='lower right')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, shift_sims)):
        ax2.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Normalized cosine similarity plots saved to {output_path}")
    else:
        plt.show()
    
    # Print summary
    print(f"\nControl words average normalized cosine similarity: {control_avg:.4f}")
    print(f"Shift words average normalized cosine similarity: {shift_avg:.4f}")
    print(f"Difference: {control_avg - shift_avg:.4f}")


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


def visualize_semantic_shift(model_before_path: str, model_after_path: str, 
                             decade_before: int = None, decade_after: int = None, output_path: str = None):
    """
    Main function to visualize semantic shift for a list of words.
    """
    words = SEMANTIC_SHIFT_WORDS + CONTROL_WORDS 

    # Load models
    model_before, model_after = load_models(model_before_path, model_after_path)
    
    # Extract all embeddings from both models
    print("\nExtracting embedding spaces from both models...")
    early_embs, word_to_idx_before = extract_all_embeddings(model_before)
    later_embs, word_to_idx_after = extract_all_embeddings(model_after)
    
    print(f"Early model vocabulary size: {len(word_to_idx_before)}")
    print(f"Later model vocabulary size: {len(word_to_idx_after)}")
    
    # Check if embedding dimensions match
    if early_embs.shape[1] != later_embs.shape[1]:
        raise ValueError(f"Embedding dimensions must match. Got {early_embs.shape[1]} and {later_embs.shape[1]}")
    
    # Find common vocabulary for alignment
    common_words = set(word_to_idx_before.keys()) & set(word_to_idx_after.keys())
    print(f"Common vocabulary size: {len(common_words)}")

    # Instead of all common words, use only high-frequency stable words
    # Filter to top N most frequent words that appear in both decades    

    if len(common_words) < 10:
        raise ValueError(f"Too few common words ({len(common_words)}) for reliable alignment. Need at least 10.")
    
    # Extract embeddings for common words only (for alignment)
    common_words_list = sorted(list(common_words))
    early_embs_common = np.array([early_embs[word_to_idx_before[word]] for word in common_words_list])
    later_embs_common = np.array([later_embs[word_to_idx_after[word]] for word in common_words_list])
    
    # Align embedding spaces using Procrustes
    print("\nAligning embedding spaces using Procrustes transformation...")
    R, early_embs_common_aligned, _ = align_embeddings(early_embs_common, later_embs_common)
    print(f"Alignment rotation matrix shape: {R.shape}")
    
    # Apply alignment to entire early embedding space
    early_embs_aligned = early_embs @ R
    print("Applied alignment transformation to full embedding space.")
    
    # Calculate semantic shift using aligned embeddings
    print(f"\nCalculating semantic shift for {len(words)} words...")
    print("Word frequencies:")
    results = calculate_semantic_shift(
        early_embs_aligned, later_embs,
        word_to_idx_before, word_to_idx_after, words,
        model_before=model_before, model_after=model_after
    )
    
    if not results:
        print("No valid words found in both models.")
        return
    
    # Print summary
    print_shift_summary(results)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Create normalized cosine similarity plots (control vs shift words)
    if output_path:
        # Modify output path for the new plot
        base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
        normalized_plot_path = f"{base_path}_normalized_cosine.png"
    else:
        normalized_plot_path = None
    
    plot_normalized_cosine_similarities(results, decade_before, decade_after, normalized_plot_path)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize semantic shift between two Word2Vec models")
    parser.add_argument("--size", type=str, choices=["small", "medium", "large"], 
                       default="small",
                       help="Dataset size: small (200), medium (1k), large (50k). Default: small")
    parser.add_argument("--decade-before", type=int, required=True,
                       help="Earlier decade (e.g., 1980)")
    parser.add_argument("--decade-after", type=int, required=True,
                       help="Later decade (e.g., 2010)")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save visualization (optional)")
    
    args = parser.parse_args()

    model_before_path = f"../models/word2vec-{args.decade_before}-{args.size}.model"
    model_after_path = f"../models/word2vec-{args.decade_after}-{args.size}.model"

    if not os.path.exists(model_before_path):
        print(f"Error: model (before) path does not exist -> {model_before_path}")
        exit(1)

    if not os.path.exists(model_after_path):
        print(f"Error: model (after) path does not exist -> {model_after_path}")
        exit(1)
    
    visualize_semantic_shift(
        model_before_path=model_before_path,
        model_after_path=model_after_path,
        decade_before=args.decade_before,
        decade_after=args.decade_after,
        output_path=args.output
    )