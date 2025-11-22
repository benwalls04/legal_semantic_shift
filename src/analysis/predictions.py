"""
Predictions module for visualizing semantic shift of predefined control and shift words.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add utils directory to path to import helpers
utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, os.path.abspath(utils_dir))
from helpers import (
    load_models, extract_all_embeddings, calculate_semantic_shift, 
    print_shift_summary, align_embeddings
)

# Predefined words for semantic shift analysis
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


def visualize_predefined_words(model_before_path: str, model_after_path: str, 
                               decade_before: int = None, decade_after: int = None, 
                               output_path: str = None):
    """
    Visualize semantic shift for predefined control and shift words.
    Uses SEMANTIC_SHIFT_WORDS and CONTROL_WORDS lists.
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
        base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
        normalized_plot_path = f"{base_path}_normalized_cosine.png"
    else:
        normalized_plot_path = None
    
    plot_normalized_cosine_similarities(results, decade_before, decade_after, normalized_plot_path)
    
    return results

