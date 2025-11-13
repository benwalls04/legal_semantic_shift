# visualize the shift in word embeddings
import os
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import argparse

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
    print(f"Loading model (before): {model_before_path}")
    model_before = Word2Vec.load(model_before_path)
    
    print(f"Loading model (after): {model_after_path}")
    model_after = Word2Vec.load(model_after_path)
    
    return model_before, model_after


def get_word_embedding(model, word: str):
    """Get embedding for a word, return None if word not in vocabulary."""
    try:
        return model.wv[word]
    except KeyError:
        return None


def calculate_semantic_shift(model_before, model_after, words: list):
    """
    Calculate semantic shift metrics for a list of words.
    Returns dictionary with cosine similarity, euclidean distance, and embeddings.
    """
    results = {}
    
    for word in words:
        emb_before = get_word_embedding(model_before, word)
        emb_after = get_word_embedding(model_after, word)
        
        if emb_before is None or emb_after is None:
            print(f"Warning: '{word}' not found in one or both models. Skipping.")
            continue
        
        # Calculate cosine similarity (higher = more similar, range: -1 to 1)
        cosine_sim = cosine_similarity([emb_before], [emb_after])[0][0]
        
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


def plot_semantic_shift_comparison(results: dict, decade_before: int = None, decade_after: int = None, output_path: str = None):
    """
    Create visualizations comparing semantic shift between words.
    """
    if not results:
        print("No results to visualize.")
        return
    
    words = list(results.keys())
    cosine_sims = [results[w]['cosine_similarity'] for w in words]
    shift_magnitudes = [results[w]['shift_magnitude'] for w in words]
    
    # Create title with decade information if provided
    if decade_before and decade_after:
        title = f'Semantic Shift Analysis: {decade_before} vs {decade_after}'
    else:
        title = 'Semantic Shift Analysis'
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Cosine Similarity (bar chart)
    ax1 = axes[0, 0]
    colors = plt.cm.RdYlGn(1 - np.array(shift_magnitudes) / 2)  # Red for high shift, green for low
    bars1 = ax1.barh(words, cosine_sims, color=colors)
    ax1.set_xlabel('Cosine Similarity', fontsize=11)
    ax1.set_title('Cosine Similarity (Higher = More Similar)', fontsize=12)
    ax1.set_xlim([0, 1])
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Low Similarity Threshold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, cosine_sims)):
        ax1.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)
    
    # Plot 2: Shift Magnitude (bar chart)
    ax2 = axes[0, 1]
    bars2 = ax2.barh(words, shift_magnitudes, color=colors)
    ax2.set_xlabel('Shift Magnitude', fontsize=11)
    ax2.set_title('Semantic Shift Magnitude (Higher = More Shift)', fontsize=12)
    ax2.set_xlim([0, 2])
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, shift_magnitudes)):
        ax2.text(val + 0.05, i, f'{val:.3f}', va='center', fontsize=9)
    
    # Plot 3: Scatter plot - Cosine Similarity vs Shift Magnitude
    ax3 = axes[1, 0]
    scatter = ax3.scatter(cosine_sims, shift_magnitudes, s=100, alpha=0.6, c=range(len(words)), cmap='viridis')
    for i, word in enumerate(words):
        ax3.annotate(word, (cosine_sims[i], shift_magnitudes[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax3.set_xlabel('Cosine Similarity', fontsize=11)
    ax3.set_ylabel('Shift Magnitude', fontsize=11)
    ax3.set_title('Similarity vs Shift', fontsize=12)
    ax3.grid(alpha=0.3)
    
    # Plot 4: 2D Projection using PCA
    ax4 = axes[1, 1]
    
    # Combine all embeddings for PCA
    all_embeddings = []
    labels = []
    before_label = f"Before ({decade_before})" if decade_before else "Before"
    after_label = f"After ({decade_after})" if decade_after else "After"
    
    for word in words:
        all_embeddings.append(results[word]['embedding_before'])
        labels.append(f"{word}_before")
        all_embeddings.append(results[word]['embedding_after'])
        labels.append(f"{word}_after")
    
    all_embeddings = np.array(all_embeddings)
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    # Plot with arrows showing shift
    colors_list = plt.cm.tab10(np.linspace(0, 1, len(words)))
    for i, word in enumerate(words):
        idx_before = i * 2
        idx_after = i * 2 + 1
        
        # Plot points
        ax4.scatter(embeddings_2d[idx_before, 0], embeddings_2d[idx_before, 1], 
                   color=colors_list[i], marker='o', s=100, alpha=0.7, label=f'{word} ({before_label})' if i == 0 else '')
        ax4.scatter(embeddings_2d[idx_after, 0], embeddings_2d[idx_after, 1], 
                   color=colors_list[i], marker='s', s=100, alpha=0.7, label=f'{word} ({after_label})' if i == 0 else '')
        
        # Draw arrow from before to after
        ax4.annotate('', xy=embeddings_2d[idx_after], xytext=embeddings_2d[idx_before],
                    arrowprops=dict(arrowstyle='->', color=colors_list[i], lw=2, alpha=0.6))
        
        # Add word labels
        ax4.text(embeddings_2d[idx_before, 0], embeddings_2d[idx_before, 1] - 0.1, 
                word, fontsize=8, ha='center', color=colors_list[i])
    
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
    ax4.set_title('2D PCA Projection of Embeddings', fontsize=12)
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualization saved to {output_path}")
    else:
        plt.show()


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
    
    # Calculate semantic shift
    print(f"\nCalculating semantic shift for {len(words)} words...")
    results = calculate_semantic_shift(model_before, model_after, words)
    
    if not results:
        print("No valid words found in both models.")
        return
    
    # Print summary
    print_shift_summary(results)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_semantic_shift_comparison(results, decade_before, decade_after, output_path)
    
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

    model_before_path = f"models/word2vec-{args.decade_before}-{args.size}.model"
    model_after_path = f"models/word2vec-{args.decade_after}-{args.size}.model"

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