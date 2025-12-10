"""
Analysis module for finding words with most and least semantic shift from a large list of legal terms.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add utils directory to path to import helpers
utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, os.path.abspath(utils_dir))
from helpers import (
    load_models, extract_all_embeddings, align_embeddings, cosine_similarity_single
)


LEGAL_TERMS = [
    "court", "judge", "law", "legal", "justice", "trial", "evidence", "jury", "verdict",
    "plaintiff", "defendant", "attorney", "lawyer", "counsel", "prosecutor", "defense",
    "appeal", "motion", "hearing", "testimony", "witness", "deposition", "subpoena",
    "indictment", "arraignment", "sentencing", "conviction", "acquittal", "plea",
    "constitution", "amendment", "statute", "regulation", "precedent", "ruling", "opinion",
    "brief", "petition", "complaint", "lawsuit", "litigation", "settlement", "damages",
    "rights", "freedom", "liberty", "equality", "privacy", "discrimination", "civil",
    "criminal", "felony", "misdemeanor", "violation", "offense", "crime",
    "technology", "digital", "internet", "cyber", "data", "surveillance", "monitoring",
    "intellectual", "property", "copyright", "patent", "trademark", "licensing",
    "contract", "agreement", "breach", "liability", "negligence", "tort", "compensation",
    "corporation", "partnership", "merger", "acquisition", "bankruptcy", "insolvency",
    "federal", "state", "jurisdiction", "sovereignty", "federalism",
    "separation", "powers", "executive", "legislative", "judicial", "branch",
    "murder", "assault", "robbery", "theft", "fraud", "embezzlement", "bribery",
    "conspiracy", "racketeering", "organized", "terrorism", "homicide",
    "divorce", "custody", "alimony", "inheritance", "estate", "will", "trust",
    "landlord", "tenant", "lease", "property", "real", "zoning",
    "regulation", "agency", "administrative", "compliance", "enforcement", "penalty",
    "fine", "sanction", "license", "permit", "inspection", "audit",
    "treaty", "international", "diplomatic", "extradition", "asylum",
    "refugee", "immigration", "citizenship", "naturalization", "deportation",
    "environment", "pollution", "conservation", "endangered", "species", "habitat",
    "public", "health", "safety", "welfare", "benefit", "interest"
]


def analyze_legal_terms_shift(model_before_path: str, model_after_path: str,
                              decade_before: int = None, decade_after: int = None,
                              top_n: int = 5, plot_chart: bool = True, chart_output_path: str = None):

    print(f"\n{'='*60}")
    print(f"Analyzing {len(LEGAL_TERMS)} legal terms for semantic shift")
    if decade_before and decade_after:
        print(f"Comparing {decade_before} vs {decade_after}")
    print(f"{'='*60}\n")
    
    model_before, model_after = load_models(model_before_path, model_after_path)
    
    # Extract embeddings and align
    print("\nExtracting embeddings...")
    early_embs, word_to_idx_before = extract_all_embeddings(model_before)
    later_embs, word_to_idx_after = extract_all_embeddings(model_after)
    
    # Check if embedding dimensions match
    if early_embs.shape[1] != later_embs.shape[1]:
        raise ValueError(f"Embedding dimensions must match. Got {early_embs.shape[1]} and {later_embs.shape[1]}")
    
    # Find common vocabulary for alignment
    common_words = set(word_to_idx_before.keys()) & set(word_to_idx_after.keys())
    print(f"Common vocabulary size: {len(common_words)}")


    # Extract embeddings for common words only 
    common_words_list = sorted(list(common_words))
    early_embs_common = np.array([early_embs[word_to_idx_before[word]] for word in common_words_list])
    later_embs_common = np.array([later_embs[word_to_idx_after[word]] for word in common_words_list])
    
    # align embeddings 
    R, early_embs_common_aligned, _ = align_embeddings(early_embs_common, later_embs_common)
    early_embs_aligned = early_embs @ R
    
    # Filter legal terms to only those present in both models
    valid_terms = []
    for term in LEGAL_TERMS:
        if term in word_to_idx_before and term in word_to_idx_after:
            valid_terms.append(term)
    
    print(f"\nFound {len(valid_terms)} valid terms out of {len(LEGAL_TERMS)} total terms")
    
    if len(valid_terms) < top_n * 2:
        print(f"Error: Not enough valid terms ({len(valid_terms)}) to find top {top_n} for each category.")
        return None
    
    print("\nCalculating semantic shift for all terms...")
    shift_results = []
    
    for term in valid_terms:
        idx_before = word_to_idx_before[term]
        idx_after = word_to_idx_after[term]
        
        emb_before = early_embs_aligned[idx_before]
        emb_after = later_embs[idx_after]
        
        cosine_sim = cosine_similarity_single(emb_before, emb_after)
        shift_magnitude = 1 - cosine_sim
        
        shift_results.append({
            'word': term,
            'cosine_similarity': cosine_sim,
            'shift_magnitude': shift_magnitude
        })
    
    shift_results_sorted = sorted(shift_results, key=lambda x: x['cosine_similarity'], reverse=True)
    
    # Get top N with most and least shift 
    least_shift = shift_results_sorted[:top_n]
    most_shift = shift_results_sorted[-top_n:]
    most_shift.reverse()  
    
    print("Generating bar chart...")
    plot_cosine_similarities_bar_chart(
        shift_results=shift_results,
        decade_before=decade_before,
        decade_after=decade_after,
        output_path=chart_output_path
    )
    
    return {
        'least_shift': [(r['word'], r['cosine_similarity'], r['shift_magnitude']) for r in least_shift],
        'most_shift': [(r['word'], r['cosine_similarity'], r['shift_magnitude']) for r in most_shift],
        'all_results': shift_results
    }


def plot_cosine_similarities_bar_chart(shift_results, decade_before=None, decade_after=None, 
                                       output_path=None, figsize=(14, 8)):
    """
    Generate a bar chart showing all words and their cosine similarities in sorted order.
    
    Args:
        shift_results: List of dictionaries with 'word', 'cosine_similarity', and 'shift_magnitude'
        decade_before: Earlier decade (for title)
        decade_after: Later decade (for title)
        output_path: Optional path to save the figure
        figsize: Figure size tuple (width, height)
    """
    # Sort by cosine similarity (highest to lowest)
    sorted_results = sorted(shift_results, key=lambda x: x['cosine_similarity'], reverse=True)
    
    # Extract words and cosine similarities
    words = [r['word'] for r in sorted_results]
    cosine_sims = [r['cosine_similarity'] for r in sorted_results]
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    bars = ax.barh(range(len(words)), cosine_sims, color='steelblue', alpha=0.7)
    
    # Set y-axis labels to words
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=8)
    
    # Set x-axis label
    ax.set_xlabel('Cosine Similarity', fontsize=12, fontweight='bold')
    
    # Set title
    title = 'Cosine Similarities for All Words'
    if decade_before and decade_after:
        title += f' ({decade_before} vs {decade_after})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Invert y-axis so highest similarity is at top
    ax.invert_yaxis()
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, sim) in enumerate(zip(bars, cosine_sims)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{sim:.3f}', ha='left', va='center', fontsize=7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nBar chart saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    return fig, ax

