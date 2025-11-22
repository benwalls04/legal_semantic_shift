"""
Analysis module for finding words with most and least semantic shift from a large list of legal terms.
"""
import os
import sys
import numpy as np

# Add utils directory to path to import helpers
utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, os.path.abspath(utils_dir))
from helpers import (
    load_models, extract_all_embeddings, align_embeddings, cosine_similarity_single
)

# Large list of legal terms to analyze
LEGAL_TERMS = [
    # Core legal concepts
    "court", "judge", "law", "legal", "justice", "trial", "evidence", "jury", "verdict",
    "plaintiff", "defendant", "attorney", "lawyer", "counsel", "prosecutor", "defense",
    
    # Legal procedures
    "appeal", "motion", "hearing", "testimony", "witness", "deposition", "subpoena",
    "indictment", "arraignment", "sentencing", "conviction", "acquittal", "plea",
    
    # Legal concepts
    "constitution", "amendment", "statute", "regulation", "precedent", "ruling", "opinion",
    "brief", "petition", "complaint", "lawsuit", "litigation", "settlement", "damages",
    
    # Rights and protections
    "rights", "freedom", "liberty", "equality", "privacy", "discrimination", "civil",
    "criminal", "felony", "misdemeanor", "violation", "offense", "crime",
    
    # Modern legal concepts
    "technology", "digital", "internet", "cyber", "data", "surveillance", "monitoring",
    "intellectual", "property", "copyright", "patent", "trademark", "licensing",
    
    # Business and contract law
    "contract", "agreement", "breach", "liability", "negligence", "tort", "compensation",
    "corporation", "partnership", "merger", "acquisition", "bankruptcy", "insolvency",
    
    # Constitutional law
    "federal", "state", "jurisdiction", "sovereignty", "federalism",
    "separation", "powers", "executive", "legislative", "judicial", "branch",
    
    # Criminal law
    "murder", "assault", "robbery", "theft", "fraud", "embezzlement", "bribery",
    "conspiracy", "racketeering", "organized", "terrorism", "homicide",
    
    # Civil law
    "divorce", "custody", "alimony", "inheritance", "estate", "will", "trust",
    "landlord", "tenant", "lease", "property", "real", "zoning",
    
    # Administrative and regulatory
    "regulation", "agency", "administrative", "compliance", "enforcement", "penalty",
    "fine", "sanction", "license", "permit", "inspection", "audit",
    
    # International law
    "treaty", "international", "diplomatic", "extradition", "asylum",
    "refugee", "immigration", "citizenship", "naturalization", "deportation",
    
    # Environmental and public interest
    "environment", "pollution", "conservation", "endangered", "species", "habitat",
    "public", "health", "safety", "welfare", "benefit", "interest"
]


def analyze_legal_terms_shift(model_before_path: str, model_after_path: str,
                              decade_before: int = None, decade_after: int = None,
                              top_n: int = 5):
    """
    Analyze a large list of legal terms to find the top N words with least and most shift.
    
    Args:
        model_before_path: Path to Word2Vec model from earlier decade
        model_after_path: Path to Word2Vec model from later decade
        decade_before: Earlier decade (for display purposes)
        decade_after: Later decade (for display purposes)
        top_n: Number of top words to return (default: 5)
    
    Returns:
        Dictionary with 'least_shift' and 'most_shift' lists
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {len(LEGAL_TERMS)} legal terms for semantic shift")
    if decade_before and decade_after:
        print(f"Comparing {decade_before} vs {decade_after}")
    print(f"{'='*60}\n")
    
    # Load models
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
    
    if len(common_words) < 10:
        raise ValueError(f"Too few common words ({len(common_words)}) for reliable alignment. Need at least 10.")
    
    # Extract embeddings for common words only (for alignment)
    common_words_list = sorted(list(common_words))
    early_embs_common = np.array([early_embs[word_to_idx_before[word]] for word in common_words_list])
    later_embs_common = np.array([later_embs[word_to_idx_after[word]] for word in common_words_list])
    
    print("Aligning embedding spaces...")
    R, early_embs_common_aligned, _ = align_embeddings(early_embs_common, later_embs_common)
    
    # Apply alignment to entire early embedding space
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
    
    # Calculate semantic shift for all valid terms
    print("\nCalculating semantic shift for all terms...")
    shift_results = []
    
    for term in valid_terms:
        idx_before = word_to_idx_before[term]
        idx_after = word_to_idx_after[term]
        
        emb_before = early_embs_aligned[idx_before]
        emb_after = later_embs[idx_after]
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity_single(emb_before, emb_after)
        
        # Calculate shift magnitude (1 - cosine similarity)
        shift_magnitude = 1 - cosine_sim
        
        shift_results.append({
            'word': term,
            'cosine_similarity': cosine_sim,
            'shift_magnitude': shift_magnitude
        })
    
    # Sort by cosine similarity (highest = least shift)
    shift_results_sorted = sorted(shift_results, key=lambda x: x['cosine_similarity'], reverse=True)
    
    # Get top N with least shift (highest cosine similarity)
    least_shift = shift_results_sorted[:top_n]
    
    # Get top N with most shift (lowest cosine similarity)
    most_shift = shift_results_sorted[-top_n:]
    most_shift.reverse()  # Reverse to show highest shift first
    
    # Print results
    print(f"\n{'='*60}")
    print(f"TOP {top_n} WORDS WITH LEAST SHIFT (Most Stable):")
    print(f"{'='*60}")
    for i, result in enumerate(least_shift, 1):
        print(f"{i}. {result['word']:20} - Cosine Similarity: {result['cosine_similarity']:.4f}, "
              f"Shift Magnitude: {result['shift_magnitude']:.4f}")
    
    print(f"\n{'='*60}")
    print(f"TOP {top_n} WORDS WITH MOST SHIFT (Most Changed):")
    print(f"{'='*60}")
    for i, result in enumerate(most_shift, 1):
        print(f"{i}. {result['word']:20} - Cosine Similarity: {result['cosine_similarity']:.4f}, "
              f"Shift Magnitude: {result['shift_magnitude']:.4f}")
    print(f"{'='*60}\n")
    
    return {
        'least_shift': [(r['word'], r['cosine_similarity'], r['shift_magnitude']) for r in least_shift],
        'most_shift': [(r['word'], r['cosine_similarity'], r['shift_magnitude']) for r in most_shift],
        'all_results': shift_results
    }

