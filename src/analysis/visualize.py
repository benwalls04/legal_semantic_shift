"""
Entry point for semantic shift visualization and analysis.
Routes to either predictions (predefined words) or analyze (legal terms) modules.
"""
import os
import sys
import argparse

# Add analysis directory to path
analysis_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(analysis_dir))

from predictions import visualize_predefined_words
from analyze import analyze_legal_terms_shift


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize semantic shift between two Word2Vec models")
    parser.add_argument("--mode", type=str, choices=["predefined", "analyze"], default="predefined",
                       help="Mode: 'predefined' uses predefined control/shift words, 'analyze' analyzes all legal terms")
    parser.add_argument("--size", type=str, choices=["small", "medium", "large"], 
                       default="small",
                       help="Dataset size: small (200), medium (1k), large (50k). Default: small")
    parser.add_argument("--decade-before", type=int, required=True,
                       help="Earlier decade (e.g., 1980)")
    parser.add_argument("--decade-after", type=int, required=True,
                       help="Later decade (e.g., 2010)")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save visualization (optional, only for predefined mode)")
    parser.add_argument("--top-n", type=int, default=5,
                       help="Number of top words to show (only for analyze mode, default: 5)")
    parser.add_argument("--model-type", type=str, default="word2vec", choices=["word2vec", "svd"])
    
    args = parser.parse_args()

    # Get the src directory (parent of analysis directory)
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(src_dir, "models")
    
    model_before_path = os.path.join(models_dir, f"{args.model_type}-{args.decade_before}-{args.size}.model")
    model_after_path = os.path.join(models_dir, f"{args.model_type}-{args.decade_after}-{args.size}.model")

    if not os.path.exists(model_before_path):
        print(f"Error: model (before) path does not exist -> {model_before_path}")
        exit(1)

    if not os.path.exists(model_after_path):
        print(f"Error: model (after) path does not exist -> {model_after_path}")
        exit(1)
    
    if args.mode == "predefined":
        visualize_predefined_words(
            model_before_path=model_before_path,
            model_after_path=model_after_path,
            decade_before=args.decade_before,
            decade_after=args.decade_after,
            output_path=args.output
        )
    elif args.mode == "analyze":
        analyze_legal_terms_shift(
            model_before_path=model_before_path,
            model_after_path=model_after_path,
            decade_before=args.decade_before,
            decade_after=args.decade_after,
            top_n=args.top_n
        )
