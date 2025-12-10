#!/usr/bin/env python3
"""
Full model pipeline script for semantic shift analysis.

Usage:
    python run_pipeline.py <before_decade> <after_decade> <model_type> <model_size>
    
Example:
    python run_pipeline.py 1800 2010 word2vec small
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def print_header(before_decade, after_decade, model_type, model_size):
    """Print pipeline header information."""
    print("=" * 50)
    print("Model Pipeline Script")
    print("=" * 50)
    print(f"Before decade: {before_decade}")
    print(f"After decade: {after_decade}")
    print(f"Model type: {model_type}")
    print(f"Model size: {model_size}")
    print("=" * 50)
    print()


def run_command(cmd, description=None):
    """Run a command and handle errors."""
    if description:
        print(description)
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Command not found: {cmd[0]}")
        sys.exit(1)


def check_and_create_mapping(mapping_file):
    """Check if cluster mapping exists, create if not."""
    if not os.path.exists(mapping_file):
        print("Cluster mapping file not found. Creating it...")
        cmd = [
            sys.executable,
            "src/preprocess/scrape_courtlistener.py",
            "--map-decades"
        ]
        run_command(cmd)
        print()
    else:
        print("Cluster mapping file exists. Skipping creation.")
        print()


def fetch_data_if_needed(decade, model_size):
    """Check and fetch data for a decade if needed."""
    data_file = f"data/inputs/opinions-{decade}-{model_size}.jsonl"
    
    if not os.path.exists(data_file):
        print(f"Data file for decade {decade} not found. Fetching...")
        cmd = [
            sys.executable,
            "src/preprocess/scrape_courtlistener.py",
            "--decade", str(decade),
            "--size", model_size
        ]
        run_command(cmd)
        print()
    else:
        print(f"Data file for decade {decade} exists. Skipping fetch.")
        print()


def train_model_if_needed(decade, model_type, model_size):
    """Check and train model for a decade if needed."""
    model_file = f"src/models/{model_type}-{decade}-{model_size}.model"
    
    if not os.path.exists(model_file):
        print(f"Model for decade {decade} not found. Training...")
        if model_type == "word2vec":
            cmd = [
                sys.executable,
                "src/train_word2vec.py",
                "--decade", str(decade),
                "--size", model_size
            ]
        else:  # svd
            cmd = [
                sys.executable,
                "src/train_svd.py",
                "--decade", str(decade),
                "--size", model_size
            ]
        run_command(cmd)
        print()
    else:
        print(f"Model for decade {decade} exists. Skipping training.")
        print()


def run_predictions(before_decade, after_decade, model_type, model_size):
    """Run predictions using visualize.py."""
    print("Running predictions...")
    cmd = [
        sys.executable,
        "src/analysis/visualize.py",
        "--mode", "predefined",
        "--decade-before", str(before_decade),
        "--decade-after", str(after_decade),
        "--model-type", model_type,
        "--size", model_size
    ]
    run_command(cmd)


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Full model pipeline for semantic shift analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py 1800 2010 word2vec small
  python run_pipeline.py 1890 2000 svd medium
        """
    )
    
    parser.add_argument(
        "before_decade",
        type=int,
        help="Earlier decade (e.g., 1800)"
    )
    parser.add_argument(
        "after_decade",
        type=int,
        help="Later decade (e.g., 2010)"
    )
    parser.add_argument(
        "model_type",
        choices=["word2vec", "svd"],
        help="Model type: 'word2vec' or 'svd'"
    )
    parser.add_argument(
        "model_size",
        choices=["small", "medium", "large"],
        help="Dataset size: 'small', 'medium', or 'large'"
    )
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # Print header
    print_header(
        args.before_decade,
        args.after_decade,
        args.model_type,
        args.model_size
    )
    
    # Step 1: Check if cluster mapping exists, create if not
    mapping_file = "data/decade_to_clusters.json"
    check_and_create_mapping(mapping_file)
    
    # Step 2: Fetch data for both decades if needed
    print("Checking data files...")
    fetch_data_if_needed(args.before_decade, args.model_size)
    fetch_data_if_needed(args.after_decade, args.model_size)
    
    # Step 3: Train models for both decades if needed
    print("Checking models...")
    train_model_if_needed(args.before_decade, args.model_type, args.model_size)
    train_model_if_needed(args.after_decade, args.model_type, args.model_size)
    
    # Step 4: Run predictions
    run_predictions(
        args.before_decade,
        args.after_decade,
        args.model_type,
        args.model_size
    )
    
    print()
    print("=" * 50)
    print("Pipeline completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()

