import os
import sys
import argparse
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

# Add src directory to path to import tokenize
src_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(src_dir))
from preprocess.tokenize import tokenize_documents


class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training"""
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        print(f"Epoch #{self.epoch} end")
        self.epoch += 1


def train_word2vec(jsonl_path: str, output_path: str, 
                   vector_size: int = 100, 
                   window: int = 5,
                   min_count: int = 2,
                   workers: int = 4,
                   epochs: int = 10):
    """
    Train a Word2Vec model on tokenized legal documents.
    
    Args:
        jsonl_path: Path to JSONL file with legal opinions
        output_path: Path to save the trained model
        vector_size: Dimensionality of word vectors
        window: Maximum distance between current and predicted word
        min_count: Minimum word count to be included in vocabulary
        workers: Number of worker threads
        epochs: Number of training epochs
    """

    print(f"Loading and tokenizing documents from {jsonl_path}...")
    sentences = tokenize_documents(jsonl_path)
    
    if not sentences:
        print("Error: No documents were tokenized. Check your input file.")
        return
    
    print(f"Tokenized {len(sentences)} documents")
    total_tokens = sum(len(s) for s in sentences)
    print(f"Total tokens: {total_tokens:,}")
    
    # Initialize callback
    epoch_logger = EpochLogger()
    
    print("\nTraining Word2Vec model...")
    print(f"Parameters: vector_size={vector_size}, window={window}, "
          f"min_count={min_count}, workers={workers}, epochs={epochs}")
    
    # Train the model
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        callbacks=[epoch_logger]
    )
    
    # Save the model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    
    print(f"\nModel saved to {output_path}")
    print(f"Vocabulary size: {len(model.wv)}")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Word2Vec model on legal documents")
    parser.add_argument("--decade", type=int, required=True,
                       help="Decade to query (e.g., 1900, 1910, 1920)")
    parser.add_argument("--size", type=str, choices=["small", "medium", "large"], 
                       default="small",
                       help="Dataset size: small (200), medium (1k), large (50k). Default: small")
    parser.add_argument("--vector-size", type=int, default=100,
                       help="Dimensionality of word vectors (default: 100)")
    parser.add_argument("--window", type=int, default=5,
                       help="Maximum distance between current and predicted word (default: 5)")
    parser.add_argument("--min-count", type=int, default=2,
                       help="Minimum word count to be included (default: 2)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of worker threads (default: 4)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs (default: 10)")
    
    args = parser.parse_args()

    input_path = f"../data/inputs/opinions-{args.decade}-{args.size}.jsonl"
    output_path = f"models/word2vec-{args.decade}-{args.size}.model"

    if not os.path.exists(input_path):
        print(f"Error: input path does not exist -> {input_path}")
        exit(1)
    
    train_word2vec(
        jsonl_path=input_path,
        output_path=output_path,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs
    )

