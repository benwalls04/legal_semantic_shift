# Legal Semantic Shift Analysis

This project analyzes semantic shift in legal language over time by comparing word embeddings from different decades of court opinions. The pipeline fetches data from CourtListener, trains word embedding models (Word2Vec or SVD), and visualizes semantic shifts for predefined control and shift words.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setting Up API Access](#setting-up-api-access)
- [Running the Pipeline](#running-the-pipeline)
- [Understanding the Output](#understanding-the-output)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.8 or higher
- A CourtListener API token (free account available at [CourtListener](https://www.courtlistener.com/api/))
- Internet connection (for downloading data)

## Installation

### Step 2: Create a Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install all required packages from the dependencies file:

```bash
pip install -r dependencies.txt
```

The dependencies include:
- `requests` - For API calls to CourtListener
- `python-dotenv` - For loading environment variables
- `gensim` - For Word2Vec model training
- `matplotlib` - For visualization
- `scikit-learn` - For SVD model training
- `scipy` - For scientific computations
- `numpy` - For numerical operations

## Running the Pipeline

### Basic Usage

The pipeline script (`run_pipeline.py`) automates the entire process from data fetching to visualization. Run it with the following command:

```bash
python run_pipeline.py <before_decade> <after_decade> <model_type> <model_size>
```

### Parameters

- **`before_decade`**: The earlier decade to compare (e.g., `1800`, `1890`, `2000`)
- **`after_decade`**: The later decade to compare (e.g., `2010`, `2000`)
- **`model_type`**: Either `word2vec` or `svd`
- **`model_size`**: Dataset size - `small` (200 opinions), `medium` (1,000 opinions), or `large` (50,000 opinions)

### Reproduce Results

```bash
python run_pipeline.py 2000 2010 svd small
```

### What the Pipeline Does

The pipeline automatically performs these steps:

1. **Creates Cluster Mapping** (if needed)
   - Generates `data/decade_to_clusters.json` mapping decades to CourtListener cluster IDs

2. **Fetches Data** (if needed)
   - Downloads court opinions from CourtListener API for both decades
   - Saves data to `data/inputs/opinions-{decade}-{size}.jsonl`
   - Skips if data files already exist

3. **Trains Models** (if needed)
   - Trains Word2Vec or SVD models for both decades
   - Saves models to `src/models/{model_type}-{decade}-{size}.model`
   - Skips if model files already exist

4. **Generates Visualizations**
   - Calculates semantic shift for predefined words
   - Creates graphs showing normalized cosine similarities
   - Displays results in the terminal

## Understanding the Output

### Terminal Output

The pipeline prints:
- Progress updates for each step
- Summary statistics about data collection
- Semantic shift metrics for each word
- Average cosine similarities for control vs. shift words

### Visualization Graphs

The pipeline generates a visualization showing two side-by-side bar charts:

1. **Left Chart - Control Words** (Expected: High Similarity)
   - Words: `court`, `law`, `evidence`, `trial`
   - These words should have minimal semantic shift over time
   - Higher normalized cosine similarity (closer to 1.0) indicates stability

2. **Right Chart - Semantic Shift Words** (Expected: Lower Similarity)
   - Words: `rights`, `discrimination`, `power`, `property`
   - These words are expected to show semantic shift over time
   - Lower normalized cosine similarity indicates more change

The graphs show:
- Individual bar for each word with its normalized cosine similarity (0-1 scale)
- Average line (dashed) showing the mean similarity for each group
- Color coding: Green for control words, Red for shift words

The visualization is displayed in a matplotlib window. Close the window to continue or complete the pipeline.

### Output Files

After running the pipeline, you'll find:

- **Data files**: `data/inputs/opinions-{decade}-{size}.jsonl`
- **Trained models**: `src/models/{model_type}-{decade}-{size}.model`
- **Cluster mapping**: `data/decade_to_clusters.json`

## Project Structure

```
legal_semantic_shift/
├── run_pipeline.py          # Main pipeline script
├── dependencies.txt          # Python package dependencies
├── .env                      # API token (create this file)
├── data/
│   ├── inputs/              # Downloaded court opinions (JSONL format)
│   └── decade_to_clusters.json  # Mapping of decades to cluster IDs
├── src/
│   ├── models/              # Trained embedding models
│   ├── preprocess/
│   │   ├── scrape_courtlistener.py  # Data fetching script
│   │   └── tokenize.py              # Text tokenization
│   ├── train_word2vec.py    # Word2Vec model training
│   ├── train_svd.py         # SVD model training
│   ├── analysis/
│   │   ├── visualize.py     # Main visualization entry point
│   │   ├── predictions.py   # Predefined words visualization
│   │   └── analyze.py       # Legal terms analysis
│   └── utils/
│       └── helpers.py        # Utility functions
└── venv/                    # Virtual environment (created during setup)
```

- The pipeline is idempotent: it skips steps if output files already exist
- To re-run from scratch, delete the relevant data files or model files
- First run will take longer as it downloads data and trains models
- Subsequent runs with the same parameters will be much faster
