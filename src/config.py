"""
Central configuration for the product categorization pipeline.
All hyperparameters and settings in one place.
"""
import os
from pathlib import Path
from typing import List, Dict, Any

# ═══════════════════════════════════════════════════════════════
# PATHS AND DIRECTORIES
# ═══════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Ensure artifacts directory exists
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# PROCESSING SETTINGS
# ═══════════════════════════════════════════════════════════════
# Chunk size for streaming large CSV files
CHUNK_SIZE = 100_000

# Maximum rows to process (None = no limit)
MAX_ROWS = None

# Number of CPU cores to use (None = auto-detect)
N_JOBS = None

# ═══════════════════════════════════════════════════════════════
# EMBEDDING SETTINGS
# ═══════════════════════════════════════════════════════════════
# Primary embedding model - upgraded to richer model (falls back to TF-IDF if download fails)
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # State-of-the-art multilingual embeddings
EMBEDDING_FALLBACK = "sentence-transformers/all-mpnet-base-v2"  # Stronger fallback than MiniLM

# Batch size for embedding generation
EMBEDDING_BATCH_SIZE = 512

# TF-IDF settings (fallback)
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_DF = 0.7
TFIDF_MIN_DF = 2

# ═══════════════════════════════════════════════════════════════
# CLUSTERING SETTINGS (Enhanced)
# ═══════════════════════════════════════════════════════════════
# Clustering algorithm: 'faiss_kmeans', 'hdbscan', 'agglomerative'
CLUSTERING_METHOD = "faiss_kmeans"

# Similarity threshold for clustering (0.0 - 1.0) - tighter for better quality
SIMILARITY_THRESHOLD = 0.6

# Minimum cluster size - adaptive based on dataset size
MIN_CLUSTER_SIZE = 3

# Maximum number of clusters (auto if None)
MAX_CLUSTERS = None

# Advanced clustering parameters
CLUSTERING_ITERATIONS = 100  # More iterations for better convergence
CLUSTERING_INIT_METHOD = "kmeans++"  # Better initialization
USE_HIERARCHICAL_REFINEMENT = True  # Post-process with hierarchical clustering
DENSITY_THRESHOLD = 0.05  # For density-based noise filtering

# ═══════════════════════════════════════════════════════════════
# CATEGORY ASSIGNMENT SETTINGS
# ═══════════════════════════════════════════════════════════════
# Minimum confidence threshold for category assignment
CATEGORY_CONFIDENCE_THRESHOLD = 0.3

# Number of top words to extract per cluster for category inference
TOP_WORDS_PER_CLUSTER = 10

# Number of representative samples to show per cluster
CLUSTER_SAMPLES = 5

# ═══════════════════════════════════════════════════════════════
# TEXT NORMALIZATION SETTINGS
# ═══════════════════════════════════════════════════════════════
# Remove these common words from analysis
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'for', 'of', 'with', 'in', 'on', 'at', 'to', 'from',
    'small', 'large', 'big', 'little', 'new', 'old', 'black', 'white', 'blue', 'red',
    'green', 'brown', 'gray', 'grey', 'professional', 'standard', 'premium', 'basic',
    'size', 'color', 'model', 'type', 'brand', 'item', 'product', 'piece'
}

# Minimum word length to consider
MIN_WORD_LENGTH = 3

# ═══════════════════════════════════════════════════════════════
# EVALUATION SETTINGS
# ═══════════════════════════════════════════════════════════════
# Sample size for human evaluation
EVAL_SAMPLE_SIZE = 100

# Random seed for reproducibility
RANDOM_SEED = 42

# ═══════════════════════════════════════════════════════════════
# SSL AND NETWORK SETTINGS
# ═══════════════════════════════════════════════════════════════
# SSL bypass for corporate networks
SSL_BYPASS = True

# Retry settings for model downloads
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT = 30

# ═══════════════════════════════════════════════════════════════
# PERFORMANCE SETTINGS
# ═══════════════════════════════════════════════════════════════
# Enable GPU if available
USE_GPU = False  # Set to False for CPU-only systems

# Memory optimization
LOW_MEMORY_MODE = False

# Cache embeddings to disk
CACHE_EMBEDDINGS = True

# ═══════════════════════════════════════════════════════════════
# OUTPUT SETTINGS
# ═══════════════════════════════════════════════════════════════
# Output format: 'feather', 'parquet', 'csv'
OUTPUT_FORMAT = "feather"

# Compression for output files
COMPRESSION = "lz4"

# Include confidence scores in output
INCLUDE_CONFIDENCE = True

# ═══════════════════════════════════════════════════════════════
# LOGGING SETTINGS
# ═══════════════════════════════════════════════════════════════
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary."""
    config_vars = {
        name: value for name, value in globals().items()
        if not name.startswith('_') and name.isupper()
    }
    return config_vars

def override_config(**kwargs):
    """Override configuration values at runtime."""
    globals().update(kwargs)

def get_artifact_path(name: str, extension: str = None) -> Path:
    """Get path for an artifact file."""
    if extension is None:
        extension = OUTPUT_FORMAT
    filename = f"{name}.{extension}"
    return ARTIFACTS_DIR / filename
