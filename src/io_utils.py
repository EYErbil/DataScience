"""
I/O utilities for efficient data loading and caching.
Handles feather/parquet serialization with compression.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Any
import logging
import time
from contextlib import contextmanager

# Handle both relative and absolute imports
try:
    from .config import ARTIFACTS_DIR, OUTPUT_FORMAT, COMPRESSION, get_artifact_path
except ImportError:
    from config import ARTIFACTS_DIR, OUTPUT_FORMAT, COMPRESSION, get_artifact_path

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# CACHING UTILITIES
# ═══════════════════════════════════════════════════════════════

def cache_exists(name: str, extension: str = None) -> bool:
    """Check if a cached artifact exists."""
    path = get_artifact_path(name, extension)
    return path.exists()

def save_dataframe(df: pd.DataFrame, name: str, extension: str = None) -> Path:
    """Save DataFrame to cache with optimal format."""
    path = get_artifact_path(name, extension or OUTPUT_FORMAT)
    
    logger.info(f"💾 Saving {len(df):,} rows to {path}")
    
    start_time = time.time()
    
    if path.suffix == '.feather':
        df.to_feather(path, compression=COMPRESSION)
    elif path.suffix == '.parquet':
        df.to_parquet(path, compression=COMPRESSION, index=False)
    elif path.suffix == '.csv':
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    
    elapsed = time.time() - start_time
    file_size = path.stat().st_size / (1024 * 1024)  # MB
    
    logger.info(f"✅ Saved {file_size:.1f}MB in {elapsed:.1f}s")
    return path

def load_dataframe(name: str, extension: str = None) -> pd.DataFrame:
    """Load DataFrame from cache."""
    path = get_artifact_path(name, extension or OUTPUT_FORMAT)
    
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")
    
    logger.info(f"📂 Loading {path}")
    
    start_time = time.time()
    
    if path.suffix == '.feather':
        df = pd.read_feather(path)
    elif path.suffix == '.parquet':
        df = pd.read_parquet(path)
    elif path.suffix == '.csv':
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Loaded {len(df):,} rows in {elapsed:.1f}s")
    
    return df

def save_embeddings(embeddings: np.ndarray, name: str) -> Path:
    """Save embeddings array to disk."""
    path = get_artifact_path(name, 'npy')
    
    logger.info(f"💾 Saving embeddings {embeddings.shape} to {path}")
    
    start_time = time.time()
    np.save(path, embeddings)
    elapsed = time.time() - start_time
    
    file_size = path.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"✅ Saved {file_size:.1f}MB in {elapsed:.1f}s")
    
    return path

def load_embeddings(name: str) -> np.ndarray:
    """Load embeddings array from disk."""
    path = get_artifact_path(name, 'npy')
    
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    
    logger.info(f"📂 Loading embeddings from {path}")
    
    start_time = time.time()
    embeddings = np.load(path)
    elapsed = time.time() - start_time
    
    logger.info(f"✅ Loaded embeddings {embeddings.shape} in {elapsed:.1f}s")
    return embeddings

# ═══════════════════════════════════════════════════════════════
# STREAMING UTILITIES
# ═══════════════════════════════════════════════════════════════

def stream_csv(file_path: Union[str, Path], chunk_size: int = 100_000, max_rows: Optional[int] = None):
    """Stream CSV file in chunks for memory-efficient processing."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    logger.info(f"📡 Streaming CSV: {file_path}")
    
    total_rows = 0
    chunk_count = 0
    
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk_count += 1
            chunk_rows = len(chunk)
            total_rows += chunk_rows
            
            logger.debug(f"Processing chunk {chunk_count}: {chunk_rows:,} rows")
            
            yield chunk
            
            if max_rows and total_rows >= max_rows:
                logger.info(f"Reached max_rows limit: {max_rows:,}")
                break
    
    except Exception as e:
        logger.error(f"Error streaming CSV: {e}")
        raise
    
    logger.info(f"✅ Streamed {total_rows:,} total rows in {chunk_count} chunks")

# ═══════════════════════════════════════════════════════════════
# SMART CACHING
# ═══════════════════════════════════════════════════════════════

@contextmanager
def smart_cache(cache_name: str, force_rebuild: bool = False):
    """
    Context manager for smart caching.
    
    Usage:
        with smart_cache("embeddings") as cache:
            if cache.exists and not force_rebuild:
                embeddings = cache.load()
            else:
                embeddings = expensive_computation()
                cache.save(embeddings)
    """
    class CacheHandler:
        def __init__(self, name: str):
            self.name = name
            self.exists = cache_exists(name)
            self._data = None
        
        def load(self):
            if cache_exists(self.name):
                try:
                    return load_dataframe(self.name)
                except:
                    # Try loading as embeddings
                    return load_embeddings(self.name)
            return None
        
        def save(self, data):
            if isinstance(data, pd.DataFrame):
                return save_dataframe(data, self.name)
            elif isinstance(data, np.ndarray):
                return save_embeddings(data, self.name)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
    
    cache_handler = CacheHandler(cache_name)
    cache_handler.exists = cache_handler.exists and not force_rebuild
    
    yield cache_handler

# ═══════════════════════════════════════════════════════════════
# METADATA UTILITIES
# ═══════════════════════════════════════════════════════════════

def save_metadata(metadata: dict, name: str) -> Path:
    """Save metadata as JSON."""
    import json
    
    path = get_artifact_path(name, 'json')
    
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"💾 Saved metadata to {path}")
    return path

def load_metadata(name: str) -> dict:
    """Load metadata from JSON."""
    import json
    
    path = get_artifact_path(name, 'json')
    
    if not path.exists():
        return {}
    
    with open(path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"📂 Loaded metadata from {path}")
    return metadata

# ═══════════════════════════════════════════════════════════════
# CLEANUP UTILITIES
# ═══════════════════════════════════════════════════════════════

def clear_cache(pattern: str = "*") -> int:
    """Clear cached artifacts matching pattern."""
    import glob
    
    cache_files = list(ARTIFACTS_DIR.glob(pattern))
    
    for file_path in cache_files:
        file_path.unlink()
        logger.info(f"🗑️ Deleted {file_path}")
    
    logger.info(f"✅ Cleared {len(cache_files)} cache files")
    return len(cache_files)

def get_cache_info() -> dict:
    """Get information about cached files."""
    cache_files = list(ARTIFACTS_DIR.glob("*"))
    
    info = {
        "total_files": len(cache_files),
        "total_size_mb": sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
        "files": []
    }
    
    for file_path in sorted(cache_files):
        file_info = {
            "name": file_path.name,
            "size_mb": file_path.stat().st_size / (1024 * 1024),
            "modified": file_path.stat().st_mtime
        }
        info["files"].append(file_info)
    
    return info
