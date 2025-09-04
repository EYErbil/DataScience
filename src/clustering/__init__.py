"""
Clustering package for scalable semantic grouping.
Supports FAISS-based clustering for millions of items.
"""
from .faiss_clusterer import FaissClusterer
from .enhanced_faiss_clusterer import EnhancedFaissClusterer
from .hdbscan_clusterer import HdbscanClusterer
from .base import BaseClusterer

__all__ = ['FaissClusterer', 'EnhancedFaissClusterer', 'HdbscanClusterer', 'BaseClusterer']
