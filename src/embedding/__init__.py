"""
Embedding package for semantic vector generation.
Supports both transformer-based and TF-IDF approaches.
"""
from .hf_encoder import HuggingFaceEncoder
from .tfidf_encoder import TfidfEncoder
from .base import BaseEncoder

__all__ = ['HuggingFaceEncoder', 'TfidfEncoder', 'BaseEncoder']
