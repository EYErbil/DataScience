"""
Categorisation package implementing ChatGPT's Approach 2 & 4.
- Approach 2: Unsupervised Clustering with Word Embeddings
- Approach 4: Zero-Shot Classification with LLMs
"""
from .cluster_mapper import AutoClusterMapper
from .zero_shot_classifier import ZeroShotClassifier, GPTClassifier, create_hybrid_classifier

__all__ = ['AutoClusterMapper', 'ZeroShotClassifier', 'GPTClassifier', 'create_hybrid_classifier']
