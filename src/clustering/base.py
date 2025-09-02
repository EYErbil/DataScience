"""
Base class for all clustering algorithms.
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class BaseClusterer(ABC):
    """Abstract base class for clustering algorithms."""
    
    def __init__(self, similarity_threshold: float = 0.4, min_cluster_size: int = 2):
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.is_fitted = False
        self.labels_ = None
        self.n_clusters_ = 0
        
    @abstractmethod
    def fit(self, embeddings: np.ndarray, texts: List[str] = None) -> 'BaseClusterer':
        """Fit the clusterer on embeddings."""
        pass
    
    @abstractmethod
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new embeddings."""
        pass
    
    def fit_predict(self, embeddings: np.ndarray, texts: List[str] = None) -> np.ndarray:
        """Fit the clusterer and return cluster labels."""
        self.fit(embeddings, texts)
        return self.labels_
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the clustering results."""
        if not self.is_fitted:
            return {}
            
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        
        return {
            'n_clusters': self.n_clusters_,
            'n_noise_points': np.sum(self.labels_ == -1),
            'largest_cluster_size': np.max(counts),
            'smallest_cluster_size': np.min(counts[counts > 0]) if len(counts) > 0 else 0,
            'average_cluster_size': np.mean(counts[counts > 0]) if len(counts) > 0 else 0,
            'cluster_sizes': dict(zip(unique_labels, counts))
        }
    
    def get_cluster_samples(self, texts: List[str], samples_per_cluster: int = 5) -> Dict[int, List[str]]:
        """Get sample texts for each cluster."""
        if not self.is_fitted or not texts:
            return {}
            
        cluster_samples = {}
        
        for cluster_id in np.unique(self.labels_):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_mask = self.labels_ == cluster_id
            cluster_texts = [texts[i] for i in np.where(cluster_mask)[0]]
            
            # Sample up to N texts per cluster
            if len(cluster_texts) > samples_per_cluster:
                import random
                cluster_samples[cluster_id] = random.sample(cluster_texts, samples_per_cluster)
            else:
                cluster_samples[cluster_id] = cluster_texts
                
        return cluster_samples
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"similarity_threshold={self.similarity_threshold}, "
                f"min_cluster_size={self.min_cluster_size}, "
                f"fitted={self.is_fitted})")
    
    @staticmethod
    def compute_silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score for clustering evaluation."""
        try:
            from sklearn.metrics import silhouette_score
            # Filter out noise points (-1 labels)
            mask = labels != -1
            if np.sum(mask) < 2:
                return 0.0
            return silhouette_score(embeddings[mask], labels[mask])
        except ImportError:
            logger.warning("sklearn not available for silhouette score")
            return 0.0
        except Exception as e:
            logger.warning(f"Error computing silhouette score: {e}")
            return 0.0
