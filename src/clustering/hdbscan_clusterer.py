"""
HDBSCAN-based clustering for density-based grouping.
Good for discovering natural clusters without specifying K.
"""
import numpy as np
from typing import List, Optional, Dict, Any
import logging

# Handle both relative and absolute imports
try:
    from .base import BaseClusterer
    from ..config import MIN_CLUSTER_SIZE
except ImportError:
    from clustering.base import BaseClusterer
    from config import MIN_CLUSTER_SIZE

logger = logging.getLogger(__name__)

class HdbscanClusterer(BaseClusterer):
    """HDBSCAN clustering for automatic cluster discovery."""
    
    def __init__(self, 
                 min_cluster_size: int = None,
                 min_samples: int = None,
                 cluster_selection_epsilon: float = 0.0):
        super().__init__(
            similarity_threshold=0.0,  # Not used in HDBSCAN
            min_cluster_size=min_cluster_size or MIN_CLUSTER_SIZE
        )
        self.min_samples = min_samples or self.min_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.clusterer = None
        
    def _install_hdbscan(self):
        """Check and suggest HDBSCAN installation."""
        try:
            import hdbscan
            return hdbscan
        except ImportError:
            logger.error("HDBSCAN not installed. Install with: pip install hdbscan")
            raise ImportError("HDBSCAN is required for HdbscanClusterer")
    
    def fit(self, embeddings: np.ndarray, texts: List[str] = None) -> 'HdbscanClusterer':
        """Fit HDBSCAN clustering on embeddings."""
        hdbscan = self._install_hdbscan()
        
        if embeddings.shape[0] < self.min_cluster_size:
            logger.warning(f"Too few samples ({embeddings.shape[0]}) for clustering")
            self.labels_ = np.zeros(embeddings.shape[0], dtype=int)
            self.n_clusters_ = 1
            self.is_fitted = True
            return self
        
        # Ensure embeddings are float64 for HDBSCAN
        embeddings = embeddings.astype(np.float64)
        
        n_samples, dim = embeddings.shape
        
        logger.info(f"🎯 HDBSCAN clustering: {n_samples:,} samples, "
                   f"min_cluster_size={self.min_cluster_size}")
        
        try:
            # Create HDBSCAN clusterer
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric='cosine',  # Use cosine distance for semantic similarity
                cluster_selection_method='eom',  # Excess of Mass
                allow_single_cluster=True
            )
            
            # Fit and predict
            self.labels_ = self.clusterer.fit_predict(embeddings)
            
            # Count clusters (excluding noise points labeled as -1)
            unique_labels = np.unique(self.labels_)
            self.n_clusters_ = len(unique_labels[unique_labels >= 0])
            
            self.is_fitted = True
            
            n_noise = np.sum(self.labels_ == -1)
            logger.info(f"✅ HDBSCAN complete: {self.n_clusters_} clusters, "
                       f"{n_noise} noise points")
            
            return self
            
        except Exception as e:
            logger.error(f"HDBSCAN clustering failed: {e}")
            return self._fallback_clustering(embeddings)
    
    def _fallback_clustering(self, embeddings: np.ndarray) -> 'HdbscanClusterer':
        """Fallback to agglomerative clustering if HDBSCAN fails."""
        logger.warning("Using agglomerative clustering as fallback")
        
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Compute cosine similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Convert to distance matrix
            distance_matrix = 1 - similarity_matrix
            
            # Use agglomerative clustering with distance threshold
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - self.similarity_threshold,
                linkage='average',
                metric='precomputed'
            )
            
            self.labels_ = clustering.fit_predict(distance_matrix)
            
            # Filter small clusters
            self._filter_small_clusters()
            
            self.n_clusters_ = len(np.unique(self.labels_[self.labels_ >= 0]))
            self.is_fitted = True
            
            logger.info(f"✅ Fallback clustering complete: {self.n_clusters_} clusters")
            return self
            
        except Exception as e:
            logger.error(f"Fallback clustering failed: {e}")
            # Ultimate fallback: no clustering
            self.labels_ = np.arange(embeddings.shape[0])
            self.n_clusters_ = embeddings.shape[0]
            self.is_fitted = True
            return self
    
    def _filter_small_clusters(self):
        """Filter out clusters smaller than min_cluster_size."""
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        
        for label, count in zip(unique_labels, counts):
            if label >= 0 and count < self.min_cluster_size:
                self.labels_[self.labels_ == label] = -1  # Mark as noise
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new embeddings."""
        if not self.is_fitted:
            raise RuntimeError("Clusterer not fitted")
        
        if self.clusterer is None:
            logger.warning("No HDBSCAN clusterer available for prediction")
            return np.full(embeddings.shape[0], -1)
        
        # HDBSCAN doesn't have a direct predict method
        # We'll use approximate prediction based on core samples
        try:
            # For new points, find nearest core sample and assign its cluster
            core_sample_indices = self.clusterer.core_sample_indices_
            
            if len(core_sample_indices) == 0:
                return np.full(embeddings.shape[0], -1)
            
            from sklearn.metrics.pairwise import cosine_similarity
            
            # This is a simplified prediction - in practice you'd want to use
            # the actual HDBSCAN prediction algorithms
            predictions = []
            
            for embedding in embeddings:
                # Find most similar core sample
                similarities = cosine_similarity([embedding], embeddings[core_sample_indices])
                best_core_idx = core_sample_indices[np.argmax(similarities)]
                predicted_label = self.labels_[best_core_idx]
                
                # Apply similarity threshold
                if np.max(similarities) < self.similarity_threshold:
                    predicted_label = -1
                    
                predictions.append(predicted_label)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"HDBSCAN prediction failed: {e}")
            return np.full(embeddings.shape[0], -1)
    
    def get_cluster_probabilities(self) -> np.ndarray:
        """Get cluster membership probabilities."""
        if not self.is_fitted or self.clusterer is None:
            return np.array([])
        
        try:
            return self.clusterer.probabilities_
        except AttributeError:
            logger.warning("Cluster probabilities not available")
            return np.array([])
    
    def get_cluster_hierarchy(self) -> Dict[str, Any]:
        """Get information about the cluster hierarchy."""
        if not self.is_fitted or self.clusterer is None:
            return {}
        
        try:
            return {
                'cluster_persistence': self.clusterer.cluster_persistence_,
                'condensed_tree': self.clusterer.condensed_tree_,
                'single_linkage_tree': self.clusterer.single_linkage_tree_
            }
        except AttributeError:
            logger.warning("Cluster hierarchy information not available")
            return {}
