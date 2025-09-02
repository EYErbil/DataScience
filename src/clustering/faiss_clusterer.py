"""
FAISS-based clustering for scalable semantic grouping.
Can handle millions of vectors efficiently.
"""
import numpy as np
from typing import List, Optional, Dict, Any
import logging

# Handle both relative and absolute imports
try:
    from .base import BaseClusterer
    from ..config import SIMILARITY_THRESHOLD, MIN_CLUSTER_SIZE, MAX_CLUSTERS
except ImportError:
    from clustering.base import BaseClusterer
    from config import SIMILARITY_THRESHOLD, MIN_CLUSTER_SIZE, MAX_CLUSTERS

logger = logging.getLogger(__name__)

class FaissClusterer(BaseClusterer):
    """FAISS-based clustering for large-scale embeddings."""
    
    def __init__(self, 
                 similarity_threshold: float = None,
                 min_cluster_size: int = None,
                 max_clusters: int = None,
                 use_gpu: bool = False):
        super().__init__(
            similarity_threshold or SIMILARITY_THRESHOLD,
            min_cluster_size or MIN_CLUSTER_SIZE
        )
        self.max_clusters = max_clusters or MAX_CLUSTERS
        self.use_gpu = use_gpu
        self.index = None
        self.centroids = None
        
    def _install_faiss(self):
        """Check and suggest FAISS installation."""
        try:
            import faiss
            return faiss
        except ImportError:
            logger.error("FAISS not installed. Install with: pip install faiss-cpu")
            raise ImportError("FAISS is required for FaissClusterer")
    
    def _estimate_clusters(self, n_samples: int) -> int:
        """Estimate optimal number of clusters."""
        if self.max_clusters:
            return min(self.max_clusters, n_samples // self.min_cluster_size)
        
        # Use heuristic: sqrt(n/2) bounded by reasonable limits
        k = max(2, min(int(np.sqrt(n_samples / 2)), n_samples // self.min_cluster_size))
        return k
    
    def fit(self, embeddings: np.ndarray, texts: List[str] = None) -> 'FaissClusterer':
        """Fit FAISS k-means clustering on embeddings."""
        faiss = self._install_faiss()
        
        if embeddings.shape[0] < self.min_cluster_size:
            logger.warning(f"Too few samples ({embeddings.shape[0]}) for clustering")
            self.labels_ = np.zeros(embeddings.shape[0], dtype=int)
            self.n_clusters_ = 1
            self.is_fitted = True
            return self
        
        # Ensure embeddings are float32 and normalized
        embeddings = embeddings.astype(np.float32)
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        
        n_samples, dim = embeddings.shape
        k = self._estimate_clusters(n_samples)
        
        logger.info(f"🎯 FAISS clustering: {n_samples:,} samples → {k} clusters")
        
        try:
            # Create FAISS k-means clusterer
            kmeans = faiss.Kmeans(
                dim,
                k,
                niter=50,
                verbose=False,
                spherical=True,  # Use cosine similarity
                gpu=self.use_gpu and faiss.get_num_gpus() > 0
            )
            
            # Fit the clusterer
            kmeans.train(embeddings)
            
            # Get cluster assignments
            _, labels = kmeans.index.search(embeddings, 1)
            self.labels_ = labels.flatten()
            
            # Store centroids for prediction
            self.centroids = kmeans.centroids
            
            # Filter small clusters and mark as noise
            self._filter_small_clusters()
            
            # Update cluster count
            self.n_clusters_ = len(np.unique(self.labels_[self.labels_ >= 0]))
            self.is_fitted = True
            
            logger.info(f"✅ FAISS clustering complete: {self.n_clusters_} clusters")
            
            return self
            
        except Exception as e:
            logger.error(f"FAISS clustering failed: {e}")
            # Fallback to simple clustering
            return self._fallback_clustering(embeddings)
    
    def _filter_small_clusters(self):
        """Filter out clusters smaller than min_cluster_size."""
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        
        for label, count in zip(unique_labels, counts):
            if count < self.min_cluster_size:
                self.labels_[self.labels_ == label] = -1  # Mark as noise
    
    def _fallback_clustering(self, embeddings: np.ndarray) -> 'FaissClusterer':
        """Fallback to simple k-means if FAISS fails."""
        logger.warning("Using sklearn k-means as fallback")
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics.pairwise import cosine_similarity
            
            n_samples = embeddings.shape[0]
            k = self._estimate_clusters(n_samples)
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            self.labels_ = kmeans.fit_predict(embeddings)
            self.centroids = kmeans.cluster_centers_
            
            self._filter_small_clusters()
            self.n_clusters_ = len(np.unique(self.labels_[self.labels_ >= 0]))
            self.is_fitted = True
            
            logger.info(f"✅ Fallback clustering complete: {self.n_clusters_} clusters")
            return self
            
        except ImportError:
            logger.error("Neither FAISS nor sklearn available")
            # Ultimate fallback: no clustering
            self.labels_ = np.arange(embeddings.shape[0])
            self.n_clusters_ = embeddings.shape[0]
            self.is_fitted = True
            return self
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new embeddings."""
        if not self.is_fitted:
            raise RuntimeError("Clusterer not fitted")
        
        if self.centroids is None:
            logger.warning("No centroids available for prediction")
            return np.full(embeddings.shape[0], -1)
        
        faiss = self._install_faiss()
        
        # Normalize embeddings
        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        try:
            # Create index for centroids
            index = faiss.IndexFlatIP(self.centroids.shape[1])  # Inner product = cosine for normalized vectors
            index.add(self.centroids)
            
            # Find nearest centroid for each embedding
            similarities, labels = index.search(embeddings, 1)
            
            # Apply similarity threshold
            labels = labels.flatten()
            similarities = similarities.flatten()
            
            # Mark low-similarity assignments as noise
            threshold = 1 - self.similarity_threshold  # Convert to cosine similarity
            labels[similarities < threshold] = -1
            
            return labels
            
        except Exception as e:
            logger.error(f"FAISS prediction failed: {e}")
            return np.full(embeddings.shape[0], -1)
    
    def build_index(self, embeddings: np.ndarray) -> None:
        """Build FAISS index for fast similarity search."""
        faiss = self._install_faiss()
        
        embeddings = embeddings.astype(np.float32)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        dim = embeddings.shape[1]
        
        # Choose index type based on data size
        if embeddings.shape[0] < 10000:
            # Small dataset: exact search
            self.index = faiss.IndexFlatIP(dim)
        else:
            # Large dataset: approximate search
            nlist = min(int(np.sqrt(embeddings.shape[0])), 1000)
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist)
            self.index.train(embeddings)
        
        self.index.add(embeddings)
        
        logger.info(f"🔍 Built FAISS index: {embeddings.shape[0]:,} vectors")
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 10) -> tuple:
        """Search for similar embeddings using FAISS index."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Normalize query
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        similarities, indices = self.index.search(query_embedding, k)
        
        return similarities[0], indices[0]
