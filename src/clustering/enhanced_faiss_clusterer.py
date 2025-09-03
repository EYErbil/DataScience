"""
Enhanced FAISS-based clustering with advanced techniques:
- Adaptive cluster estimation
- Hierarchical post-processing
- Density-based noise filtering
- Multi-level clustering refinement
"""
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# Handle both relative and absolute imports
try:
    from .base import BaseClusterer
    from ..config import (SIMILARITY_THRESHOLD, MIN_CLUSTER_SIZE, MAX_CLUSTERS,
                         CLUSTERING_ITERATIONS, CLUSTERING_INIT_METHOD,
                         USE_HIERARCHICAL_REFINEMENT, DENSITY_THRESHOLD)
except ImportError:
    from clustering.base import BaseClusterer
    from config import (SIMILARITY_THRESHOLD, MIN_CLUSTER_SIZE, MAX_CLUSTERS,
                       CLUSTERING_ITERATIONS, CLUSTERING_INIT_METHOD,
                       USE_HIERARCHICAL_REFINEMENT, DENSITY_THRESHOLD)

logger = logging.getLogger(__name__)

class EnhancedFaissClusterer(BaseClusterer):
    """Enhanced FAISS clustering with advanced techniques."""
    
    def __init__(self, 
                 similarity_threshold: float = None,
                 min_cluster_size: int = None,
                 max_clusters: int = None,
                 use_gpu: bool = False,
                 use_hierarchical_refinement: bool = None,
                 density_threshold: float = None):
        super().__init__(
            similarity_threshold or SIMILARITY_THRESHOLD,
            min_cluster_size or MIN_CLUSTER_SIZE
        )
        self.max_clusters = max_clusters or MAX_CLUSTERS
        self.use_gpu = use_gpu
        self.use_hierarchical_refinement = use_hierarchical_refinement if use_hierarchical_refinement is not None else USE_HIERARCHICAL_REFINEMENT
        self.density_threshold = density_threshold or DENSITY_THRESHOLD
        
        self.index = None
        self.centroids = None
        self.cluster_densities = None
        self.silhouette_scores = None
        
    def _install_faiss(self):
        """Check and suggest FAISS installation."""
        try:
            import faiss
            return faiss
        except ImportError:
            logger.error("FAISS not installed. Install with: pip install faiss-cpu")
            raise ImportError("FAISS is required for EnhancedFaissClusterer")
    
    def _adaptive_cluster_estimation(self, embeddings: np.ndarray) -> int:
        """
        Adaptively estimate optimal number of clusters using multiple heuristics.
        """
        n_samples, dim = embeddings.shape
        
        # Heuristic 1: Elbow method approximation
        # For text data, typically sqrt(n/2) works well
        k_sqrt = max(3, min(int(np.sqrt(n_samples / 2)), n_samples // self.min_cluster_size))
        
        # Heuristic 2: Dimension-based estimation
        # Higher dimensions can support more clusters
        k_dim = max(3, min(dim // 10, n_samples // self.min_cluster_size))
        
        # Heuristic 3: Density-based estimation
        # Sample pairwise distances to estimate natural groupings
        if n_samples > 1000:
            # Sample for efficiency
            indices = np.random.choice(n_samples, 1000, replace=False)
            sample_embeddings = embeddings[indices]
        else:
            sample_embeddings = embeddings
            
        # Compute pairwise cosine distances
        distances = pdist(sample_embeddings, metric='cosine')
        
        # Find natural breakpoints in distance distribution
        sorted_distances = np.sort(distances)
        # Look for large gaps in distance distribution
        gaps = np.diff(sorted_distances)
        gap_threshold = np.percentile(gaps, 95)  # Top 5% of gaps
        
        significant_gaps = np.where(gaps > gap_threshold)[0]
        if len(significant_gaps) > 0:
            # Estimate clusters based on natural breakpoints
            k_density = min(len(significant_gaps) + 1, n_samples // self.min_cluster_size)
        else:
            k_density = k_sqrt
        
        # Combine heuristics with weights
        k_combined = int(0.4 * k_sqrt + 0.3 * k_dim + 0.3 * k_density)
        
        # Apply constraints
        if self.max_clusters:
            k_combined = min(self.max_clusters, k_combined)
        
        k_final = max(2, min(k_combined, n_samples // self.min_cluster_size))
        
        logger.info(f"🧠 Adaptive cluster estimation:")
        logger.info(f"   Sqrt heuristic: {k_sqrt}")
        logger.info(f"   Dimension heuristic: {k_dim}")
        logger.info(f"   Density heuristic: {k_density}")
        logger.info(f"   Final estimate: {k_final}")
        
        return k_final
    
    def _compute_cluster_densities(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
        """Compute density for each cluster to identify outliers."""
        densities = {}
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise
                continue
                
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            
            if len(cluster_embeddings) < 2:
                densities[cluster_id] = 0.0
                continue
            
            # Compute average pairwise cosine similarity within cluster
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(cluster_embeddings)
            
            # Remove diagonal (self-similarity)
            np.fill_diagonal(sim_matrix, 0)
            
            # Average similarity = cluster density
            avg_similarity = np.mean(sim_matrix)
            densities[cluster_id] = avg_similarity
        
        return densities
    
    def _hierarchical_refinement(self, embeddings: np.ndarray, initial_labels: np.ndarray) -> np.ndarray:
        """
        Refine clustering using hierarchical clustering on initial clusters.
        """
        logger.info("🔄 Applying hierarchical refinement...")
        
        refined_labels = initial_labels.copy()
        
        # For each initial cluster, check if it should be split further
        for cluster_id in np.unique(initial_labels):
            if cluster_id == -1:  # Skip noise
                continue
                
            cluster_mask = initial_labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            # Only refine if cluster is large enough
            if len(cluster_embeddings) < 2 * self.min_cluster_size:
                continue
            
            # Apply hierarchical clustering within this cluster
            try:
                # Use Ward linkage with cosine distance
                linkage_matrix = linkage(cluster_embeddings, method='ward')
                
                # Determine optimal number of sub-clusters
                max_subclusters = min(3, len(cluster_embeddings) // self.min_cluster_size)
                
                if max_subclusters > 1:
                    # Try different numbers of subclusters and pick best silhouette score
                    best_score = -1
                    best_subcluster_labels = None
                    
                    for n_subclusters in range(2, max_subclusters + 1):
                        subcluster_labels = fcluster(linkage_matrix, n_subclusters, criterion='maxclust')
                        
                        if len(np.unique(subcluster_labels)) > 1:
                            try:
                                score = silhouette_score(cluster_embeddings, subcluster_labels)
                                if score > best_score:
                                    best_score = score
                                    best_subcluster_labels = subcluster_labels
                            except:
                                continue
                    
                    # Apply refinement if it improves clustering
                    if best_subcluster_labels is not None and best_score > 0.1:
                        # Reassign labels
                        unique_subclusters = np.unique(best_subcluster_labels)
                        next_label = np.max(refined_labels) + 1
                        
                        for i, subcluster_id in enumerate(unique_subclusters):
                            subcluster_mask = best_subcluster_labels == subcluster_id
                            subcluster_indices = cluster_indices[subcluster_mask]
                            
                            if i == 0:
                                # Keep original cluster ID for first subcluster
                                refined_labels[subcluster_indices] = cluster_id
                            else:
                                # Assign new IDs for additional subclusters
                                refined_labels[subcluster_indices] = next_label
                                next_label += 1
                        
                        logger.info(f"   Split cluster {cluster_id} into {len(unique_subclusters)} subclusters")
                        
            except Exception as e:
                logger.warning(f"   Hierarchical refinement failed for cluster {cluster_id}: {e}")
                continue
        
        return refined_labels
    
    def _filter_by_density(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Filter out low-density clusters as noise."""
        if self.cluster_densities is None:
            self.cluster_densities = self._compute_cluster_densities(embeddings, labels)
        
        filtered_labels = labels.copy()
        
        for cluster_id, density in self.cluster_densities.items():
            if density < self.density_threshold:
                logger.info(f"   Filtering low-density cluster {cluster_id} (density: {density:.3f})")
                filtered_labels[labels == cluster_id] = -1  # Mark as noise
        
        return filtered_labels
    
    def fit(self, embeddings: np.ndarray, texts: List[str] = None) -> 'EnhancedFaissClusterer':
        """Enhanced FAISS clustering with advanced techniques."""
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
        k = self._adaptive_cluster_estimation(embeddings)
        
        logger.info(f"🎯 Enhanced FAISS clustering: {n_samples:,} samples → {k} clusters")
        
        try:
            # Create FAISS k-means clusterer with enhanced parameters
            kmeans = faiss.Kmeans(
                dim,
                k,
                niter=CLUSTERING_ITERATIONS,
                verbose=False,
                spherical=True,  # Use cosine similarity
                gpu=self.use_gpu and faiss.get_num_gpus() > 0,
                seed=42  # Reproducibility
            )
            
            # Fit the clusterer
            kmeans.train(embeddings)
            
            # Get cluster assignments
            _, labels = kmeans.index.search(embeddings, 1)
            self.labels_ = labels.flatten()
            
            # Store centroids for prediction
            self.centroids = kmeans.centroids
            
            # Apply advanced filtering and refinement
            logger.info("🔧 Applying advanced post-processing...")
            
            # 1. Filter small clusters
            self._filter_small_clusters()
            
            # 2. Compute cluster densities
            self.cluster_densities = self._compute_cluster_densities(embeddings, self.labels_)
            
            # 3. Filter low-density clusters
            self.labels_ = self._filter_by_density(embeddings, self.labels_)
            
            # 4. Hierarchical refinement (optional)
            if self.use_hierarchical_refinement:
                self.labels_ = self._hierarchical_refinement(embeddings, self.labels_)
            
            # 5. Final cleanup - filter small clusters again after refinement
            self._filter_small_clusters()
            
            # Update cluster count
            self.n_clusters_ = len(np.unique(self.labels_[self.labels_ >= 0]))
            
            # Compute silhouette scores for quality assessment
            if self.n_clusters_ > 1:
                try:
                    self.silhouette_scores = silhouette_score(embeddings, self.labels_)
                    logger.info(f"📊 Silhouette score: {self.silhouette_scores:.3f}")
                except:
                    self.silhouette_scores = None
            
            self.is_fitted = True
            
            logger.info(f"✅ Enhanced clustering complete: {self.n_clusters_} clusters")
            logger.info(f"   Noise points: {np.sum(self.labels_ == -1)} ({np.sum(self.labels_ == -1)/len(self.labels_)*100:.1f}%)")
            
            return self
            
        except Exception as e:
            logger.error(f"Enhanced FAISS clustering failed: {e}")
            # Fallback to simple clustering
            return self._fallback_clustering(embeddings)
    
    def _filter_small_clusters(self):
        """Filter out clusters smaller than min_cluster_size."""
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        
        for label, count in zip(unique_labels, counts):
            if label != -1 and count < self.min_cluster_size:  # Don't modify noise label
                self.labels_[self.labels_ == label] = -1  # Mark as noise
    
    def _fallback_clustering(self, embeddings: np.ndarray) -> 'EnhancedFaissClusterer':
        """Enhanced fallback clustering."""
        logger.warning("Using enhanced sklearn clustering as fallback")
        
        try:
            from sklearn.cluster import KMeans
            
            n_samples = embeddings.shape[0]
            k = self._adaptive_cluster_estimation(embeddings)
            
            # Use enhanced parameters
            kmeans = KMeans(
                n_clusters=k, 
                random_state=42, 
                n_init=10,
                init='k-means++',
                max_iter=CLUSTERING_ITERATIONS
            )
            self.labels_ = kmeans.fit_predict(embeddings)
            self.centroids = kmeans.cluster_centers_
            
            # Apply post-processing
            self._filter_small_clusters()
            self.labels_ = self._filter_by_density(embeddings, self.labels_)
            
            if self.use_hierarchical_refinement:
                self.labels_ = self._hierarchical_refinement(embeddings, self.labels_)
                self._filter_small_clusters()
            
            self.n_clusters_ = len(np.unique(self.labels_[self.labels_ >= 0]))
            
            if self.n_clusters_ > 1:
                try:
                    self.silhouette_scores = silhouette_score(embeddings, self.labels_)
                except:
                    self.silhouette_scores = None
            
            self.is_fitted = True
            
            logger.info(f"✅ Enhanced fallback clustering complete: {self.n_clusters_} clusters")
            return self
            
        except ImportError:
            logger.error("Neither FAISS nor sklearn available")
            # Ultimate fallback: no clustering
            self.labels_ = np.arange(embeddings.shape[0])
            self.n_clusters_ = embeddings.shape[0]
            self.is_fitted = True
            return self
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new embeddings with confidence filtering."""
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
            
            labels = labels.flatten()
            similarities = similarities.flatten()
            
            # Apply similarity threshold
            labels[similarities < self.similarity_threshold] = -1
            
            # Apply density filtering if available
            if self.cluster_densities is not None:
                for i, label in enumerate(labels):
                    if label >= 0 and label in self.cluster_densities:
                        cluster_density = self.cluster_densities[label]
                        if cluster_density < self.density_threshold:
                            labels[i] = -1
            
            return labels
            
        except Exception as e:
            logger.error(f"Enhanced FAISS prediction failed: {e}")
            return np.full(embeddings.shape[0], -1)
    
    def get_cluster_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics for the clustering."""
        if not self.is_fitted:
            return {}
        
        metrics = {
            'n_clusters': self.n_clusters_,
            'silhouette_score': self.silhouette_scores,
            'noise_ratio': np.sum(self.labels_ == -1) / len(self.labels_) if len(self.labels_) > 0 else 0,
            'cluster_densities': self.cluster_densities,
            'min_cluster_size': self.min_cluster_size,
            'similarity_threshold': self.similarity_threshold
        }
        
        return metrics
