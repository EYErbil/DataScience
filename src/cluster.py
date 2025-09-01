"""
Clustering module for grouping similar product names.
Supports multiple clustering algorithms with configurable parameters.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class NameClusterer:
    """
    Clusters similar product names using various algorithms.
    Optimized for handling multilingual product name similarities.
    """
    
    def __init__(self, method: str = 'agglomerative'):
        """
        Initialize clusterer with specified method.
        
        Args:
            method: Clustering method ('agglomerative', 'dbscan', 'kmeans')
        """
        self.method = method
        self.model = None
        self.labels = None
        self.texts = None
        self.embeddings = None
        self.similarity_matrix = None
        
    def fit_agglomerative(self, embeddings: np.ndarray, 
                         similarity_threshold: float = 0.7,
                         linkage: str = 'average',
                         max_clusters: int = None) -> np.ndarray:
        """
        Fit agglomerative clustering based on similarity threshold.
        
        Args:
            embeddings: Text embeddings array
            similarity_threshold: Minimum similarity for clustering
            linkage: Linkage method ('ward', 'complete', 'average', 'single')
            max_clusters: Maximum number of clusters (if None, uses distance threshold)
            
        Returns:
            Array of cluster labels
        """
        # Convert similarity to distance
        distance_threshold = 1 - similarity_threshold
        
        if max_clusters:
            self.model = AgglomerativeClustering(
                n_clusters=max_clusters,
                linkage=linkage
            )
        else:
            self.model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage=linkage
            )
        
        self.labels = self.model.fit_predict(embeddings)
        logger.info(f"Agglomerative clustering: {len(set(self.labels))} clusters found")
        
        return self.labels
    
    def fit_dbscan(self, embeddings: np.ndarray,
                   eps: float = 0.3,
                   min_samples: int = 2,
                   metric: str = 'cosine') -> np.ndarray:
        """
        Fit DBSCAN clustering.
        
        Args:
            embeddings: Text embeddings array
            eps: Maximum distance between samples in a cluster
            min_samples: Minimum number of samples in a cluster
            metric: Distance metric ('cosine', 'euclidean', etc.)
            
        Returns:
            Array of cluster labels (-1 for noise/outliers)
        """
        self.model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        self.labels = self.model.fit_predict(embeddings)
        
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        
        logger.info(f"DBSCAN clustering: {n_clusters} clusters, {n_noise} noise points")
        
        return self.labels
    
    def fit_kmeans(self, embeddings: np.ndarray,
                   n_clusters: int = 8,
                   random_state: int = 42) -> np.ndarray:
        """
        Fit K-means clustering.
        
        Args:
            embeddings: Text embeddings array
            n_clusters: Number of clusters
            random_state: Random state for reproducibility
            
        Returns:
            Array of cluster labels
        """
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.labels = self.model.fit_predict(embeddings)
        
        logger.info(f"K-means clustering: {n_clusters} clusters")
        
        return self.labels
    
    def fit(self, embeddings: np.ndarray, texts: List[str], **kwargs) -> np.ndarray:
        """
        Fit clustering model with specified method.
        
        Args:
            embeddings: Text embeddings array
            texts: List of original texts
            **kwargs: Method-specific parameters
            
        Returns:
            Array of cluster labels
        """
        self.embeddings = embeddings
        self.texts = texts
        
        if self.method == 'agglomerative':
            return self.fit_agglomerative(embeddings, **kwargs)
        elif self.method == 'dbscan':
            return self.fit_dbscan(embeddings, **kwargs)
        elif self.method == 'kmeans':
            return self.fit_kmeans(embeddings, **kwargs)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
    
    def get_clusters_dataframe(self) -> pd.DataFrame:
        """
        Get clustering results as a pandas DataFrame.
        
        Returns:
            DataFrame with columns: text, cluster_id, cluster_size
        """
        if self.labels is None or self.texts is None:
            raise ValueError("No clustering results available. Call fit() first.")
        
        df = pd.DataFrame({
            'text': self.texts,
            'cluster_id': self.labels
        })
        
        # Add cluster size information
        cluster_sizes = df['cluster_id'].value_counts().to_dict()
        df['cluster_size'] = df['cluster_id'].map(cluster_sizes)
        
        # Sort by cluster ID and cluster size
        df = df.sort_values(['cluster_size', 'cluster_id'], ascending=[False, True])
        
        return df
    
    def get_cluster_summary(self) -> Dict:
        """
        Get summary statistics about the clustering results.
        
        Returns:
            Dictionary with clustering statistics
        """
        if self.labels is None:
            raise ValueError("No clustering results available. Call fit() first.")
        
        unique_labels = set(self.labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(self.labels).count(-1) if -1 in self.labels else 0
        
        cluster_sizes = [list(self.labels).count(label) for label in unique_labels if label != -1]
        
        summary = {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'n_total_points': len(self.labels),
            'largest_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'smallest_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'average_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'singleton_clusters': sum(1 for size in cluster_sizes if size == 1),
            'method': self.method
        }
        
        return summary
    
    def evaluate_clustering(self) -> Dict:
        """
        Evaluate clustering quality using various metrics.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.labels is None or self.embeddings is None:
            raise ValueError("No clustering results available. Call fit() first.")
        
        metrics = {}
        
        # Only calculate silhouette score if we have more than one cluster
        # and not all points are in the same cluster
        unique_labels = set(self.labels)
        if len(unique_labels) > 1 and len(unique_labels) < len(self.labels):
            try:
                # Remove noise points for silhouette score (DBSCAN)
                if -1 in self.labels:
                    mask = self.labels != -1
                    filtered_embeddings = self.embeddings[mask]
                    filtered_labels = self.labels[mask]
                    
                    if len(set(filtered_labels)) > 1:
                        metrics['silhouette_score'] = silhouette_score(
                            filtered_embeddings, filtered_labels
                        )
                else:
                    metrics['silhouette_score'] = silhouette_score(
                        self.embeddings, self.labels
                    )
            except Exception as e:
                logger.warning(f"Could not calculate silhouette score: {e}")
                metrics['silhouette_score'] = None
        else:
            metrics['silhouette_score'] = None
        
        # Add summary statistics
        metrics.update(self.get_cluster_summary())
        
        return metrics


class ClusterOptimizer:
    """
    Optimizes clustering parameters by testing different configurations.
    """
    
    def __init__(self, embeddings: np.ndarray, texts: List[str]):
        """
        Initialize optimizer with embeddings and texts.
        
        Args:
            embeddings: Text embeddings array
            texts: List of original texts
        """
        self.embeddings = embeddings
        self.texts = texts
        
    def optimize_agglomerative(self, similarity_thresholds: List[float] = None) -> Dict:
        """
        Find optimal similarity threshold for agglomerative clustering.
        
        Args:
            similarity_thresholds: List of thresholds to test
            
        Returns:
            Dictionary with results for each threshold
        """
        if similarity_thresholds is None:
            similarity_thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
        
        results = {}
        
        for threshold in similarity_thresholds:
            clusterer = NameClusterer('agglomerative')
            clusterer.fit(self.embeddings, self.texts, similarity_threshold=threshold)
            
            evaluation = clusterer.evaluate_clustering()
            results[threshold] = evaluation
            
            logger.info(f"Threshold {threshold}: {evaluation['n_clusters']} clusters, "
                       f"silhouette: {evaluation.get('silhouette_score', 'N/A')}")
        
        return results
    
    def optimize_dbscan(self, eps_values: List[float] = None, 
                       min_samples_values: List[int] = None) -> Dict:
        """
        Find optimal parameters for DBSCAN clustering.
        
        Args:
            eps_values: List of eps values to test
            min_samples_values: List of min_samples values to test
            
        Returns:
            Dictionary with results for each parameter combination
        """
        if eps_values is None:
            eps_values = [0.2, 0.3, 0.4, 0.5]
        if min_samples_values is None:
            min_samples_values = [2, 3, 4, 5]
        
        results = {}
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                clusterer = NameClusterer('dbscan')
                clusterer.fit(self.embeddings, self.texts, 
                            eps=eps, min_samples=min_samples)
                
                evaluation = clusterer.evaluate_clustering()
                key = f"eps_{eps}_min_{min_samples}"
                results[key] = evaluation
                results[key]['eps'] = eps
                results[key]['min_samples'] = min_samples
                
                logger.info(f"DBSCAN eps={eps}, min_samples={min_samples}: "
                           f"{evaluation['n_clusters']} clusters, "
                           f"{evaluation['n_noise_points']} noise points")
        
        return results
    
    def find_best_configuration(self, methods: List[str] = None) -> Dict:
        """
        Find the best clustering configuration across multiple methods.
        
        Args:
            methods: List of methods to test ('agglomerative', 'dbscan', 'kmeans')
            
        Returns:
            Dictionary with best configuration and results
        """
        if methods is None:
            methods = ['agglomerative', 'dbscan']
        
        all_results = {}
        
        if 'agglomerative' in methods:
            agg_results = self.optimize_agglomerative()
            for threshold, result in agg_results.items():
                key = f"agglomerative_threshold_{threshold}"
                all_results[key] = result
                all_results[key]['method'] = 'agglomerative'
                all_results[key]['threshold'] = threshold
        
        if 'dbscan' in methods:
            dbscan_results = self.optimize_dbscan()
            all_results.update(dbscan_results)
        
        if 'kmeans' in methods:
            # Test different numbers of clusters for K-means
            for n_clusters in [5, 8, 10, 15, 20]:
                clusterer = NameClusterer('kmeans')
                clusterer.fit(self.embeddings, self.texts, n_clusters=n_clusters)
                
                evaluation = clusterer.evaluate_clustering()
                key = f"kmeans_n_{n_clusters}"
                all_results[key] = evaluation
                all_results[key]['n_clusters_param'] = n_clusters
        
        # Find best configuration based on criteria
        best_config = None
        best_score = -1
        
        for config, result in all_results.items():
            # Score based on multiple criteria
            score = 0
            
            # Prefer reasonable number of clusters
            n_clusters = result['n_clusters']
            if 3 <= n_clusters <= 20:
                score += 2
            elif 2 <= n_clusters <= 30:
                score += 1
            
            # Prefer higher silhouette score
            if result.get('silhouette_score'):
                score += result['silhouette_score'] * 3
            
            # Prefer fewer singleton clusters
            singleton_ratio = result['singleton_clusters'] / result['n_clusters'] if result['n_clusters'] > 0 else 1
            score -= singleton_ratio
            
            # Prefer fewer noise points (for DBSCAN)
            noise_ratio = result['n_noise_points'] / result['n_total_points']
            score -= noise_ratio * 2
            
            if score > best_score:
                best_score = score
                best_config = config
        
        return {
            'best_config': best_config,
            'best_result': all_results[best_config] if best_config else None,
            'all_results': all_results,
            'best_score': best_score
        }


def demo_clustering(sample_texts: List[str] = None):
    """
    Demo function showing clustering capabilities.
    
    Args:
        sample_texts: List of texts to cluster (uses default if None)
    """
    if sample_texts is None:
        sample_texts = [
            "office table", "desk for office", "çalışma masası", "masa",
            "office chair", "computer chair", "sandalye", "gaming chair",
            "laptop computer", "desktop pc", "bilgisayar", "notebook",
            "led lamp", "table lamp", "masa lambası", "light",
            "book shelf", "bookcase", "kitap rafı", "shelf",
            "monitor screen", "display", "ekran", "lcd monitor"
        ]
    
    print("=" * 60)
    print("CLUSTERING DEMO")
    print("=" * 60)
    
    # Import embedding module for demo
    try:
        from .embedding import NameEmbedder
    except ImportError:
        import sys
        sys.path.append('.')
        from embedding import NameEmbedder
    
    # Generate embeddings
    embedder = NameEmbedder()
    embeddings = embedder.generate_embeddings(sample_texts)
    
    print(f"Sample texts ({len(sample_texts)}):")
    for i, text in enumerate(sample_texts):
        print(f"  {i+1:2d}. {text}")
    
    # Test different clustering methods
    print("\n" + "=" * 40)
    print("AGGLOMERATIVE CLUSTERING")
    print("=" * 40)
    
    clusterer_agg = NameClusterer('agglomerative')
    clusterer_agg.fit(embeddings, sample_texts, similarity_threshold=0.7)
    
    df_agg = clusterer_agg.get_clusters_dataframe()
    print(df_agg.to_string(index=False))
    
    eval_agg = clusterer_agg.evaluate_clustering()
    print(f"\nClusters: {eval_agg['n_clusters']}")
    print(f"Silhouette Score: {eval_agg.get('silhouette_score', 'N/A')}")
    
    print("\n" + "=" * 40)
    print("DBSCAN CLUSTERING")
    print("=" * 40)
    
    clusterer_dbscan = NameClusterer('dbscan')
    clusterer_dbscan.fit(embeddings, sample_texts, eps=0.3, min_samples=2)
    
    df_dbscan = clusterer_dbscan.get_clusters_dataframe()
    print(df_dbscan.to_string(index=False))
    
    eval_dbscan = clusterer_dbscan.evaluate_clustering()
    print(f"\nClusters: {eval_dbscan['n_clusters']}")
    print(f"Noise points: {eval_dbscan['n_noise_points']}")
    print(f"Silhouette Score: {eval_dbscan.get('silhouette_score', 'N/A')}")
    
    # Optimize parameters
    print("\n" + "=" * 40)
    print("PARAMETER OPTIMIZATION")
    print("=" * 40)
    
    optimizer = ClusterOptimizer(embeddings, sample_texts)
    best_config = optimizer.find_best_configuration()
    
    print(f"Best configuration: {best_config['best_config']}")
    if best_config['best_result']:
        result = best_config['best_result']
        print(f"Clusters: {result['n_clusters']}")
        print(f"Silhouette Score: {result.get('silhouette_score', 'N/A')}")


if __name__ == "__main__":
    demo_clustering()
