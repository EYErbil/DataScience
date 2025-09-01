"""
Cluster merging and canonicalization module.
Combines clustering results with barcode information to create canonical product representations.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


class ClusterMerger:
    """
    Merges name clusters with barcode information to create canonical product entries.
    Handles the logic of combining products that share barcodes across different name clusters.
    """
    
    def __init__(self):
        """Initialize cluster merger."""
        self.original_data = None
        self.clustered_data = None
        self.merged_clusters = None
        self.canonical_names = None
        
    def merge_clusters_by_barcode(self, df: pd.DataFrame, 
                                cluster_column: str = 'cluster_id',
                                name_column: str = 'name',
                                barcode_column: str = 'barcode') -> pd.DataFrame:
        """
        Merge clusters based on shared barcodes.
        
        Args:
            df: DataFrame with clustered data
            cluster_column: Name of cluster ID column
            name_column: Name of product name column
            barcode_column: Name of barcode column
            
        Returns:
            DataFrame with merged cluster information
        """
        self.original_data = df.copy()
        
        # Create barcode to cluster mapping
        barcode_clusters = defaultdict(set)
        for _, row in df.iterrows():
            barcode_clusters[row[barcode_column]].add(row[cluster_column])
        
        # Find clusters that should be merged (share barcodes)
        cluster_groups = defaultdict(set)
        cluster_to_group = {}
        group_id = 0
        
        for barcode, clusters in barcode_clusters.items():
            if len(clusters) > 1:
                # Multiple clusters share this barcode - they should be merged
                clusters_list = list(clusters)
                
                # Check if any of these clusters are already in a group
                existing_groups = {cluster_to_group.get(c) for c in clusters_list if c in cluster_to_group}
                existing_groups.discard(None)
                
                if existing_groups:
                    # Merge with existing group(s)
                    target_group = min(existing_groups)
                    for cluster in clusters_list:
                        cluster_groups[target_group].add(cluster)
                        cluster_to_group[cluster] = target_group
                    
                    # If multiple existing groups, merge them
                    for group in existing_groups:
                        if group != target_group:
                            cluster_groups[target_group].update(cluster_groups[group])
                            for cluster in cluster_groups[group]:
                                cluster_to_group[cluster] = target_group
                            del cluster_groups[group]
                else:
                    # Create new group
                    for cluster in clusters_list:
                        cluster_groups[group_id].add(cluster)
                        cluster_to_group[cluster] = group_id
                    group_id += 1
        
        # Assign remaining clusters to their own groups
        for cluster in df[cluster_column].unique():
            if cluster not in cluster_to_group:
                cluster_groups[group_id] = {cluster}
                cluster_to_group[cluster] = group_id
                group_id += 1
        
        # Add merged cluster information to dataframe
        df_merged = df.copy()
        df_merged['merged_cluster_id'] = df_merged[cluster_column].map(cluster_to_group)
        
        # Add group size information
        group_sizes = {}
        for group_id, clusters in cluster_groups.items():
            group_size = df_merged[df_merged['merged_cluster_id'] == group_id].shape[0]
            group_sizes[group_id] = group_size
        
        df_merged['merged_cluster_size'] = df_merged['merged_cluster_id'].map(group_sizes)
        
        self.clustered_data = df_merged
        self.merged_clusters = cluster_groups
        
        logger.info(f"Merged {len(df[cluster_column].unique())} original clusters into "
                   f"{len(cluster_groups)} merged clusters")
        
        return df_merged
    
    def generate_canonical_names(self, df: pd.DataFrame,
                               name_column: str = 'name',
                               cluster_column: str = 'merged_cluster_id',
                               method: str = 'most_common') -> Dict[int, str]:
        """
        Generate canonical names for each merged cluster.
        
        Args:
            df: DataFrame with merged cluster data
            name_column: Name of product name column
            cluster_column: Name of merged cluster ID column
            method: Method for selecting canonical name ('most_common', 'shortest', 'longest', 'centroid')
            
        Returns:
            Dictionary mapping cluster ID to canonical name
        """
        canonical_names = {}
        
        for cluster_id in df[cluster_column].unique():
            cluster_data = df[df[cluster_column] == cluster_id]
            names = cluster_data[name_column].tolist()
            
            if method == 'most_common':
                # Use most frequent name
                name_counts = Counter(names)
                canonical_name = name_counts.most_common(1)[0][0]
                
            elif method == 'shortest':
                # Use shortest name
                canonical_name = min(names, key=len)
                
            elif method == 'longest':
                # Use longest name
                canonical_name = max(names, key=len)
                
            elif method == 'centroid':
                # Use name closest to cluster centroid (requires embeddings)
                # For now, fallback to most common
                name_counts = Counter(names)
                canonical_name = name_counts.most_common(1)[0][0]
                logger.warning("Centroid method not implemented, using most_common")
                
            else:
                raise ValueError(f"Unknown method: {method}")
            
            canonical_names[cluster_id] = canonical_name
        
        self.canonical_names = canonical_names
        logger.info(f"Generated canonical names for {len(canonical_names)} clusters")
        
        return canonical_names
    
    def create_enriched_dataset(self, df: pd.DataFrame,
                              name_column: str = 'name',
                              barcode_column: str = 'barcode',
                              cluster_column: str = 'merged_cluster_id') -> pd.DataFrame:
        """
        Create enriched dataset with canonical names and cluster information.
        
        Args:
            df: DataFrame with merged cluster data
            name_column: Name of product name column
            barcode_column: Name of barcode column
            cluster_column: Name of merged cluster ID column
            
        Returns:
            Enriched DataFrame with canonical representations
        """
        if self.canonical_names is None:
            self.generate_canonical_names(df, name_column, cluster_column)
        
        enriched_df = df.copy()
        
        # Add canonical names
        enriched_df['canonical_name'] = enriched_df[cluster_column].map(self.canonical_names)
        
        # Add alternative names for each cluster
        cluster_alternatives = {}
        for cluster_id in df[cluster_column].unique():
            cluster_data = df[df[cluster_column] == cluster_id]
            alternatives = list(set(cluster_data[name_column].tolist()))
            cluster_alternatives[cluster_id] = alternatives
        
        enriched_df['alternative_names'] = enriched_df[cluster_column].map(
            lambda x: ', '.join(cluster_alternatives[x])
        )
        
        # Add cluster statistics
        enriched_df['unique_names_in_cluster'] = enriched_df[cluster_column].map(
            lambda x: len(set(df[df[cluster_column] == x][name_column]))
        )
        
        enriched_df['unique_barcodes_in_cluster'] = enriched_df[cluster_column].map(
            lambda x: len(set(df[df[cluster_column] == x][barcode_column]))
        )
        
        return enriched_df
    
    def get_cluster_summary(self, df: pd.DataFrame,
                          cluster_column: str = 'merged_cluster_id',
                          name_column: str = 'name',
                          barcode_column: str = 'barcode') -> pd.DataFrame:
        """
        Get summary statistics for each cluster.
        
        Args:
            df: DataFrame with merged cluster data
            cluster_column: Name of merged cluster ID column
            name_column: Name of product name column
            barcode_column: Name of barcode column
            
        Returns:
            DataFrame with cluster summary statistics
        """
        summary_data = []
        
        for cluster_id in df[cluster_column].unique():
            cluster_data = df[df[cluster_column] == cluster_id]
            
            unique_names = cluster_data[name_column].nunique()
            unique_barcodes = cluster_data[barcode_column].nunique()
            total_records = len(cluster_data)
            
            most_common_name = cluster_data[name_column].value_counts().index[0]
            most_common_barcode = cluster_data[barcode_column].value_counts().index[0]
            
            canonical_name = self.canonical_names.get(cluster_id, most_common_name)
            
            summary_data.append({
                'cluster_id': cluster_id,
                'canonical_name': canonical_name,
                'unique_names': unique_names,
                'unique_barcodes': unique_barcodes,
                'total_records': total_records,
                'most_common_name': most_common_name,
                'most_common_barcode': most_common_barcode,
                'name_variety_ratio': unique_names / total_records,
                'barcode_variety_ratio': unique_barcodes / total_records
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('total_records', ascending=False)
        
        return summary_df


class DataEnricher:
    """
    Orchestrates the complete data enrichment pipeline.
    """
    
    def __init__(self):
        """Initialize data enricher."""
        self.ingester = None
        self.normalizer = None
        self.embedder = None
        self.clusterer = None
        self.merger = None
        
    def enrich_csv(self, file_path: str,
                  name_column: str = None,
                  barcode_column: str = None,
                  similarity_threshold: float = 0.7,
                  clustering_method: str = 'agglomerative',
                  canonical_method: str = 'most_common',
                  normalize_text: bool = True) -> pd.DataFrame:
        """
        Complete enrichment pipeline for a CSV file.
        
        Args:
            file_path: Path to input CSV file
            name_column: Name column (auto-detected if None)
            barcode_column: Barcode column (auto-detected if None)
            similarity_threshold: Threshold for clustering
            clustering_method: Clustering algorithm to use
            canonical_method: Method for selecting canonical names
            normalize_text: Whether to normalize text before clustering
            
        Returns:
            Enriched DataFrame
        """
        # Import modules
        try:
            from .ingest import CSVIngester
            from .normalize import TextNormalizer
            from .embedding import NameEmbedder
            from .cluster import NameClusterer
        except ImportError:
            import sys
            sys.path.append('.')
            from ingest import CSVIngester
            from normalize import TextNormalizer
            from embedding import NameEmbedder
            from cluster import NameClusterer
        
        logger.info(f"Starting enrichment pipeline for {file_path}")
        
        # 1. Ingest data
        self.ingester = CSVIngester()
        self.ingester.load_csv(file_path)
        
        if name_column and barcode_column:
            self.ingester.name_column = name_column
            self.ingester.barcode_column = barcode_column
        else:
            self.ingester.detect_columns()
        
        clean_data = self.ingester.get_clean_data()
        logger.info(f"Loaded {len(clean_data)} clean records")
        
        # 2. Normalize text (optional)
        if normalize_text:
            self.normalizer = TextNormalizer()
            clean_data['normalized_name'] = self.normalizer.normalize_batch(
                clean_data['name'].tolist()
            )
            embedding_column = 'normalized_name'
        else:
            embedding_column = 'name'
        
        # 3. Generate embeddings
        self.embedder = NameEmbedder()
        embeddings = self.embedder.generate_embeddings(clean_data[embedding_column].tolist())
        
        # 4. Cluster similar names
        self.clusterer = NameClusterer(clustering_method)
        cluster_labels = self.clusterer.fit(
            embeddings, 
            clean_data[embedding_column].tolist(),
            similarity_threshold=similarity_threshold
        )
        
        clean_data['cluster_id'] = cluster_labels
        
        # 5. Merge clusters by barcode
        self.merger = ClusterMerger()
        merged_data = self.merger.merge_clusters_by_barcode(clean_data)
        
        # 6. Generate canonical names
        canonical_names = self.merger.generate_canonical_names(
            merged_data, 
            name_column='name',
            method=canonical_method
        )
        
        # 7. Create enriched dataset
        enriched_data = self.merger.create_enriched_dataset(merged_data)
        
        logger.info(f"Enrichment complete: {len(enriched_data)} records processed")
        
        return enriched_data
    
    def save_results(self, enriched_data: pd.DataFrame, 
                    output_path: str,
                    include_summary: bool = True):
        """
        Save enriched results to CSV file.
        
        Args:
            enriched_data: Enriched DataFrame
            output_path: Output file path
            include_summary: Whether to save cluster summary
        """
        # Save main enriched data
        enriched_data.to_csv(output_path, index=False)
        logger.info(f"Saved enriched data to {output_path}")
        
        if include_summary and self.merger:
            # Save cluster summary
            summary_path = output_path.replace('.csv', '_cluster_summary.csv')
            summary = self.merger.get_cluster_summary(enriched_data)
            summary.to_csv(summary_path, index=False)
            logger.info(f"Saved cluster summary to {summary_path}")


def demo_merging():
    """Demo function showing cluster merging capabilities."""
    # Create sample data
    sample_data = pd.DataFrame({
        'name': [
            'office table', 'desk for office', 'çalışma masası',
            'office chair', 'computer chair', 'sandalye',
            'laptop computer', 'desktop pc', 'bilgisayar',
            'led lamp', 'table lamp', 'masa lambası'
        ],
        'barcode': [
            'TBL001', 'TBL001', 'TBL002',  # Same table, different names
            'CHR001', 'CHR001', 'CHR002',  # Same chair, different names
            'PC001', 'PC001', 'PC001',     # Same computer, different names
            'LMP001', 'LMP002', 'LMP002'   # Different lamps
        ],
        'cluster_id': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]  # Initial clusters
    })
    
    print("=" * 60)
    print("CLUSTER MERGING DEMO")
    print("=" * 60)
    
    print("Original data:")
    print(sample_data.to_string(index=False))
    
    # Initialize merger
    merger = ClusterMerger()
    
    # Merge clusters by barcode
    merged_data = merger.merge_clusters_by_barcode(sample_data)
    
    print("\nAfter merging by barcode:")
    print(merged_data[['name', 'barcode', 'cluster_id', 'merged_cluster_id']].to_string(index=False))
    
    # Generate canonical names
    canonical_names = merger.generate_canonical_names(merged_data)
    
    print("\nCanonical names:")
    for cluster_id, name in canonical_names.items():
        print(f"  Cluster {cluster_id}: {name}")
    
    # Create enriched dataset
    enriched_data = merger.create_enriched_dataset(merged_data)
    
    print("\nEnriched data:")
    print(enriched_data[['name', 'barcode', 'canonical_name', 'merged_cluster_id']].to_string(index=False))
    
    # Get cluster summary
    summary = merger.get_cluster_summary(merged_data)
    
    print("\nCluster summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    demo_merging()
