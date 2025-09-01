"""
Semantic analysis module for product categorization.
Groups semantically similar product names regardless of barcode differences.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    """
    Analyzes semantic clusters to identify product categories and provide insights.
    """
    
    def __init__(self):
        """Initialize semantic analyzer."""
        self.clusters_df = None
        self.category_mapping = None
        self.category_stats = None
        
    def analyze_clusters(self, clustered_data: pd.DataFrame, 
                        name_column: str = 'name',
                        barcode_column: str = 'barcode',
                        cluster_column: str = 'cluster_id') -> pd.DataFrame:
        """
        Analyze clustering results to identify semantic categories.
        
        Args:
            clustered_data: DataFrame with clustering results
            name_column: Name of product name column
            barcode_column: Name of barcode column
            cluster_column: Name of cluster ID column
            
        Returns:
            DataFrame with cluster analysis
        """
        self.clusters_df = clustered_data.copy()
        
        # Analyze each cluster
        cluster_analysis = []
        
        for cluster_id in clustered_data[cluster_column].unique():
            cluster_data = clustered_data[clustered_data[cluster_column] == cluster_id]
            
            # Get all names in this cluster
            names = cluster_data[name_column].tolist()
            barcodes = cluster_data[barcode_column].tolist()
            
            # Find most common name as representative
            name_counts = Counter(names)
            representative_name = name_counts.most_common(1)[0][0]
            
            # Infer category from representative name
            category = self._infer_category(representative_name, names)
            
            cluster_analysis.append({
                'cluster_id': cluster_id,
                'category': category,
                'representative_name': representative_name,
                'unique_names': len(set(names)),
                'unique_barcodes': len(set(barcodes)),
                'total_items': len(cluster_data),
                'all_names': ', '.join(set(names)),
                'name_variety': len(set(names)) / len(names) if len(names) > 0 else 0
            })
        
        analysis_df = pd.DataFrame(cluster_analysis)
        analysis_df = analysis_df.sort_values('total_items', ascending=False)
        
        return analysis_df
    
    def _infer_category(self, representative_name: str, all_names: List[str]) -> str:
        """
        Infer product category from names in the cluster.
        
        Args:
            representative_name: Most common name in cluster
            all_names: All names in the cluster
            
        Returns:
            Inferred category name
        """
        # Combine all names for analysis
        combined_text = ' '.join(all_names).lower()
        
        # Category keywords (multilingual)
        category_keywords = {
            'Tables': ['table', 'desk', 'masa', 'mesa', 'writing', 'work', 'study', 'çalışma'],
            'Chairs': ['chair', 'sandalye', 'silla', 'seat', 'ergonomic', 'gaming', 'task'],
            'Computers': ['computer', 'pc', 'bilgisayar', 'ordenador', 'laptop', 'desktop', 'workstation'],
            'Monitors': ['monitor', 'display', 'screen', 'ekran', 'lcd'],
            'Lamps': ['lamp', 'light', 'lambası', 'reading', 'desk', 'led', 'study'],
            'Cups': ['cup', 'mug', 'fincan', 'taza', 'coffee', 'tea', 'drinking'],
            'Books': ['book', 'notebook', 'journal', 'kitap', 'cuaderno', 'writing', 'pad'],
            'Cabinets': ['cabinet', 'dolap', 'armario', 'storage', 'file', 'office'],
            'Shelves': ['shelf', 'rack', 'rafı', 'estantería', 'book', 'library'],
            'Mice': ['mouse', 'fare', 'ratón', 'wireless', 'optical', 'computer']
        }
        
        # Score each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            category_scores[category] = score
        
        # Return category with highest score, or 'Other' if no matches
        if category_scores and max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        else:
            return 'Other'
    
    def get_category_summary(self, analysis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics by product category.
        
        Args:
            analysis_df: DataFrame from analyze_clusters()
            
        Returns:
            DataFrame with category summaries
        """
        category_summary = analysis_df.groupby('category').agg({
            'cluster_id': 'count',  # Number of clusters per category
            'unique_barcodes': 'sum',  # Total unique barcodes per category
            'total_items': 'sum',  # Total items per category
            'unique_names': 'sum',  # Total unique names per category
            'representative_name': lambda x: ', '.join(x[:3])  # Sample names
        }).rename(columns={
            'cluster_id': 'num_clusters',
            'unique_barcodes': 'total_barcodes',
            'representative_name': 'example_names'
        })
        
        # Calculate average items per barcode
        category_summary['avg_items_per_barcode'] = (
            category_summary['total_items'] / category_summary['total_barcodes']
        ).round(2)
        
        # Sort by total barcodes (most important categories first)
        category_summary = category_summary.sort_values('total_barcodes', ascending=False)
        
        return category_summary.reset_index()
    
    def find_potential_duplicates(self, analysis_df: pd.DataFrame, 
                                similarity_threshold: float = 0.8) -> List[Dict]:
        """
        Find potential duplicate categories that might need manual review.
        
        Args:
            analysis_df: DataFrame from analyze_clusters()
            similarity_threshold: Minimum similarity to flag as potential duplicate
            
        Returns:
            List of potential duplicate pairs
        """
        from rapidfuzz import fuzz
        
        potential_duplicates = []
        categories = analysis_df['category'].unique()
        
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                # Compare representative names from each category
                cat1_names = analysis_df[analysis_df['category'] == cat1]['representative_name'].tolist()
                cat2_names = analysis_df[analysis_df['category'] == cat2]['representative_name'].tolist()
                
                # Check for high similarity between any names
                max_similarity = 0
                best_pair = None
                
                for name1 in cat1_names:
                    for name2 in cat2_names:
                        sim = fuzz.ratio(name1.lower(), name2.lower()) / 100.0
                        if sim > max_similarity:
                            max_similarity = sim
                            best_pair = (name1, name2)
                
                if max_similarity >= similarity_threshold:
                    potential_duplicates.append({
                        'category1': cat1,
                        'category2': cat2,
                        'similarity': max_similarity,
                        'example_pair': best_pair,
                        'suggestion': 'Consider merging these categories'
                    })
        
        return potential_duplicates
    
    def create_detailed_report(self, analysis_df: pd.DataFrame, 
                             category_summary: pd.DataFrame) -> str:
        """
        Create a detailed semantic analysis report.
        
        Args:
            analysis_df: DataFrame from analyze_clusters()
            category_summary: DataFrame from get_category_summary()
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("SEMANTIC PRODUCT ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Overall statistics
        total_items = analysis_df['total_items'].sum()
        total_categories = len(category_summary)
        total_clusters = len(analysis_df)
        total_barcodes = analysis_df['unique_barcodes'].sum()
        
        report.append(f"📊 OVERVIEW:")
        report.append(f"  • Total product items: {total_items:,}")
        report.append(f"  • Total unique barcodes: {total_barcodes:,}")
        report.append(f"  • Product categories found: {total_categories}")
        report.append(f"  • Semantic clusters: {total_clusters}")
        report.append(f"  • Average items per barcode: {total_items/total_barcodes:.2f}")
        report.append("")
        
        # Category breakdown
        report.append("🏷️ CATEGORY BREAKDOWN:")
        for _, row in category_summary.iterrows():
            category = row['category']
            barcodes = row['total_barcodes']
            items = row['total_items']
            clusters = row['num_clusters']
            examples = row['example_names']
            
            percentage = (barcodes / total_barcodes) * 100
            report.append(f"  📂 {category}:")
            report.append(f"     • {barcodes} unique products ({percentage:.1f}% of inventory)")
            report.append(f"     • {items} total items across {clusters} name variations")
            report.append(f"     • Examples: {examples}")
            report.append("")
        
        # Detailed cluster information
        report.append("🔍 DETAILED CLUSTER ANALYSIS:")
        for _, row in analysis_df.head(10).iterrows():  # Top 10 clusters
            cluster_id = row['cluster_id']
            category = row['category']
            rep_name = row['representative_name']
            unique_names = row['unique_names']
            unique_barcodes = row['unique_barcodes']
            all_names = row['all_names']
            
            report.append(f"  🏷️ Cluster {cluster_id} ({category}):")
            report.append(f"     • Representative: '{rep_name}'")
            report.append(f"     • {unique_barcodes} unique products, {unique_names} name variations")
            report.append(f"     • All names: {all_names}")
            report.append("")
        
        # Potential issues
        small_clusters = analysis_df[analysis_df['unique_barcodes'] == 1]
        if len(small_clusters) > 0:
            report.append("⚠️ POTENTIAL ISSUES:")
            report.append(f"  • {len(small_clusters)} singleton clusters (may need manual review)")
            report.append("  • Consider lowering clustering threshold if too many singletons")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def demo_semantic_analysis():
    """Demo function showing semantic analysis capabilities."""
    # This would normally use real clustering results
    print("🧪 Semantic Analysis Demo")
    print("=" * 40)
    print("This demo shows how semantic analysis works with clustering results.")
    print("Run the full pipeline to see real semantic analysis!")


if __name__ == "__main__":
    demo_semantic_analysis()
