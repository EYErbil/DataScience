"""
Main pipeline runner for the product categorization system.
Orchestrates the full pipeline with smart caching and resumption.
"""
import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np

# Handle both relative and absolute imports
try:
    from .config import *
    from .io_utils import smart_cache, save_dataframe, load_dataframe, stream_csv, save_metadata
    from .ingest import CSVIngester
    from .normalize import MultilingualNormalizer
    from .embedding import HuggingFaceEncoder, TfidfEncoder
    from .clustering import FaissClusterer, HdbscanClusterer
    from .categorisation import AutoClusterMapper
except ImportError:
    # Fallback for direct execution or notebook imports
    from config import *
    from io_utils import smart_cache, save_dataframe, load_dataframe, stream_csv, save_metadata
    from ingest import CSVIngester
    from normalize import MultilingualNormalizer
    from embedding import HuggingFaceEncoder, TfidfEncoder
    from clustering import FaissClusterer, HdbscanClusterer
    from categorisation import AutoClusterMapper

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class ProductCategorizationPipeline:
    """
    Complete product categorization pipeline.
    """
    
    def __init__(self, 
                 main_categories: List[str],
                 encoder_type: str = 'auto',
                 clusterer_type: str = 'faiss',
                 force_rebuild: bool = False):
        self.main_categories = main_categories
        self.encoder_type = encoder_type
        self.clusterer_type = clusterer_type
        self.force_rebuild = force_rebuild
        
        # Pipeline components
        self.ingester = None
        self.normalizer = None
        self.encoder = None
        self.clusterer = None
        self.mapper = None
        
        # Results
        self.clean_data = None
        self.embeddings = None
        self.cluster_labels = None
        self.analysis_df = None
        
        logger.info(f"🚀 Pipeline initialized: {encoder_type} encoder, {clusterer_type} clusterer")
        logger.info(f"🎯 Target categories: {main_categories}")
    
    def run(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Run the complete pipeline.
        
        Args:
            csv_path: Path to input CSV file
            output_path: Path for output file (auto-generated if None)
            
        Returns:
            DataFrame with categorized results
        """
        start_time = time.time()
        
        logger.info("🏁 Starting product categorization pipeline")
        logger.info(f"📁 Input: {csv_path}")
        
        # Step 1: Data Ingestion and Cleaning
        self.clean_data = self._run_ingestion(csv_path)
        
        # Step 2: Text Normalization
        self.clean_data = self._run_normalization()
        
        # Step 3: Embedding Generation
        self.embeddings = self._run_embedding()
        
        # Step 4: Clustering
        self.cluster_labels = self._run_clustering()
        
        # Step 5: Category Assignment
        self.analysis_df = self._run_categorization()
        
        # Step 6: Save Results
        if output_path:
            final_results = self._prepare_final_results()
            save_dataframe(final_results, Path(output_path).stem)
            logger.info(f"💾 Results saved to {output_path}")
        
        # Pipeline summary
        elapsed = time.time() - start_time
        self._log_pipeline_summary(elapsed)
        
        return self.analysis_df
    
    def _run_ingestion(self, csv_path: str) -> pd.DataFrame:
        """Step 1: Ingest and clean data."""
        with smart_cache("clean_data", self.force_rebuild) as cache:
            if cache.exists:
                logger.info("📂 Loading cached clean data")
                return cache.load()
            
            logger.info("🔄 Running data ingestion...")
            self.ingester = CSVIngester()
            
            # Handle large files with streaming
            if Path(csv_path).stat().st_size > 100 * 1024 * 1024:  # > 100MB
                logger.info("📡 Large file detected, using streaming ingestion")
                all_chunks = []
                for chunk in stream_csv(csv_path, CHUNK_SIZE, MAX_ROWS):
                    clean_chunk = self.ingester.load_csv_data(chunk)
                    all_chunks.append(clean_chunk)
                clean_data = pd.concat(all_chunks, ignore_index=True)
            else:
                clean_data = self.ingester.load_csv(csv_path)
                clean_data = self.ingester.get_clean_data()
            
            logger.info(f"✅ Ingestion complete: {len(clean_data):,} clean records")
            cache.save(clean_data)
            return clean_data
    
    def _run_normalization(self) -> pd.DataFrame:
        """Step 2: Normalize text."""
        with smart_cache("normalized_data", self.force_rebuild) as cache:
            if cache.exists:
                logger.info("📂 Loading cached normalized data")
                return cache.load()
            
            logger.info("🔄 Running text normalization...")
            self.normalizer = MultilingualNormalizer()
            
            # Normalize in batches for memory efficiency
            normalized_names = []
            batch_size = 10000
            
            for i in range(0, len(self.clean_data), batch_size):
                batch = self.clean_data['name'].iloc[i:i+batch_size]
                batch_normalized = [self.normalizer.normalize_multilingual(name) for name in batch]
                normalized_names.extend(batch_normalized)
                
                if i % (batch_size * 10) == 0:
                    logger.info(f"Normalized {i:,}/{len(self.clean_data):,} items")
            
            self.clean_data['normalized_name'] = normalized_names
            
            logger.info("✅ Normalization complete")
            cache.save(self.clean_data)
            return self.clean_data
    
    def _run_embedding(self) -> np.ndarray:
        """Step 3: Generate embeddings."""
        with smart_cache("embeddings", self.force_rebuild) as cache:
            if cache.exists:
                logger.info("📂 Loading cached embeddings")
                return cache.load()
            
            logger.info("🔄 Running embedding generation...")
            
            # Choose encoder
            if self.encoder_type == 'auto':
                # Try HuggingFace first, fallback to TF-IDF
                try:
                    self.encoder = HuggingFaceEncoder()
                    self.encoder.fit(self.clean_data['normalized_name'].tolist())
                    logger.info("🤖 Using HuggingFace transformer encoder")
                except Exception as e:
                    logger.warning(f"HuggingFace encoder failed: {e}")
                    self.encoder = TfidfEncoder()
                    self.encoder.fit(self.clean_data['normalized_name'].tolist())
                    logger.info("🔤 Using TF-IDF encoder")
            elif self.encoder_type == 'hf':
                self.encoder = HuggingFaceEncoder()
                self.encoder.fit(self.clean_data['normalized_name'].tolist())
            elif self.encoder_type == 'tfidf':
                self.encoder = TfidfEncoder()
                self.encoder.fit(self.clean_data['normalized_name'].tolist())
            else:
                raise ValueError(f"Unknown encoder type: {self.encoder_type}")
            
            # Generate embeddings
            embeddings = self.encoder.encode(
                self.clean_data['normalized_name'].tolist(),
                batch_size=EMBEDDING_BATCH_SIZE
            )
            
            logger.info(f"✅ Embeddings generated: {embeddings.shape}")
            cache.save(embeddings)
            return embeddings
    
    def _run_clustering(self) -> np.ndarray:
        """Step 4: Cluster embeddings."""
        with smart_cache("cluster_labels", self.force_rebuild) as cache:
            if cache.exists:
                logger.info("📂 Loading cached cluster labels")
                labels = cache.load()
                return labels.values if hasattr(labels, 'values') else labels
            
            logger.info("🔄 Running clustering...")
            
            # Choose clusterer
            if self.clusterer_type == 'faiss':
                self.clusterer = FaissClusterer(
                    similarity_threshold=SIMILARITY_THRESHOLD,
                    min_cluster_size=MIN_CLUSTER_SIZE,
                    max_clusters=MAX_CLUSTERS
                )
            elif self.clusterer_type == 'hdbscan':
                self.clusterer = HdbscanClusterer(
                    min_cluster_size=MIN_CLUSTER_SIZE
                )
            else:
                raise ValueError(f"Unknown clusterer type: {self.clusterer_type}")
            
            # Fit and predict
            cluster_labels = self.clusterer.fit_predict(
                self.embeddings, 
                self.clean_data['normalized_name'].tolist()
            )
            
            # Log clustering results
            info = self.clusterer.get_cluster_info()
            logger.info(f"✅ Clustering complete: {info['n_clusters']} clusters")
            logger.info(f"📊 Largest cluster: {info['largest_cluster_size']} items")
            logger.info(f"📊 Average cluster size: {info['average_cluster_size']:.1f}")
            
            # Save as DataFrame for caching
            labels_df = pd.DataFrame({'cluster_id': cluster_labels})
            cache.save(labels_df)
            return cluster_labels
    
    def _run_categorization(self) -> pd.DataFrame:
        """Step 5: Assign clusters to categories."""
        with smart_cache("category_analysis", self.force_rebuild) as cache:
            if cache.exists:
                logger.info("📂 Loading cached category analysis")
                return cache.load()
            
            logger.info("🔄 Running category assignment...")
            
            # Prepare data for analysis
            self.clean_data['cluster_id'] = self.cluster_labels
            
            # Initialize mapper
            self.mapper = AutoClusterMapper(
                main_categories=self.main_categories,
                confidence_threshold=CATEGORY_CONFIDENCE_THRESHOLD
            )
            
            # Analyze and assign
            analysis_df = self.mapper.analyze_clusters(
                self.clean_data,
                self.embeddings,
                name_column='name',
                cluster_column='cluster_id'
            )
            
            # Log results
            category_summary = self.mapper.get_category_summary(analysis_df)
            logger.info("✅ Category assignment complete")
            
            for _, row in category_summary.iterrows():
                logger.info(f"📂 {row['category']}: {row['total_items']} items "
                           f"({row['percentage']:.1f}%), confidence: {row['avg_confidence']:.2f}")
            
            cache.save(analysis_df)
            return analysis_df
    
    def _prepare_final_results(self) -> pd.DataFrame:
        """Prepare final results for output."""
        # Merge original data with cluster and category info
        results = self.clean_data.copy()
        
        # Add category assignments
        cluster_to_category = dict(zip(
            self.analysis_df['cluster_id'],
            self.analysis_df['category']
        ))
        cluster_to_confidence = dict(zip(
            self.analysis_df['cluster_id'],
            self.analysis_df['confidence']
        ))
        
        results['category'] = results['cluster_id'].map(cluster_to_category)
        results['category_confidence'] = results['cluster_id'].map(cluster_to_confidence)
        
        # Clean up columns
        if not INCLUDE_CONFIDENCE:
            results = results.drop('category_confidence', axis=1)
        
        return results
    
    def _log_pipeline_summary(self, elapsed_time: float):
        """Log pipeline execution summary."""
        logger.info("🎉 Pipeline execution complete!")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total time: {elapsed_time:.1f} seconds")
        logger.info(f"📊 Processed: {len(self.clean_data):,} items")
        logger.info(f"🎯 Found: {len(self.analysis_df)} clusters")
        logger.info(f"📂 Categories: {len(self.main_categories)}")
        
        # Category breakdown
        if self.analysis_df is not None:
            summary = self.mapper.get_category_summary(self.analysis_df)
            logger.info("\n📈 Final Category Distribution:")
            for _, row in summary.iterrows():
                logger.info(f"  {row['category']}: {row['total_items']} items ({row['percentage']:.1f}%)")
        
        logger.info("=" * 60)

def main():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Product Categorization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline_runner --csv data/products.csv --categories Tables Chairs Computers
  python -m src.pipeline_runner --csv data/products.csv --categories Tables Chairs --encoder tfidf --force
        """
    )
    
    # Required arguments
    parser.add_argument('--csv', required=True, help='Input CSV file path')
    parser.add_argument('--categories', nargs='+', required=True, 
                       help='Main categories (e.g., Tables Chairs Computers)')
    
    # Optional arguments
    parser.add_argument('--output', help='Output file path (auto-generated if not specified)')
    parser.add_argument('--encoder', choices=['auto', 'hf', 'tfidf'], default='auto',
                       help='Encoder type (default: auto)')
    parser.add_argument('--clusterer', choices=['faiss', 'hdbscan'], default='faiss',
                       help='Clusterer type (default: faiss)')
    parser.add_argument('--force', action='store_true', 
                       help='Force rebuild all cached artifacts')
    
    # Configuration overrides
    parser.add_argument('--similarity-threshold', type=float, default=SIMILARITY_THRESHOLD,
                       help=f'Similarity threshold (default: {SIMILARITY_THRESHOLD})')
    parser.add_argument('--min-cluster-size', type=int, default=MIN_CLUSTER_SIZE,
                       help=f'Minimum cluster size (default: {MIN_CLUSTER_SIZE})')
    parser.add_argument('--max-rows', type=int, help='Maximum rows to process')
    
    args = parser.parse_args()
    
    # Override configuration
    override_config(
        SIMILARITY_THRESHOLD=args.similarity_threshold,
        MIN_CLUSTER_SIZE=args.min_cluster_size,
        MAX_ROWS=args.max_rows
    )
    
    # Initialize pipeline
    pipeline = ProductCategorizationPipeline(
        main_categories=args.categories,
        encoder_type=args.encoder,
        clusterer_type=args.clusterer,
        force_rebuild=args.force
    )
    
    # Generate output path if not specified
    output_path = args.output
    if not output_path:
        csv_stem = Path(args.csv).stem
        output_path = f"artifacts/{csv_stem}_categorized.{OUTPUT_FORMAT}"
    
    try:
        # Run pipeline
        results = pipeline.run(args.csv, output_path)
        
        # Save metadata
        metadata = {
            'input_file': args.csv,
            'categories': args.categories,
            'encoder': args.encoder,
            'clusterer': args.clusterer,
            'total_items': len(pipeline.clean_data),
            'total_clusters': len(results),
            'execution_time': time.time()
        }
        save_metadata(metadata, f"{Path(output_path).stem}_metadata")
        
        print(f"\n✅ Success! Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
