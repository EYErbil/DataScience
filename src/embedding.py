"""
Text embedding generation for semantic similarity calculation.
Uses sentence transformers for multilingual embeddings.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)


class NameEmbedder:
    """
    Generates embeddings for product names using sentence transformers.
    Supports multilingual models for cross-language similarity.
    """
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialize embedder with specified model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.texts = None
        self._use_tfidf_fallback = False
        self._tfidf_vectorizer = None
        
    def load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Try to disable SSL verification for corporate networks
            import ssl
            import certifi
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Set environment variables to bypass SSL issues
            import os
            os.environ['CURL_CA_BUNDLE'] = ''
            os.environ['REQUESTS_CA_BUNDLE'] = ''
            
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            
            # Try different approaches for SSL issues
            try:
                logger.info("Trying with SSL verification disabled...")
                import requests
                from requests.adapters import HTTPAdapter
                from urllib3.util.retry import Retry
                
                # Disable SSL warnings
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                
                # Try fallback model
                logger.info("Trying fallback model: all-MiniLM-L6-v2")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                
            except Exception as e2:
                logger.error(f"All models failed to load: {e2}")
                logger.info("Using simple TF-IDF fallback for embeddings")
                self.model = None
                self._use_tfidf_fallback = True
    
    def _generate_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Fallback method using TF-IDF when transformer models fail.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of TF-IDF embeddings
        """
        logger.info("Using TF-IDF fallback for embeddings")
        
        if self._tfidf_vectorizer is None:
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,  # More features for better discrimination
                ngram_range=(1, 2),  # 1-2 grams work better for semantic similarity
                stop_words=None,  # Don't remove stop words for multilingual support
                lowercase=True,
                min_df=1,  # Keep even single-occurrence terms
                max_df=0.7,  # Remove very common terms (more aggressive)
                norm='l2',  # L2 normalization for better cosine similarity
                token_pattern=r'\b[a-zA-ZÀ-ÿ]+\b'  # Include accented characters
            )
        
        # Fit and transform texts
        embeddings = self._tfidf_vectorizer.fit_transform(texts).toarray()
        logger.info(f"Generated TF-IDF embeddings with shape: {embeddings.shape}")
        
        return embeddings
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        if self.model is None and not self._use_tfidf_fallback:
            self.load_model()
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Convert to strings and clean
        clean_texts = [str(text).strip() for text in texts if text and str(text).strip()]
        
        if not clean_texts:
            raise ValueError("No valid texts provided for embedding")
        
        # Generate embeddings based on available method
        if self._use_tfidf_fallback or self.model is None:
            embeddings = self._generate_tfidf_embeddings(clean_texts)
        else:
            try:
                embeddings = self.model.encode(
                    clean_texts,
                    batch_size=batch_size,
                    show_progress_bar=len(clean_texts) > 100,
                    convert_to_numpy=True
                )
            except Exception as e:
                logger.error(f"Transformer model failed, falling back to TF-IDF: {e}")
                self._use_tfidf_fallback = True
                embeddings = self._generate_tfidf_embeddings(clean_texts)
        
        self.texts = clean_texts
        self.embeddings = embeddings
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def calculate_similarity_matrix(self, embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate pairwise cosine similarity matrix.
        
        Args:
            embeddings: Embeddings array (uses self.embeddings if None)
            
        Returns:
            Similarity matrix with shape (n_texts, n_texts)
        """
        if embeddings is None:
            if self.embeddings is None:
                raise ValueError("No embeddings available. Call generate_embeddings() first.")
            embeddings = self.embeddings
        
        similarity_matrix = cosine_similarity(embeddings)
        logger.info(f"Calculated similarity matrix with shape: {similarity_matrix.shape}")
        
        return similarity_matrix
    
    def find_similar_pairs(self, similarity_threshold: float = 0.7,
                          exclude_identical: bool = True) -> List[Tuple[int, int, float]]:
        """
        Find pairs of texts with similarity above threshold.
        
        Args:
            similarity_threshold: Minimum similarity score
            exclude_identical: Whether to exclude identical matches
            
        Returns:
            List of tuples (index1, index2, similarity_score)
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Call generate_embeddings() first.")
        
        similarity_matrix = self.calculate_similarity_matrix()
        similar_pairs = []
        
        n_texts = len(self.texts)
        for i in range(n_texts):
            for j in range(i + 1, n_texts):
                similarity = similarity_matrix[i, j]
                
                if similarity >= similarity_threshold:
                    # Check if texts are identical (if excluding)
                    if exclude_identical and self.texts[i].lower() == self.texts[j].lower():
                        continue
                    
                    similar_pairs.append((i, j, similarity))
        
        # Sort by similarity score (descending)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"Found {len(similar_pairs)} similar pairs above threshold {similarity_threshold}")
        return similar_pairs
    
    def get_similarity_dataframe(self, similarity_threshold: float = 0.7) -> pd.DataFrame:
        """
        Get similarity results as a pandas DataFrame.
        
        Args:
            similarity_threshold: Minimum similarity score
            
        Returns:
            DataFrame with columns: text1, text2, similarity
        """
        similar_pairs = self.find_similar_pairs(similarity_threshold)
        
        data = []
        for i, j, similarity in similar_pairs:
            data.append({
                'text1': self.texts[i],
                'text2': self.texts[j],
                'similarity': similarity,
                'index1': i,
                'index2': j
            })
        
        return pd.DataFrame(data)
    
    def find_most_similar(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar texts to a query.
        
        Args:
            query_text: Text to find similarities for
            top_k: Number of top similar texts to return
            
        Returns:
            List of tuples (text, similarity_score)
        """
        if self.model is None and not self._use_tfidf_fallback:
            self.load_model()
        
        if self.embeddings is None:
            raise ValueError("No embeddings available. Call generate_embeddings() first.")
        
        # Embed the query
        if self._use_tfidf_fallback or self.model is None:
            if self._tfidf_vectorizer is None:
                raise ValueError("TF-IDF vectorizer not fitted. Call generate_embeddings() first.")
            query_embedding = self._tfidf_vectorizer.transform([query_text]).toarray()
        else:
            try:
                query_embedding = self.model.encode([query_text])
            except Exception as e:
                logger.error(f"Query embedding failed: {e}")
                if self._tfidf_vectorizer is not None:
                    query_embedding = self._tfidf_vectorizer.transform([query_text]).toarray()
                else:
                    raise
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(self.texts[idx], similarities[idx]) for idx in top_indices]
        return results


class SimilarityAnalyzer:
    """
    Analyzes similarity patterns in product names for clustering insights.
    """
    
    def __init__(self, embedder: NameEmbedder):
        """
        Initialize analyzer with an embedder.
        
        Args:
            embedder: NameEmbedder instance with generated embeddings
        """
        self.embedder = embedder
    
    def analyze_similarity_distribution(self) -> Dict:
        """
        Analyze the distribution of similarity scores.
        
        Returns:
            Dictionary with distribution statistics
        """
        if self.embedder.embeddings is None:
            raise ValueError("No embeddings available in the embedder.")
        
        similarity_matrix = self.embedder.calculate_similarity_matrix()
        
        # Get upper triangle (excluding diagonal)
        n = similarity_matrix.shape[0]
        upper_triangle = similarity_matrix[np.triu_indices(n, k=1)]
        
        stats = {
            'mean_similarity': np.mean(upper_triangle),
            'std_similarity': np.std(upper_triangle),
            'median_similarity': np.median(upper_triangle),
            'min_similarity': np.min(upper_triangle),
            'max_similarity': np.max(upper_triangle),
            'q25': np.percentile(upper_triangle, 25),
            'q75': np.percentile(upper_triangle, 75),
            'n_pairs': len(upper_triangle)
        }
        
        return stats
    
    def suggest_clustering_threshold(self, target_clusters_ratio: float = 0.1) -> float:
        """
        Suggest a similarity threshold for clustering based on distribution.
        
        Args:
            target_clusters_ratio: Target ratio of similar pairs to total pairs
            
        Returns:
            Suggested threshold value
        """
        stats = self.analyze_similarity_distribution()
        
        # Start with 75th percentile as a reasonable threshold
        suggested_threshold = stats['q75']
        
        # Adjust based on distribution characteristics
        if stats['std_similarity'] > 0.15:  # High variance
            suggested_threshold = stats['q75'] + 0.05
        elif stats['mean_similarity'] > 0.6:  # Generally high similarity
            suggested_threshold = stats['mean_similarity'] + stats['std_similarity']
        
        # Ensure reasonable bounds
        suggested_threshold = max(0.5, min(0.9, suggested_threshold))
        
        logger.info(f"Suggested clustering threshold: {suggested_threshold:.3f}")
        return suggested_threshold
    
    def create_similarity_report(self, similarity_threshold: float = None) -> str:
        """
        Create a detailed similarity analysis report.
        
        Args:
            similarity_threshold: Threshold for similarity analysis
            
        Returns:
            Formatted report string
        """
        stats = self.analyze_similarity_distribution()
        
        if similarity_threshold is None:
            similarity_threshold = self.suggest_clustering_threshold()
        
        similar_pairs = self.embedder.find_similar_pairs(similarity_threshold)
        
        report = []
        report.append("=" * 60)
        report.append("SIMILARITY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Total texts analyzed: {len(self.embedder.texts):,}")
        report.append(f"Total possible pairs: {stats['n_pairs']:,}")
        report.append("")
        report.append("Similarity Statistics:")
        report.append(f"  Mean similarity: {stats['mean_similarity']:.3f}")
        report.append(f"  Std deviation: {stats['std_similarity']:.3f}")
        report.append(f"  Median similarity: {stats['median_similarity']:.3f}")
        report.append(f"  25th percentile: {stats['q25']:.3f}")
        report.append(f"  75th percentile: {stats['q75']:.3f}")
        report.append(f"  Min similarity: {stats['min_similarity']:.3f}")
        report.append(f"  Max similarity: {stats['max_similarity']:.3f}")
        report.append("")
        report.append(f"Using threshold: {similarity_threshold:.3f}")
        report.append(f"Similar pairs found: {len(similar_pairs):,}")
        report.append(f"Percentage of similar pairs: {len(similar_pairs)/stats['n_pairs']*100:.2f}%")
        report.append("")
        
        if similar_pairs:
            report.append("Top 10 Most Similar Pairs:")
            for i, (idx1, idx2, sim) in enumerate(similar_pairs[:10]):
                text1 = self.embedder.texts[idx1]
                text2 = self.embedder.texts[idx2]
                report.append(f"  {i+1:2d}. {sim:.3f} | {text1:<25} <-> {text2}")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def demo_embeddings(sample_texts: List[str] = None):
    """
    Demo function showing embedding generation and similarity analysis.
    
    Args:
        sample_texts: List of texts to analyze (uses default if None)
    """
    if sample_texts is None:
        sample_texts = [
            "office table",
            "desk for office",
            "çalışma masası",
            "office chair",
            "computer chair",
            "sandalye",
            "gaming chair",
            "laptop computer",
            "desktop pc",
            "bilgisayar",
            "led lamp",
            "table lamp",
            "masa lambası",
            "book shelf",
            "bookcase",
            "kitap rafı"
        ]
    
    print("=" * 60)
    print("EMBEDDING GENERATION DEMO")
    print("=" * 60)
    
    # Initialize embedder
    embedder = NameEmbedder()
    
    print(f"Sample texts ({len(sample_texts)}):")
    for i, text in enumerate(sample_texts):
        print(f"  {i+1:2d}. {text}")
    
    print("\nGenerating embeddings...")
    embeddings = embedder.generate_embeddings(sample_texts)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Analyze similarities
    analyzer = SimilarityAnalyzer(embedder)
    
    print("\n" + analyzer.create_similarity_report())
    
    # Find similar to a query
    query = "office desk"
    print(f"\nMost similar to '{query}':")
    similar = embedder.find_most_similar(query, top_k=5)
    for i, (text, sim) in enumerate(similar):
        print(f"  {i+1}. {sim:.3f} | {text}")


if __name__ == "__main__":
    demo_embeddings()
