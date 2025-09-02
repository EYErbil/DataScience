"""
TF-IDF based encoder as a fast fallback option.
"""
import numpy as np
from typing import List, Optional
import logging
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Handle both relative and absolute imports
try:
    from .base import BaseEncoder
    from ..config import (
        TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MAX_DF, TFIDF_MIN_DF
    )
except ImportError:
    from embedding.base import BaseEncoder
    from config import (
        TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MAX_DF, TFIDF_MIN_DF
    )

logger = logging.getLogger(__name__)

class TfidfEncoder(BaseEncoder):
    """TF-IDF based encoder for fast semantic embeddings."""
    
    def __init__(self, 
                 max_features: int = None,
                 ngram_range: tuple = None,
                 max_df: float = None,
                 min_df: int = None):
        super().__init__("tfidf")
        
        # Configuration
        self.max_features = max_features or TFIDF_MAX_FEATURES
        self.ngram_range = ngram_range or TFIDF_NGRAM_RANGE
        self.max_df = max_df or TFIDF_MAX_DF
        self.min_df = min_df or TFIDF_MIN_DF
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            max_df=self.max_df,
            min_df=self.min_df,
            stop_words=None,  # We'll handle stop words in preprocessing
            lowercase=True,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            token_pattern=r'\b[a-zA-ZÀ-ÿ]+\b',  # Multilingual support
            strip_accents='unicode'
        )
        
        logger.info(f"🔤 Initialized TF-IDF encoder: "
                   f"max_features={self.max_features}, "
                   f"ngram_range={self.ngram_range}")
    
    def fit(self, texts: List[str]) -> 'TfidfEncoder':
        """Fit the TF-IDF vectorizer on the corpus."""
        if not texts:
            raise ValueError("Cannot fit on empty text list")
            
        logger.info(f"🎯 Fitting TF-IDF on {len(texts):,} texts")
        
        # Filter out empty texts
        clean_texts = [text for text in texts if text and text.strip()]
        
        if not clean_texts:
            raise ValueError("No valid texts found after filtering")
            
        # Fit the vectorizer
        self.vectorizer.fit(clean_texts)
        self.is_fitted = True
        
        # Log vocabulary statistics
        vocab_size = len(self.vectorizer.vocabulary_)
        feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info(f"✅ TF-IDF fitted: {vocab_size:,} features")
        logger.debug(f"Sample features: {list(feature_names[:10])}")
        
        return self
    
    def encode(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Encode texts into TF-IDF embeddings."""
        if not self.is_fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
            
        if not texts:
            return np.array([]).reshape(0, self.get_embedding_dim())
            
        logger.info(f"🔢 Encoding {len(texts):,} texts with TF-IDF")
        
        # Handle empty texts
        clean_texts = []
        empty_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                clean_texts.append(text)
            else:
                clean_texts.append(" ")  # Placeholder for empty text
                empty_indices.append(i)
        
        # Transform texts to sparse matrix
        sparse_matrix = self.vectorizer.transform(clean_texts)
        
        # Convert to dense array
        embeddings = sparse_matrix.toarray().astype(np.float32)
        
        # Zero out embeddings for empty texts
        for idx in empty_indices:
            embeddings[idx] = 0.0
            
        logger.info(f"✅ Generated TF-IDF embeddings: {embeddings.shape}")
        
        return embeddings
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if not self.is_fitted:
            return 0
        return len(self.vectorizer.vocabulary_)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        if not self.is_fitted:
            return self.max_features
        return len(self.vectorizer.get_feature_names_out())
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (vocabulary)."""
        if not self.is_fitted:
            return []
        return list(self.vectorizer.get_feature_names_out())
    
    def get_top_features(self, embedding: np.ndarray, top_k: int = 10) -> List[tuple]:
        """Get top features for a given embedding."""
        if not self.is_fitted:
            return []
            
        feature_names = self.get_feature_names()
        top_indices = np.argsort(embedding)[-top_k:][::-1]
        
        return [(feature_names[i], embedding[i]) for i in top_indices if embedding[i] > 0]
    
    def save_model(self, path: str):
        """Save the fitted vectorizer."""
        if not self.is_fitted:
            raise RuntimeError("No model to save")
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            
        logger.info(f"💾 Saved TF-IDF model to {path}")
    
    def load_model(self, path: str):
        """Load a fitted vectorizer."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
            
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)
            
        self.is_fitted = True
        logger.info(f"📂 Loaded TF-IDF model from {path}")
    
    def get_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between embeddings."""
        return cosine_similarity(embeddings1, embeddings2)
    
    def analyze_corpus(self, texts: List[str]) -> dict:
        """Analyze the corpus and return statistics."""
        if not self.is_fitted:
            self.fit(texts)
            
        # Get embeddings for analysis
        embeddings = self.encode(texts)
        
        # Compute statistics
        stats = {
            'n_documents': len(texts),
            'vocab_size': self.get_vocab_size(),
            'embedding_dim': self.get_embedding_dim(),
            'sparsity': 1 - (np.count_nonzero(embeddings) / embeddings.size),
            'mean_doc_length': np.mean([len(text.split()) for text in texts if text]),
            'top_features': []
        }
        
        # Get most important features across corpus
        mean_weights = np.mean(embeddings, axis=0)
        top_indices = np.argsort(mean_weights)[-20:][::-1]
        feature_names = self.get_feature_names()
        
        stats['top_features'] = [
            (feature_names[i], mean_weights[i]) 
            for i in top_indices if mean_weights[i] > 0
        ]
        
        return stats
