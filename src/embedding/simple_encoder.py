"""
Simple, reliable encoder that works in corporate environments.
Falls back gracefully from HuggingFace to TF-IDF.
"""

import numpy as np
from typing import List
import logging
import os

try:
    from .base import BaseEncoder
except ImportError:
    from embedding.base import BaseEncoder

logger = logging.getLogger(__name__)

class SimpleEncoder(BaseEncoder):
    """Simple encoder with HuggingFace first, TF-IDF fallback."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.model = None
        self.encoder_type = "unknown"
        
    def fit(self, texts: List[str]) -> 'SimpleEncoder':
        """Try HuggingFace first, fall back to TF-IDF."""
        if self.is_fitted:
            return self
            
        # Try HuggingFace (if models are cached locally)
        if self._try_huggingface(texts):
            return self
            
        # Fall back to TF-IDF
        logger.info("🔄 Using TF-IDF fallback")
        self._setup_tfidf(texts)
        return self
    
    def _try_huggingface(self, texts: List[str]) -> bool:
        """Try to use HuggingFace if models are cached."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Only try if offline mode or models are cached
            logger.info("🤖 Checking for cached HuggingFace models...")
            
            # Try to load without network access
            model = SentenceTransformer(self.model_name, device='cpu')
            
            # Test if it works
            test_emb = model.encode(['test'], convert_to_numpy=True)
            if test_emb.shape[1] > 0:
                self.model = model
                self.encoder_type = "huggingface"
                self.is_fitted = True
                logger.info(f"✅ Using cached HuggingFace model: {self.model_name}")
                return True
                
        except Exception as e:
            logger.info(f"HuggingFace not available: {e}")
            
        return False
    
    def _setup_tfidf(self, texts: List[str]):
        """Setup TF-IDF encoder."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.model = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            stop_words='english',
            norm='l2'
        )
        
        self.model.fit(texts)
        self.encoder_type = "tfidf"
        self.is_fitted = True
        logger.info(f"✅ TF-IDF encoder ready: {len(self.model.vocabulary_)} features")
    
    def encode(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Encode texts using the fitted model."""
        if not self.is_fitted:
            raise RuntimeError("Encoder not fitted")
            
        if self.encoder_type == "huggingface":
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        elif self.encoder_type == "tfidf":
            return self.model.transform(texts).toarray()
        else:
            raise RuntimeError("Unknown encoder type")
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        if not self.is_fitted:
            return 0
            
        if self.encoder_type == "huggingface":
            return self.model.get_sentence_embedding_dimension()
        elif self.encoder_type == "tfidf":
            return len(self.model.vocabulary_)
        else:
            return 0
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if not self.is_fitted:
            return 0
            
        if self.encoder_type == "huggingface":
            return getattr(self.model.tokenizer, 'vocab_size', 30000)
        elif self.encoder_type == "tfidf":
            return len(self.model.vocabulary_)
        else:
            return 0
    
    def get_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity."""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(embeddings1, embeddings2)
