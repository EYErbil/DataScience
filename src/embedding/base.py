"""
Base class for all embedding encoders.
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class BaseEncoder(ABC):
    """Abstract base class for text embedding encoders."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, texts: List[str]) -> 'BaseEncoder':
        """Fit the encoder on a corpus of texts."""
        pass
    
    @abstractmethod
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into embeddings."""
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        pass
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text."""
        return self.encode([text])[0]
    
    def save_model(self, path: str):
        """Save the fitted model."""
        raise NotImplementedError("Subclasses should implement save_model")
    
    def load_model(self, path: str):
        """Load a fitted model."""
        raise NotImplementedError("Subclasses should implement load_model")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model_name='{self.model_name}', fitted={self.is_fitted})"
