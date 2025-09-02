"""
Hugging Face transformer-based encoder with SSL bypass and batching.
"""
import numpy as np
from typing import List, Optional
import logging
import warnings
import os
import ssl

# Handle both relative and absolute imports
try:
    from .base import BaseEncoder
    from ..config import (
        EMBEDDING_MODEL, EMBEDDING_FALLBACK, EMBEDDING_BATCH_SIZE,
        SSL_BYPASS, DOWNLOAD_RETRIES, DOWNLOAD_TIMEOUT, USE_GPU
    )
except ImportError:
    from embedding.base import BaseEncoder
    from config import (
        EMBEDDING_MODEL, EMBEDDING_FALLBACK, EMBEDDING_BATCH_SIZE,
        SSL_BYPASS, DOWNLOAD_RETRIES, DOWNLOAD_TIMEOUT, USE_GPU
    )

logger = logging.getLogger(__name__)

class HuggingFaceEncoder(BaseEncoder):
    """Transformer-based encoder using Sentence Transformers."""
    
    def __init__(self, model_name: str = None, device: str = None):
        super().__init__(model_name or EMBEDDING_MODEL)
        # Always use CPU for better compatibility
        self.device = device or 'cpu'
        self.model = None
        self._setup_ssl_bypass()
        
    def _setup_ssl_bypass(self):
        """Setup aggressive SSL bypass for corporate networks."""
        if SSL_BYPASS:
            # Disable SSL warnings
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                warnings.filterwarnings('ignore', message='Unverified HTTPS request')
            except ImportError:
                pass
            
            # Set environment variables for SSL bypass
            os.environ['CURL_CA_BUNDLE'] = ''
            os.environ['REQUESTS_CA_BUNDLE'] = ''
            os.environ['SSL_VERIFY'] = 'false'
            os.environ['PYTHONHTTPSVERIFY'] = '0'
            os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Don't force offline mode
            
            # Create unverified SSL context
            ssl._create_default_https_context = ssl._create_unverified_context
            
            # Patch requests to disable SSL verification
            try:
                import requests
                requests.packages.urllib3.disable_warnings()
                # Monkey patch requests
                original_request = requests.Session.request
                def patched_request(self, *args, **kwargs):
                    kwargs['verify'] = False
                    return original_request(self, *args, **kwargs)
                requests.Session.request = patched_request
                
                # Also patch the module-level functions
                original_get = requests.get
                original_post = requests.post
                def patched_get(*args, **kwargs):
                    kwargs['verify'] = False
                    return original_get(*args, **kwargs)
                def patched_post(*args, **kwargs):
                    kwargs['verify'] = False
                    return original_post(*args, **kwargs)
                requests.get = patched_get
                requests.post = patched_post
                
            except ImportError:
                pass
            
            # Setup urllib3 poolmanager to disable SSL
            try:
                import urllib3.util.ssl_
                urllib3.util.ssl_.DEFAULT_CIPHERS = 'ALL'
                urllib3.util.ssl_.DEFAULT_CIPHERS += ':!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA'
            except Exception:
                pass
            
            logger.info("🔓 Aggressive SSL bypass enabled for corporate networks")
    
    def _try_load_model(self, model_name: str) -> bool:
        """Try to load a specific model with retries."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Additional HuggingFace Hub SSL bypass
            if SSL_BYPASS:
                try:
                    import huggingface_hub
                    # Disable SSL for HuggingFace Hub
                    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
                    os.environ['HF_HUB_OFFLINE'] = '0'
                    
                    # Patch huggingface_hub requests
                    if hasattr(huggingface_hub, 'utils'):
                        original_get_session = getattr(huggingface_hub.utils, 'get_session', None)
                        if original_get_session:
                            def patched_get_session():
                                session = original_get_session()
                                session.verify = False
                                return session
                            huggingface_hub.utils.get_session = patched_get_session
                except ImportError:
                    pass
            
            logger.info(f"🤖 Attempting to load model: {model_name}")
            
            # Try loading with retries
            for attempt in range(DOWNLOAD_RETRIES):
                try:
                    # Force download with no SSL verification
                    self.model = SentenceTransformer(
                        model_name,
                        device=self.device,
                        trust_remote_code=True,
                        use_auth_token=False
                    )
                    logger.info(f"✅ Successfully loaded {model_name}")
                    return True
                    
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == DOWNLOAD_RETRIES - 1:
                        raise e
                    
        except ImportError:
            logger.error("sentence-transformers not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return False
    
    def fit(self, texts: List[str]) -> 'HuggingFaceEncoder':
        """Fit the encoder (load the model)."""
        if self.is_fitted:
            return self
            
        # Try primary model first
        if self._try_load_model(self.model_name):
            self.is_fitted = True
            return self
            
        # Try fallback model
        logger.warning(f"Primary model failed, trying fallback: {EMBEDDING_FALLBACK}")
        if self._try_load_model(EMBEDDING_FALLBACK):
            self.model_name = EMBEDDING_FALLBACK
            self.is_fitted = True
            return self
            
        # If all models fail, raise error
        raise RuntimeError("All transformer models failed to load")
    
    def encode(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Encode texts into embeddings with batching."""
        if not self.is_fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
            
        if not texts:
            return np.array([]).reshape(0, self.get_embedding_dim())
            
        batch_size = batch_size or EMBEDDING_BATCH_SIZE
        
        logger.info(f"🔢 Encoding {len(texts):,} texts in batches of {batch_size}")
        
        try:
            # Use sentence-transformers batching
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 1000,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            logger.info(f"✅ Generated embeddings: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if not self.is_fitted:
            return 0
        # Sentence transformers don't expose vocab size directly
        return getattr(self.model.tokenizer, 'vocab_size', 30000)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        if not self.is_fitted:
            return 0
        return self.model.get_sentence_embedding_dimension()
    
    def save_model(self, path: str):
        """Save the model to disk."""
        if not self.is_fitted:
            raise RuntimeError("No model to save")
        self.model.save(path)
        logger.info(f"💾 Saved model to {path}")
    
    def load_model(self, path: str):
        """Load model from disk."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(path, device=self.device)
            self.is_fitted = True
            logger.info(f"📂 Loaded model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            raise
    
    def get_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between embeddings."""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(embeddings1, embeddings2)
