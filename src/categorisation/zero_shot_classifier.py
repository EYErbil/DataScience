"""
Zero-shot classification using pre-trained LLMs.
Implements Approach 4 from the ChatGPT roadmap.
"""
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class ZeroShotClassifier:
    """
    Zero-shot classifier using BART-large MNLI or similar models.
    Can classify products into categories without training data.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.model_name = model_name
        self.classifier = None
        self._setup_classifier()
    
    def _setup_classifier(self):
        """Setup the zero-shot classification pipeline."""
        try:
            from transformers import pipeline
            
            logger.info(f"🤖 Loading zero-shot classifier: {self.model_name}")
            
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if self._check_gpu() else -1
            )
            
            logger.info("✅ Zero-shot classifier loaded successfully")
            
        except ImportError:
            logger.error("transformers not installed. Install with: pip install transformers")
            self.classifier = None
        except Exception as e:
            logger.error(f"Failed to load zero-shot classifier: {e}")
            self.classifier = None
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def classify_batch(self, 
                      texts: List[str], 
                      candidate_labels: List[str],
                      batch_size: int = 32) -> List[Dict]:
        """
        Classify a batch of texts using zero-shot classification.
        
        Args:
            texts: List of product names to classify
            candidate_labels: List of category names
            batch_size: Batch size for processing
            
        Returns:
            List of classification results with scores
        """
        if not self.classifier:
            logger.error("Zero-shot classifier not available")
            return [{'labels': candidate_labels, 'scores': [0.0] * len(candidate_labels)} for _ in texts]
        
        results = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            logger.info(f"🔍 Zero-shot classifying batch {i//batch_size + 1}: {len(batch_texts)} items")
            
            try:
                # Use transformers pipeline with explicit truncation to avoid long inputs
                batch_results = self.classifier(batch_texts, candidate_labels, truncation=True)
                
                # Ensure results are in list format (single item vs batch)
                if not isinstance(batch_results, list):
                    batch_results = [batch_results]
                
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Zero-shot classification failed for batch: {e}")
                # Return dummy results for failed batch
                dummy_results = [
                    {'labels': candidate_labels, 'scores': [1.0/len(candidate_labels)] * len(candidate_labels)}
                    for _ in batch_texts
                ]
                results.extend(dummy_results)
        
        return results
    
    def classify_single(self, text: str, candidate_labels: List[str]) -> Dict:
        """Classify a single text."""
        results = self.classify_batch([text], candidate_labels, batch_size=1)
        return results[0] if results else {'labels': candidate_labels, 'scores': [0.0] * len(candidate_labels)}
    
    def get_best_category(self, text: str, candidate_labels: List[str], threshold: float = 0.3) -> Tuple[str, float]:
        """
        Get the best category for a text with confidence threshold.
        
        Args:
            text: Product name to classify
            candidate_labels: List of category names
            threshold: Minimum confidence threshold
            
        Returns:
            Tuple of (category, confidence_score)
        """
        result = self.classify_single(text, candidate_labels)
        
        if result['scores'] and result['labels']:
            best_label = result['labels'][0]
            best_score = result['scores'][0]
            
            if best_score >= threshold:
                return best_label, best_score
        
        return 'Unclassified', 0.0
    
    def enhance_cluster_classification(self, 
                                     cluster_representatives: List[str],
                                     candidate_labels: List[str],
                                     confidence_threshold: float = 0.5) -> Dict[int, Tuple[str, float]]:
        """
        Use zero-shot classification to enhance cluster-to-category mapping.
        
        Args:
            cluster_representatives: Representative names for each cluster
            candidate_labels: Category names
            confidence_threshold: Minimum confidence for assignment
            
        Returns:
            Dictionary mapping cluster_id to (category, confidence)
        """
        logger.info(f"🎯 Zero-shot enhancing {len(cluster_representatives)} cluster assignments")
        
        # Classify all cluster representatives
        results = self.classify_batch(cluster_representatives, candidate_labels)
        
        assignments = {}
        
        for cluster_id, (representative, result) in enumerate(zip(cluster_representatives, results)):
            if result['scores'] and result['labels']:
                best_category = result['labels'][0]
                best_score = result['scores'][0]
                
                # Apply confidence threshold
                if best_score >= confidence_threshold:
                    assignments[cluster_id] = (best_category, best_score)
                    logger.debug(f"Cluster {cluster_id} ('{representative}') → {best_category} ({best_score:.3f})")
                else:
                    assignments[cluster_id] = ('Unclassified', best_score)
            else:
                assignments[cluster_id] = ('Unclassified', 0.0)
        
        # Log summary
        assigned_categories = [cat for cat, conf in assignments.values() if cat != 'Unclassified']
        from collections import Counter
        category_counts = Counter(assigned_categories)
        
        logger.info(f"📊 Zero-shot assignments: {dict(category_counts)}")
        
        return assignments


class GPTClassifier:
    """
    GPT-based few-shot classifier using OpenAI API.
    Alternative zero-shot approach using GPT models.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup OpenAI client."""
        try:
            import openai
            
            if self.api_key:
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info(f"🤖 GPT classifier initialized: {self.model}")
            else:
                logger.warning("No OpenAI API key provided - GPT classifier unavailable")
                
        except ImportError:
            logger.error("openai not installed. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Failed to setup GPT client: {e}")
    
    def classify_with_prompt(self, 
                            product_name: str, 
                            categories: List[str],
                            few_shot_examples: Optional[List[Tuple[str, str]]] = None) -> Tuple[str, float]:
        """
        Classify using GPT with few-shot prompting.
        
        Args:
            product_name: Product name to classify
            categories: List of category names
            few_shot_examples: Optional list of (product, category) examples
            
        Returns:
            Tuple of (category, confidence)
        """
        if not self.client:
            logger.error("GPT client not available")
            return 'Unclassified', 0.0
        
        # Build prompt
        prompt = self._build_classification_prompt(product_name, categories, few_shot_examples)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert product categorization assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse result and assign confidence
            for category in categories:
                if category.lower() in result.lower():
                    return category, 0.9  # High confidence for GPT
            
            return 'Unclassified', 0.0
            
        except Exception as e:
            logger.error(f"GPT classification failed: {e}")
            return 'Unclassified', 0.0
    
    def _build_classification_prompt(self, 
                                   product_name: str, 
                                   categories: List[str],
                                   examples: Optional[List[Tuple[str, str]]] = None) -> str:
        """Build few-shot classification prompt."""
        prompt_parts = [
            "Assign the following product to one of these categories:",
            f"Categories: {', '.join(categories)}",
            ""
        ]
        
        # Add few-shot examples if provided
        if examples:
            prompt_parts.append("Examples:")
            for product, category in examples:
                prompt_parts.append(f"Product: '{product}' → Category: {category}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            f"Product: '{product_name}'",
            "Category:"
        ])
        
        return "\n".join(prompt_parts)


def create_hybrid_classifier(main_categories: List[str], 
                           use_gpt: bool = False,
                           openai_api_key: Optional[str] = None) -> object:
    """
    Create a hybrid classifier that combines zero-shot and traditional methods.
    
    Args:
        main_categories: List of category names
        use_gpt: Whether to use GPT as backup
        openai_api_key: OpenAI API key for GPT
        
    Returns:
        Configured classifier instance
    """
    # Primary: BART zero-shot classifier
    bart_classifier = ZeroShotClassifier()
    
    # Backup: GPT classifier (if requested and API key available)
    gpt_classifier = None
    if use_gpt and openai_api_key:
        gpt_classifier = GPTClassifier(api_key=openai_api_key)
    
    class HybridClassifier:
        def __init__(self):
            self.bart = bart_classifier
            self.gpt = gpt_classifier
            self.categories = main_categories
        
        def classify(self, text: str, threshold: float = 0.5) -> Tuple[str, float]:
            """Classify using best available method."""
            # Try BART first
            if self.bart.classifier:
                category, confidence = self.bart.get_best_category(text, self.categories, threshold)
                if category != 'Unclassified':
                    return category, confidence
            
            # Fallback to GPT if available and BART failed
            if self.gpt and self.gpt.client:
                logger.info("🔄 Falling back to GPT classification")
                return self.gpt.classify_with_prompt(text, self.categories)
            
            return 'Unclassified', 0.0
        
        def classify_batch(self, texts: List[str], threshold: float = 0.5) -> List[Tuple[str, float]]:
            """Classify batch of texts."""
            if self.bart.classifier:
                results = self.bart.classify_batch(texts, self.categories)
                return [
                    (result['labels'][0], result['scores'][0]) 
                    if result['scores'][0] >= threshold 
                    else ('Unclassified', result['scores'][0])
                    for result in results
                ]
            
            # Fallback to individual GPT calls
            return [self.classify(text, threshold) for text in texts]
    
    return HybridClassifier()
