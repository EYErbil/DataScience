"""
Fully automated cluster-to-category mapping using Approach 2: Unsupervised Clustering with Word Embeddings.
Implements semantic similarity analysis and zero-shot classification fallback.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict
import logging
import re

# Handle both relative and absolute imports
try:
    from ..config import (
        CATEGORY_CONFIDENCE_THRESHOLD, TOP_WORDS_PER_CLUSTER, 
        CLUSTER_SAMPLES, STOP_WORDS, MIN_WORD_LENGTH
    )
    from .zero_shot_classifier import ZeroShotClassifier, create_hybrid_classifier
except ImportError:
    # Fallback for direct execution or notebook imports
    from config import (
        CATEGORY_CONFIDENCE_THRESHOLD, TOP_WORDS_PER_CLUSTER, 
        CLUSTER_SAMPLES, STOP_WORDS, MIN_WORD_LENGTH
    )
    from categorisation.zero_shot_classifier import ZeroShotClassifier, create_hybrid_classifier

logger = logging.getLogger(__name__)

class AutoClusterMapper:
    """
    Automatically maps clusters to user-defined categories using:
    1. **Approach 2**: Semantic embedding analysis (cosine similarity)
    2. **Approach 4**: Zero-shot classification fallback (BART/GPT)
    3. Word frequency patterns for interpretability
    
    Follows ChatGPT's roadmap for unsupervised clustering with word embeddings.
    """
    
    def __init__(self, 
                 main_categories: List[str],
                 confidence_threshold: float = None,
                 use_zero_shot: bool = True,
                 use_gpt_fallback: bool = False,
                 openai_api_key: Optional[str] = None):
        self.main_categories = main_categories
        self.confidence_threshold = confidence_threshold or CATEGORY_CONFIDENCE_THRESHOLD
        self.use_zero_shot = use_zero_shot
        
        # Analysis results
        self.cluster_analysis = {}
        self.category_assignments = {}
        self.category_centroids = {}
        
        # Prepare enriched labels for zero-shot and mapping back to base categories
        self.zero_shot_labels: List[str] = list(main_categories)
        self.label_to_category: Dict[str, str] = {cat: cat for cat in main_categories}
        try:
            from user_categories import CATEGORY_DESCRIPTIONS  # optional
            enriched: List[str] = []
            for cat in main_categories:
                desc = CATEGORY_DESCRIPTIONS.get(cat)
                if desc:
                    label = f"{cat}: {desc}"
                    enriched.append(label)
                    self.label_to_category[label] = cat
                else:
                    enriched.append(cat)
            if enriched:
                self.zero_shot_labels = enriched
        except Exception:
            # Descriptions not available; stick to plain names
            pass

        # Zero-shot classifier (Approach 4)
        self.zero_shot_classifier = None
        if use_zero_shot:
            try:
                # Initialize hybrid with enriched labels so batch classification uses them
                self.zero_shot_classifier = create_hybrid_classifier(
                    self.zero_shot_labels, use_gpt_fallback, openai_api_key
                )
                logger.info("🤖 Zero-shot classifier initialized")
            except Exception as e:
                logger.warning(f"Zero-shot classifier failed to initialize: {e}")
        
        logger.info(f"🎯 AutoClusterMapper initialized for categories: {main_categories}")
        logger.info(f"🔧 Using zero-shot enhancement: {self.zero_shot_classifier is not None}")
    
    def analyze_clusters(self, 
                        clustered_data: pd.DataFrame,
                        embeddings: np.ndarray,
                        name_column: str = 'name',
                        cluster_column: str = 'cluster_id') -> pd.DataFrame:
        """
        Analyze clusters and automatically assign to categories.
        
        Args:
            clustered_data: DataFrame with cluster assignments
            embeddings: Embeddings corresponding to each row
            name_column: Column name containing product names
            cluster_column: Column name containing cluster IDs
            
        Returns:
            DataFrame with cluster analysis and category assignments
        """
        logger.info(f"🔍 Analyzing {clustered_data[cluster_column].nunique()} clusters")
        
        # Step 1: Extract meaningful features from each cluster
        cluster_features = self._extract_cluster_features(clustered_data, name_column, cluster_column)
        
        # Step 2: Compute cluster embeddings (centroids)
        cluster_embeddings = self._compute_cluster_embeddings(clustered_data, embeddings, cluster_column)
        
        # Step 3: Use unsupervised learning to group clusters into categories
        category_assignments = self._assign_clusters_to_categories(
            cluster_features, cluster_embeddings
        )
        
        # Step 4: Build analysis DataFrame
        analysis_data = []
        
        for cluster_id in clustered_data[cluster_column].unique():
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_data = clustered_data[clustered_data[cluster_column] == cluster_id]
            features = cluster_features[cluster_id]
            
            category, confidence = category_assignments.get(cluster_id, ('Unclassified', 0.0))
            
            analysis_data.append({
                'cluster_id': cluster_id,
                'category': category,
                'confidence': confidence,
                'representative_name': features['representative_name'],
                'total_items': len(cluster_data),
                'unique_barcodes': cluster_data['barcode'].nunique() if 'barcode' in cluster_data.columns else len(cluster_data),
                'top_words': features['top_words'][:5],  # Top 5 words
                'sample_names': features['sample_names'][:3],  # 3 examples
                'word_diversity': features['word_diversity'],
                'cluster_embedding': cluster_embeddings.get(cluster_id)
            })
        
        analysis_df = pd.DataFrame(analysis_data)
        analysis_df = analysis_df.sort_values('total_items', ascending=False)
        
        # Store results
        self.cluster_analysis = cluster_features
        self.category_assignments = category_assignments
        
        logger.info(f"✅ Cluster analysis complete")
        return analysis_df
    
    def _extract_cluster_features(self, 
                                 clustered_data: pd.DataFrame,
                                 name_column: str,
                                 cluster_column: str) -> Dict[int, Dict]:
        """Extract meaningful features from each cluster."""
        cluster_features = {}
        
        for cluster_id in clustered_data[cluster_column].unique():
            if cluster_id == -1:  # Skip noise
                continue
                
            cluster_data = clustered_data[clustered_data[cluster_column] == cluster_id]
            names = cluster_data[name_column].tolist()
            
            # Extract meaningful words
            all_words = self._extract_meaningful_words(names)
            word_counts = Counter(all_words)
            
            # Get most common name as representative
            name_counts = Counter(names)
            representative_name = name_counts.most_common(1)[0][0]
            
            # Calculate word diversity (entropy-like measure)
            total_words = sum(word_counts.values())
            word_diversity = len(word_counts) / max(total_words, 1)
            
            cluster_features[cluster_id] = {
                'representative_name': representative_name,
                'sample_names': names[:CLUSTER_SAMPLES],
                'all_names': names,
                'top_words': [word for word, count in word_counts.most_common(TOP_WORDS_PER_CLUSTER)],
                'word_counts': word_counts,
                'word_diversity': word_diversity,
                'size': len(cluster_data)
            }
        
        return cluster_features
    
    def _extract_meaningful_words(self, names: List[str]) -> List[str]:
        """Extract meaningful words from product names."""
        all_words = []
        
        for name in names:
            if not name or not isinstance(name, str):
                continue
                
            # Clean and split the name
            clean_name = re.sub(r'[^\w\s]', ' ', name.lower())
            words = clean_name.split()
            
            for word in words:
                # Filter meaningful words
                if (len(word) >= MIN_WORD_LENGTH and 
                    word not in STOP_WORDS and
                    not word.isdigit() and
                    word.isalpha()):
                    all_words.append(word)
        
        return all_words
    
    def _compute_cluster_embeddings(self, 
                                   clustered_data: pd.DataFrame,
                                   embeddings: np.ndarray,
                                   cluster_column: str) -> Dict[int, np.ndarray]:
        """Compute centroid embeddings for each cluster."""
        cluster_embeddings = {}
        
        for cluster_id in clustered_data[cluster_column].unique():
            if cluster_id == -1:  # Skip noise
                continue
                
            cluster_mask = clustered_data[cluster_column] == cluster_id
            cluster_indices = clustered_data[cluster_mask].index.tolist()
            
            if cluster_indices:
                # Compute centroid
                cluster_vecs = embeddings[cluster_indices]
                centroid = np.mean(cluster_vecs, axis=0)
                cluster_embeddings[cluster_id] = centroid
        
        return cluster_embeddings
    
    def _assign_clusters_to_categories(self,
                                     cluster_features: Dict[int, Dict],
                                     cluster_embeddings: Dict[int, np.ndarray]) -> Dict[int, Tuple[str, float]]:
        """
        **Approach 2: Unsupervised Clustering with Word Embeddings**
        
        Automatically assign clusters to categories using:
        1. Semantic embedding similarity (cosine similarity in high-dimensional space)
        2. K-means clustering of cluster centroids 
        3. Zero-shot classification enhancement (Approach 4)
        4. Word pattern analysis for interpretability
        """
        if not cluster_embeddings:
            return {}
        
        logger.info(f"🧠 Approach 2: Auto-assigning {len(cluster_embeddings)} clusters using semantic embeddings")
        
        # Convert cluster embeddings to matrix for vectorized operations
        cluster_ids = list(cluster_embeddings.keys())
        embedding_matrix = np.array([cluster_embeddings[cid] for cid in cluster_ids])
        
        # **Step 1: Semantic Embedding Analysis**
        # Use cosine similarity to group semantically similar clusters
        category_groups = self._cluster_centroids_semantic(embedding_matrix, len(self.main_categories))
        
        # **Step 2: Zero-shot Enhancement (Approach 4)**
        zero_shot_assignments = self._enhance_with_zero_shot(cluster_features, cluster_ids)
        
        # **Step 3: Word Pattern Analysis for Interpretability**
        category_meanings = self._infer_category_meanings(cluster_features, category_groups, cluster_ids)
        
        # **Step 4: Hybrid Assignment with Multiple Signals**
        assignments = self._make_hybrid_assignments(
            cluster_ids, category_groups, category_meanings, 
            cluster_features, zero_shot_assignments
        )
        
        logger.info("✅ Approach 2 assignment complete using semantic similarity")
        return assignments
    
    def _enhance_with_zero_shot(self, 
                               cluster_features: Dict[int, Dict],
                               cluster_ids: List[int]) -> Dict[int, Tuple[str, float]]:
        """
        **Approach 4: Zero-Shot Classification Enhancement**
        Use pre-trained LLMs to enhance cluster assignments.
        """
        zero_shot_assignments = {}
        
        if not self.zero_shot_classifier:
            logger.info("⚠️ Zero-shot classifier not available, skipping enhancement")
            return zero_shot_assignments
        
        logger.info("🤖 Approach 4: Enhancing assignments with zero-shot classification")
        
        # Get representative names for zero-shot classification
        representatives = []
        for cluster_id in cluster_ids:
            features = cluster_features[cluster_id]
            representatives.append(features['representative_name'])
        
        try:
            # Use zero-shot classifier to get category assignments
            results = self.zero_shot_classifier.classify_batch(
                representatives, threshold=self.confidence_threshold
            )

            for cluster_id, (pred_label, confidence) in zip(cluster_ids, results):
                # Map enriched label back to base category name if needed
                base_category = self.label_to_category.get(pred_label, pred_label)
                zero_shot_assignments[cluster_id] = (base_category, confidence)

            logger.info(f"✅ Zero-shot enhanced {len(zero_shot_assignments)} cluster assignments")
            
        except Exception as e:
            logger.warning(f"Zero-shot enhancement failed: {e}")
        
        return zero_shot_assignments
    
    def _cluster_centroids_semantic(self, embedding_matrix: np.ndarray, n_categories: int) -> np.ndarray:
        """
        **Core of Approach 2**: Cluster embeddings using cosine similarity.
        Groups semantically similar cluster centroids together.
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import normalize
            
            logger.info(f"🔢 Semantic clustering: {len(embedding_matrix)} cluster centroids → {n_categories} groups")
            
            # **Key**: Normalize embeddings for cosine similarity
            # This is crucial for semantic similarity in high-dimensional space
            normalized_embeddings = normalize(embedding_matrix, norm='l2')
            
            # Use K-means with cosine similarity (via normalization)
            kmeans = KMeans(
                n_clusters=min(n_categories, len(embedding_matrix)), 
                random_state=42, 
                n_init=10,
                max_iter=300
            )
            category_groups = kmeans.fit_predict(normalized_embeddings)
            
            # Store semantic centroids for category representation
            self.category_centroids = {
                i: centroid for i, centroid in enumerate(kmeans.cluster_centers_)
            }
            
            logger.info(f"✅ Semantic clustering complete: {len(np.unique(category_groups))} groups found")
            return category_groups
            
        except ImportError:
            logger.warning("sklearn not available, using random assignment")
            return np.random.randint(0, n_categories, len(embedding_matrix))
        except Exception as e:
            logger.error(f"Semantic clustering failed: {e}")
            return np.random.randint(0, n_categories, len(embedding_matrix))
    
    def _cluster_centroids(self, embedding_matrix: np.ndarray, n_categories: int) -> np.ndarray:
        """Cluster the cluster centroids into category groups."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import normalize
            
            # Normalize embeddings
            normalized_embeddings = normalize(embedding_matrix, norm='l2')
            
            # Cluster into category groups
            kmeans = KMeans(n_clusters=min(n_categories, len(embedding_matrix)), 
                          random_state=42, n_init=10)
            category_groups = kmeans.fit_predict(normalized_embeddings)
            
            # Store category centroids for later use
            self.category_centroids = {i: centroid for i, centroid in enumerate(kmeans.cluster_centers_)}
            
            return category_groups
            
        except ImportError:
            logger.warning("sklearn not available, using random assignment")
            return np.random.randint(0, n_categories, len(embedding_matrix))
    
    def _infer_category_meanings(self,
                               cluster_features: Dict[int, Dict],
                               category_groups: np.ndarray,
                               cluster_ids: List[int]) -> Dict[int, Dict]:
        """
        Infer what each category group represents by analyzing words.
        """
        category_meanings = {}
        
        # Group clusters by category assignment
        for group_id in np.unique(category_groups):
            group_cluster_ids = [cluster_ids[i] for i in np.where(category_groups == group_id)[0]]
            
            # Aggregate words from all clusters in this group
            all_words = []
            all_names = []
            total_size = 0
            
            for cluster_id in group_cluster_ids:
                features = cluster_features[cluster_id]
                all_words.extend(features['top_words'])
                all_names.extend(features['all_names'])
                total_size += features['size']
            
            # Find most common words in this category group
            word_counts = Counter(all_words)
            top_words = [word for word, count in word_counts.most_common(10)]
            
            # Try to match with user-defined category names
            best_category_match = self._match_category_by_words(top_words, all_names)
            
            category_meanings[group_id] = {
                'cluster_ids': group_cluster_ids,
                'top_words': top_words,
                'sample_names': all_names[:10],
                'total_size': total_size,
                'inferred_category': best_category_match[0],
                'match_confidence': best_category_match[1]
            }
        
        return category_meanings
    
    def _match_category_by_words(self, top_words: List[str], sample_names: List[str]) -> Tuple[str, float]:
        """
        Match a group of words/names to the most likely user category.
        """
        category_scores = {}
        
        # Combine words and names for analysis
        all_text = ' '.join(top_words + sample_names).lower()
        
        for category in self.main_categories:
            score = 0.0
            category_lower = category.lower()
            
            # Direct category name matches
            category_word = category_lower.rstrip('s')  # Remove plural
            
            # Score based on word/name content
            for word in top_words:
                # Exact match with category
                if category_word in word or word in category_word:
                    score += 3.0
                
                # Semantic similarity heuristics
                elif self._are_semantically_related(category_word, word):
                    score += 2.0
                
                # Partial matches
                elif category_word[:3] in word or word[:3] in category_word:
                    score += 1.0
            
            # Normalize by number of words
            if top_words:
                score = score / len(top_words)
            
            category_scores[category] = score
        
        # Find best match
        if not category_scores or max(category_scores.values()) == 0:
            return 'Unclassified', 0.0
        
        best_category = max(category_scores, key=category_scores.get)
        confidence = min(category_scores[best_category], 1.0)
        
        return best_category, confidence
    
    def _are_semantically_related(self, category: str, word: str) -> bool:
        """
        Simple heuristic to check if a word is semantically related to a category.
        This could be enhanced with word embeddings or WordNet.
        """
        # Define some common semantic relationships
        semantic_groups = {
            'table': ['desk', 'surface', 'workstation', 'mesa', 'masa'],
            'chair': ['seat', 'stool', 'sandalye', 'silla'],
            'computer': ['pc', 'laptop', 'desktop', 'bilgisayar', 'ordenador'],
            'office': ['supplies', 'equipment', 'furniture', 'material'],
            'supply': ['pen', 'paper', 'notebook', 'kalem', 'defter'],
            'service': ['services', 'support', 'subscription', 'license', 'licence', 'maintenance', 'saas', 'software', 'internet', 'cloud']
        }
        
        category_key = category.lower().rstrip('s')
        category_words = [category_key] + semantic_groups.get(category_key, [])
        
        for cat_word in category_words:
            if cat_word in word or word in cat_word:
                return True
        
        return False
    
    def _make_hybrid_assignments(self,
                               cluster_ids: List[int],
                               category_groups: np.ndarray,
                               category_meanings: Dict[int, Dict],
                               cluster_features: Dict[int, Dict],
                               zero_shot_assignments: Dict[int, Tuple[str, float]]) -> Dict[int, Tuple[str, float]]:
        """
        **Hybrid Assignment**: Combine semantic clustering + zero-shot + word patterns.
        
        Priority order:
        1. High-confidence zero-shot assignments (Approach 4)
        2. Semantic embedding clusters (Approach 2) 
        3. Word pattern fallback
        """
        assignments = {}
        
        logger.info("🔀 Making hybrid assignments from multiple signals")
        
        for i, cluster_id in enumerate(cluster_ids):
            group_id = category_groups[i]
            group_info = category_meanings[group_id]
            
            # **Priority 1: High-confidence zero-shot (slightly lower to favor correct service mappings)**
            if (cluster_id in zero_shot_assignments and 
                zero_shot_assignments[cluster_id][1] >= 0.6):  # High confidence threshold
                category, confidence = zero_shot_assignments[cluster_id]
                assignments[cluster_id] = (category, confidence)
                logger.debug(f"Cluster {cluster_id}: Zero-shot assignment (high conf)")
                continue
            
            # **Priority 2: Semantic embedding cluster assignment**
            embedding_category = group_info['inferred_category']
            embedding_confidence = group_info['match_confidence']
            
            # **Priority 3: Medium-confidence zero-shot as enhancement**
            if (cluster_id in zero_shot_assignments and 
                zero_shot_assignments[cluster_id][1] >= 0.4):
                zero_category, zero_conf = zero_shot_assignments[cluster_id]
                
                # If zero-shot agrees with embedding clustering, boost confidence
                if zero_category == embedding_category:
                    final_confidence = min(1.0, (embedding_confidence + zero_conf) / 2 + 0.2)
                    assignments[cluster_id] = (embedding_category, final_confidence)
                    logger.debug(f"Cluster {cluster_id}: Agreement boost")
                # If they disagree, use higher confidence
                elif zero_conf + 0.05 >= embedding_confidence:
                    assignments[cluster_id] = (zero_category, zero_conf)
                    logger.debug(f"Cluster {cluster_id}: Zero-shot override (medium conf)")
                else:
                    assignments[cluster_id] = (embedding_category, embedding_confidence)
                    logger.debug(f"Cluster {cluster_id}: Embedding wins")
            else:
                # **Priority 4: Pure embedding-based assignment**
                assignments[cluster_id] = (embedding_category, embedding_confidence)
                logger.debug(f"Cluster {cluster_id}: Pure embedding")
            
            # **Apply confidence threshold**
            category, confidence = assignments[cluster_id]
            
            # Adjust confidence based on cluster quality
            cluster_size = cluster_features[cluster_id]['size']
            word_diversity = cluster_features[cluster_id]['word_diversity']
            
            # Larger, more diverse clusters get slight confidence boost
            size_boost = min(0.15, cluster_size / 100)
            diversity_boost = min(0.05, word_diversity)
            final_confidence = min(1.0, confidence + size_boost + diversity_boost)
            
            # Apply threshold
            if final_confidence < self.confidence_threshold:
                assignments[cluster_id] = ('Unclassified', final_confidence)
            else:
                assignments[cluster_id] = (category, final_confidence)
        
        # Log hybrid assignment summary
        assignment_methods = {
            'zero_shot_high': sum(1 for cid in cluster_ids if cid in zero_shot_assignments and zero_shot_assignments[cid][1] >= 0.7),
            'embedding_primary': len([cid for cid in cluster_ids if assignments[cid][0] != 'Unclassified']),
            'unclassified': len([cid for cid in cluster_ids if assignments[cid][0] == 'Unclassified'])
        }
        
        logger.info(f"📊 Hybrid assignment summary: {assignment_methods}")
        
        return assignments
    
    def _make_final_assignments(self,
                              cluster_ids: List[int],
                              category_groups: np.ndarray,
                              category_meanings: Dict[int, Dict],
                              cluster_features: Dict[int, Dict]) -> Dict[int, Tuple[str, float]]:
        """Make final category assignments with confidence scores."""
        assignments = {}
        
        for i, cluster_id in enumerate(cluster_ids):
            group_id = category_groups[i]
            group_info = category_meanings[group_id]
            
            # Base assignment from group
            category = group_info['inferred_category']
            confidence = group_info['match_confidence']
            
            # Adjust confidence based on cluster-specific factors
            cluster_size = cluster_features[cluster_id]['size']
            word_diversity = cluster_features[cluster_id]['word_diversity']
            
            # Larger, more diverse clusters get higher confidence
            size_boost = min(0.2, cluster_size / 100)
            diversity_boost = min(0.1, word_diversity)
            
            final_confidence = min(1.0, confidence + size_boost + diversity_boost)
            
            # Apply threshold
            if final_confidence < self.confidence_threshold:
                category = 'Unclassified'
                final_confidence = 0.0
            
            assignments[cluster_id] = (category, final_confidence)
        
        # Log summary
        assigned_categories = [cat for cat, conf in assignments.values() if cat != 'Unclassified']
        category_counts = Counter(assigned_categories)
        
        logger.info(f"📊 Category assignments: {dict(category_counts)}")
        
        return assignments
    
    def get_category_summary(self, analysis_df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics by category."""
        if analysis_df.empty:
            return pd.DataFrame()
        
        summary = analysis_df.groupby('category').agg({
            'cluster_id': 'count',
            'total_items': 'sum',
            'unique_barcodes': 'sum',
            'confidence': 'mean',
            'representative_name': lambda x: ', '.join(x[:3])
        }).rename(columns={
            'cluster_id': 'num_clusters',
            'confidence': 'avg_confidence',
            'representative_name': 'example_names'
        })
        
        # Calculate percentage
        total_items = summary['total_items'].sum()
        summary['percentage'] = (summary['total_items'] / total_items * 100).round(1)
        
        # Sort by total items
        summary = summary.sort_values('total_items', ascending=False)
        
        return summary.reset_index()
    
    def explain_assignment(self, cluster_id: int) -> str:
        """Explain why a cluster was assigned to its category."""
        if cluster_id not in self.cluster_analysis:
            return f"Cluster {cluster_id} not found in analysis"
        
        features = self.cluster_analysis[cluster_id]
        category, confidence = self.category_assignments.get(cluster_id, ('Unclassified', 0.0))
        
        explanation = [
            f"🔍 Cluster {cluster_id} → {category} (confidence: {confidence:.2f})",
            f"📝 Representative: '{features['representative_name']}'",
            f"📊 Size: {features['size']} items",
            f"🏷️ Top words: {', '.join(features['top_words'][:5])}",
            f"📋 Examples: {', '.join(features['sample_names'][:3])}"
        ]
        
        return '\n'.join(explanation)

# Enhanced alias for the enhanced pipeline
EnhancedClusterMapper = AutoClusterMapper
