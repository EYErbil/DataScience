#!/usr/bin/env python3
"""
Test script to check enhanced_demo.ipynb cells for errors
"""

import sys
sys.path.append('src')
sys.path.append('config')

# Module level imports
from pipeline_runner import ProductCategorizationPipeline
from user_categories import MAIN_CATEGORIES
try:
    from config import *
except ImportError:
    pass  # config might not exist
import pandas as pd
import numpy as np

def test_pipeline_setup():
    """Test the pipeline setup cell"""
    print('🚀 Testing enhanced_demo.ipynb pipeline setup cell...')
    
    try:

        print(f'✅ Target categories: {MAIN_CATEGORIES}')

        # Load data
        print('📊 Loading ultra-challenging dataset...')
        data_path = 'data/ultra_challenging_dataset.csv'
        raw_data = pd.read_csv(data_path)

        print(f'📋 Raw dataset: {len(raw_data):,} items')
        print(f'📋 Columns: {list(raw_data.columns)}')

        # Extract ground truth from true_category column with normalized keys
        ground_truth = {}
        if 'true_category' in raw_data.columns:
            for _, row in raw_data.iterrows():
                # Normalize the key to match clean_data['name'] format (lowercase)
                normalized_key = row['product_name'].lower()
                ground_truth[normalized_key] = row['true_category']
            print(f'✅ Ground truth extracted: {len(ground_truth):,} labeled items')
        else:
            print('⚠️ No true_category column found')

        # Initialize pipeline with correct parameters
        print('🔄 Initializing pipeline...')
        pipeline = ProductCategorizationPipeline(
            main_categories=MAIN_CATEGORIES,
            encoder_type='auto',
            clusterer_type='faiss',
            force_rebuild=False
        )

        # Run the pipeline to get all prerequisites
        print('⚙️ Running full pipeline to generate prerequisites...')
        results_df = pipeline.run(data_path)

        # Extract the prerequisites that approaches need from pipeline components
        clean_data = pipeline.clean_data
        embeddings = pipeline.embeddings 
        cluster_labels = pipeline.cluster_labels
        n_clusters = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        # Add ground truth to clean_data for evaluation
        if 'true_category' in raw_data.columns:
            # Map ground truth to clean_data
            clean_data['true_category'] = clean_data['name'].map(ground_truth)

        print(f'✅ Pipeline setup complete!')
        print(f'   📊 Dataset: {len(clean_data):,} items')
        print(f'   🎯 Categories: {len(MAIN_CATEGORIES)} target categories') 
        print(f'   🧠 Embeddings: {embeddings.shape[1]}D vectors ({embeddings.shape[0]:,} items)')
        print(f'   🔗 Clusters: {n_clusters} discovered clusters')
        print(f'   🎯 Ground truth: {len(ground_truth):,} labeled items available')
        
        # Check ground truth mapping
        mapped_count = clean_data['true_category'].notna().sum()
        print(f'   ✅ Clean data true_category column: {mapped_count} mapped items')
        print(f'   🚀 Ready for pure approach analysis!')
        
        return True, clean_data, embeddings, cluster_labels, ground_truth

    except Exception as e:
        print(f'❌ Pipeline setup error: {e}')
        import traceback
        traceback.print_exc()
        return False, None, None, None, None

def test_approach4():
    """Test Approach 4 zero-shot classification"""
    print('\n🤖 Testing Approach 4 Zero-Shot Classification...')
    
    try:
        from categorisation.zero_shot_classifier import ZeroShotClassifier
        
        # Initialize zero-shot classifier
        zero_shot = ZeroShotClassifier()
        print('✅ ZeroShotClassifier initialized')
        
        # Test categories with descriptions
        enhanced_categories = [
            "Furniture: Office furniture, desks, chairs, tables, cabinets, storage",
            "Technology: Computers, laptops, monitors, printers, hardware, electronics", 
            "Services: Software licenses, subscriptions, internet services, support"
        ]
        
        # Test a few sample classifications
        test_items = ["office chair", "laptop computer", "software license"]
        
        for item in test_items:
            enhanced_text = f"Product: {item} | Type: office/business item"
            pred_category, confidence = zero_shot.get_best_category(enhanced_text, enhanced_categories)
            
            # Handle the 'Unclassified' return from zero-shot and normalize to 'Uncategorized'
            if pred_category == 'Unclassified':
                pred_category = 'Uncategorized'
                confidence = 0.0
                
            print(f'   📝 "{item}" → {pred_category} (confidence: {confidence:.3f})')
        
        print('✅ Approach 4 working correctly')
        return True
        
    except Exception as e:
        print(f'❌ Approach 4 error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Enhanced Demo Notebook Components\n")
    print("🔄 Starting tests...")
    
    try:
        # Test pipeline setup
        setup_ok, clean_data, embeddings, cluster_labels, ground_truth = test_pipeline_setup()
        
        # Test Approach 4
        approach4_ok = test_approach4()
        
        print(f"\n📊 Test Results:")
        print(f"   Pipeline Setup: {'✅ PASS' if setup_ok else '❌ FAIL'}")
        print(f"   Approach 4: {'✅ PASS' if approach4_ok else '❌ FAIL'}")
        
        if setup_ok and approach4_ok:
            print("\n🎯 All core components working - notebook should execute successfully!")
        else:
            print("\n⚠️ Some components have errors - check output above")
    except Exception as e:
        print(f"❌ Unexpected error in main: {e}")
        import traceback
        traceback.print_exc()
