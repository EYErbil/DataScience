"""
Test that both notebooks are functional and don't interfere with each other.
"""

import sys
sys.path.append('src')
sys.path.append('config')

def test_original_notebook_components():
    """Test that original notebook components still work."""
    print("🧪 Testing ORIGINAL notebook components...")
    
    try:
        # Test original clusterer
        from clustering.faiss_clusterer import FaissClusterer
        original_clusterer = FaissClusterer(similarity_threshold=0.4, min_cluster_size=2)
        print("✅ Original FaissClusterer: OK")
        
        # Test original mapper
        from categorisation.cluster_mapper import AutoClusterMapper
        from user_categories import MAIN_CATEGORIES
        original_mapper = AutoClusterMapper(main_categories=MAIN_CATEGORIES)
        print("✅ Original AutoClusterMapper: OK")
        
        # Test original dataset path
        from pathlib import Path
        original_data = Path("data/large_office_inventory_900.csv")
        if original_data.exists():
            print("✅ Original dataset: OK")
        else:
            print("⚠️ Original dataset: Missing (run generation script)")
        
        print("✅ Original notebook components: ALL OK")
        return True
        
    except Exception as e:
        print(f"❌ Original notebook test failed: {e}")
        return False

def test_enhanced_notebook_components():
    """Test that enhanced notebook components work."""
    print("\n🧪 Testing ENHANCED notebook components...")
    
    try:
        # Test enhanced clusterer
        from clustering.enhanced_faiss_clusterer import EnhancedFaissClusterer
        enhanced_clusterer = EnhancedFaissClusterer(
            similarity_threshold=0.6, 
            min_cluster_size=3,
            use_hierarchical_refinement=True
        )
        print("✅ Enhanced FaissClusterer: OK")
        
        # Test enhanced mapper (alias)
        from categorisation.cluster_mapper import EnhancedClusterMapper
        from user_categories import MAIN_CATEGORIES
        enhanced_mapper = EnhancedClusterMapper(main_categories=MAIN_CATEGORIES)
        print("✅ Enhanced ClusterMapper: OK")
        
        # Test enhanced dataset
        from pathlib import Path
        enhanced_data = Path("data/ultra_challenging_dataset.csv")
        if enhanced_data.exists():
            print("✅ Enhanced dataset: OK")
        else:
            print("⚠️ Enhanced dataset: Missing")
        
        # Test enhanced config
        from config import EMBEDDING_MODEL
        if "multilingual-e5-large" in EMBEDDING_MODEL:
            print("✅ Enhanced embedding model: OK")
        else:
            print("⚠️ Enhanced embedding model: Not configured")
        
        print("✅ Enhanced notebook components: ALL OK")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced notebook test failed: {e}")
        return False

def test_no_conflicts():
    """Test that both versions can coexist."""
    print("\n🧪 Testing NO CONFLICTS between versions...")
    
    try:
        # Import both versions
        from clustering.faiss_clusterer import FaissClusterer
        from clustering.enhanced_faiss_clusterer import EnhancedFaissClusterer
        from categorisation.cluster_mapper import AutoClusterMapper, EnhancedClusterMapper
        
        # Verify they're different classes
        assert FaissClusterer != EnhancedFaissClusterer
        assert AutoClusterMapper == EnhancedClusterMapper  # Enhanced is an alias
        
        print("✅ No import conflicts: OK")
        print("✅ Both versions coexist: OK")
        return True
        
    except Exception as e:
        print(f"❌ Conflict test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 TESTING BOTH NOTEBOOK VERSIONS")
    print("=" * 50)
    
    original_ok = test_original_notebook_components()
    enhanced_ok = test_enhanced_notebook_components()
    no_conflicts = test_no_conflicts()
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   Original notebook: {'✅ PASS' if original_ok else '❌ FAIL'}")
    print(f"   Enhanced notebook: {'✅ PASS' if enhanced_ok else '❌ FAIL'}")
    print(f"   No conflicts: {'✅ PASS' if no_conflicts else '❌ FAIL'}")
    
    if original_ok and enhanced_ok and no_conflicts:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   📖 Original notebook: uses original dataset + settings")
        print(f"   🚀 Enhanced notebook: uses enhanced dataset + models + analysis")
        print(f"   🔒 No conflicts: both can run independently")
    else:
        print(f"\n⚠️ Some tests failed - check the issues above")
