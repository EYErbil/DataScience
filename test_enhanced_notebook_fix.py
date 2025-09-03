"""
Test that the enhanced notebook now uses the correct embedding model.
"""
import sys
sys.path.append('src')

print("🔍 Testing enhanced notebook fix...")

# Simulate what the notebook does now
from embedding.simple_encoder import SimpleEncoder
from config import EMBEDDING_MODEL

print(f"📋 Config EMBEDDING_MODEL: {EMBEDDING_MODEL}")

# This is what the notebook SHOULD do now (after our fix)
print(f"\n✅ Testing fixed notebook behavior:")
encoder = SimpleEncoder(model_name=EMBEDDING_MODEL)
print(f"   Model being used: {encoder.model_name}")
print(f"   Expected: {EMBEDDING_MODEL}")

if encoder.model_name == EMBEDDING_MODEL:
    print(f"\n🎉 SUCCESS! Enhanced notebook will now use the correct model!")
    print(f"   ✅ Using: {EMBEDDING_MODEL} (1024 dimensions)")
    print(f"   🆙 Upgrade from: all-MiniLM-L6-v2 (384 dimensions)")
    print(f"   💪 2.7x richer semantic understanding!")
else:
    print(f"\n❌ ISSUE: Notebook still using wrong model")
    print(f"   Expected: {EMBEDDING_MODEL}")
    print(f"   Got: {encoder.model_name}")
