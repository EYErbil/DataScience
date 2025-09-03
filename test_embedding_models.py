"""
Test which embedding model is actually being used in different scenarios.
"""
import sys
sys.path.append('src')

from embedding.simple_encoder import SimpleEncoder
from config import EMBEDDING_MODEL

print("🔍 Testing embedding model usage:")
print(f"   Config EMBEDDING_MODEL: {EMBEDDING_MODEL}")

# Test 1: Default SimpleEncoder (what notebook currently uses)
print("\n1️⃣ Testing SimpleEncoder() - default behavior:")
encoder1 = SimpleEncoder()
print(f"   Default model_name: {encoder1.model_name}")

# Test 2: Enhanced SimpleEncoder with explicit model
print("\n2️⃣ Testing SimpleEncoder(model_name=EMBEDDING_MODEL):")
encoder2 = SimpleEncoder(model_name=EMBEDDING_MODEL)
print(f"   Enhanced model_name: {encoder2.model_name}")

print(f"\n🎯 ISSUE IDENTIFIED:")
print(f"   ❌ Notebook uses: SimpleEncoder() → defaults to '{encoder1.model_name}'")
print(f"   ✅ Should use: SimpleEncoder(model_name=EMBEDDING_MODEL) → uses '{encoder2.model_name}'")

print(f"\n💡 SOLUTION:")
print(f"   The notebook needs to explicitly pass the enhanced model name!")
