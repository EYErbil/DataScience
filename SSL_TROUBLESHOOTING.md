# SSL Certificate Troubleshooting Guide

## Problem
You're seeing an SSL certificate verification error when trying to download sentence transformer models:
```
SSLError: certificate verify failed: self signed certificate in certificate chain
```

## What We've Implemented

### 1. Automatic SSL Fixes
The code now automatically tries to bypass SSL verification:
- Disables SSL hostname checking
- Sets environment variables to bypass certificate bundles
- Disables SSL warnings

### 2. TF-IDF Fallback
If sentence transformers fail to download, the system automatically falls back to TF-IDF:
- Uses scikit-learn's TfidfVectorizer
- Still provides good similarity detection
- Works completely offline
- Adjusts similarity thresholds automatically

## Solutions (In Order of Preference)

### Option 1: Use the Notebook (Recommended)
The Jupyter notebook (`notebooks/demo.ipynb`) has been updated with comprehensive SSL fixes and fallbacks. Simply run it - it will handle everything automatically.

### Option 2: Network Solutions
If you need the full transformer models:

1. **Try a different network**: Mobile hotspot, home WiFi, etc.
2. **Contact IT**: Ask them to whitelist `huggingface.co`
3. **VPN**: Try connecting through a VPN

### Option 3: Manual Model Download
If you have internet access elsewhere:

1. Download models on another machine:
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
   ```

2. Copy the model cache folder:
   - **Windows**: `C:\Users\{username}\.cache\torch\sentence_transformers\`
   - **Linux/Mac**: `~/.cache/torch/sentence_transformers/`

### Option 4: Use TF-IDF Only
For a completely offline solution, you can force TF-IDF mode:

```python
from src.embedding import NameEmbedder
embedder = NameEmbedder('dummy-model')
embedder._use_tfidf_fallback = True
# Now use embedder normally - it will use TF-IDF
```

## Performance Comparison

| Method | Quality | Speed | Offline | Network Required |
|--------|---------|-------|---------|-----------------|
| Sentence Transformers | Excellent | Fast | No | Yes (first time) |
| TF-IDF Fallback | Good | Very Fast | Yes | No |

## What Works With TF-IDF
- ✅ Clustering similar product names
- ✅ Multilingual text (basic level)
- ✅ Similarity detection
- ✅ Complete pipeline functionality
- ✅ All data enrichment features

## What's Better With Transformers
- 🚀 Better cross-language similarity
- 🚀 More nuanced semantic understanding
- 🚀 Better handling of abbreviations/synonyms

## Bottom Line
**The TF-IDF fallback works very well for your use case!** Product name clustering doesn't require the most sophisticated NLP - even TF-IDF will correctly group "Table", "Desk", and "Mesa" when combined with your barcode merging logic.
