import json
from pathlib import Path


def replace_block(lines, start_pred, end_pred, new_block):
    try:
        start = next(i for i, l in enumerate(lines) if start_pred(l))
    except StopIteration:
        return lines, False
    end = start
    while end < len(lines) and not end_pred(lines[end]):
        end += 1
    if end < len(lines):
        lines = lines[:start] + new_block + lines[end + 1 :]
    else:
        lines = lines[:start] + new_block
    return lines, True


def patch_notebook(nb_path: Path) -> bool:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    changed = False

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source") or []
        text = "".join(src)

        # Patch Approach 2 centroid-to-category mapping and confidence
        if (
            "APPROACH 2: PURE SEMANTIC CLUSTERING" in text
            or "PURE SEMANTIC CLUSTERING ANALYSIS" in text
        ):
            # Replace KMeans mapping block with cosine-prototype mapping
            def start_pred(l):
                return "Grouping cluster centroids using K-means" in l

            def end_pred(l):
                return "Computing semantic confidence" in l

            new_block = [
                "print(\"🧠 PURE SEMANTIC: Mapping centroids to categories via cosine similarity...\")\n",
                "\n",
                "# Build category prototype embeddings (use descriptions if available)\n",
                "try:\n",
                "    from user_categories import CATEGORY_DESCRIPTIONS\n",
                "    cat_texts = [CATEGORY_DESCRIPTIONS.get(cat, cat) for cat in MAIN_CATEGORIES]\n",
                "except Exception:\n",
                "    cat_texts = MAIN_CATEGORIES\n",
                "\n",
                "# Use the same encoder as pipeline (semantic-only)\n",
                "assert 'pipeline' in globals() or 'pipeline' in locals(), \"pipeline must exist to access encoder\"\n",
                "category_embeddings = pipeline.encoder.encode(cat_texts)\n",
                "\n",
                "# Compute assignments by nearest category (cosine similarity)\n",
                "from sklearn.metrics.pairwise import cosine_similarity\n",
                "cluster_ids = list(cluster_centroids.keys())\n",
                "if len(cluster_ids) > 0:\n",
                "    centroid_matrix = np.array([cluster_centroids[cid] for cid in cluster_ids])\n",
                "    sims = cosine_similarity(centroid_matrix, category_embeddings)\n",
                "    order = np.argsort(-sims, axis=1)\n",
                "    best_idx = order[:, 0]\n",
                "    second_idx = order[:, 1] if sims.shape[1] > 1 else best_idx\n",
                "    category_assignments = {cid: MAIN_CATEGORIES[best_idx[i]] for i, cid in enumerate(cluster_ids)}\n",
                "    centroid_to_cat_sim = {cid: float(sims[i, best_idx[i]]) for i, cid in enumerate(cluster_ids)}\n",
                "    centroid_to_margin = {cid: float(sims[i, best_idx[i]] - sims[i, second_idx[i]]) for i, cid in enumerate(cluster_ids)}\n",
                "    print(f\"✅ Mapped {len(cluster_ids)} centroids to categories using cosine similarity\")\n",
                "else:\n",
                "    category_assignments = {}\n",
                "    centroid_to_cat_sim = {}\n",
                "    centroid_to_margin = {}\n",
                "    print(\"❌ No valid clusters found - all items will be uncategorized\")\n",
                "\n",
                "print(\"📊 Computing semantic confidence scores...\")\n",
            ]

            new_src, ok = replace_block(src, start_pred, end_pred, new_block)
            if ok:
                cell["source"] = new_src
                src = new_src
                changed = True

            # Replace confidence computation to use margin-based blend
            try:
                k = next(
                    i
                    for i, l in enumerate(src)
                    if "item_cluster_similarity = cosine_similarity" in l
                )
                rep = [
                    "        # Cosine similarity between item and its cluster centroid\n",
                    "        item_cluster_similarity = float(cosine_similarity([item_embedding], [cluster_centroid])[0][0])\n",
                    "        # Base confidence from centroid-to-category similarity and margin\n",
                    "        base_cat_sim = centroid_to_cat_sim.get(cluster_id, item_cluster_similarity)\n",
                    "        cat_margin = centroid_to_margin.get(cluster_id, 0.0)\n",
                    "        # Normalize by cluster size (log scale to avoid huge numbers)\n",
                    "        cluster_size_factor = min(1.0, np.log(cluster_sizes[cluster_id] + 1) / 10)\n",
                    "        # Final confidence combines signals\n",
                    "        confidence = (base_cat_sim * 0.6) + (cat_margin * 0.2) + (item_cluster_similarity * 0.2)\n",
                    "        confidence = max(0.0, min(1.0, confidence))\n",
                ]
                # Try to find the end of the old confidence block by searching for the clamp line
                endk = k
                while endk < len(src) and "confidence =" not in src[endk]:
                    endk += 1
                endk = min(endk + 2, len(src))
                cell["source"] = src[:k] + rep + src[endk:]
                changed = True
            except StopIteration:
                pass

        # Patch Optional: Confusion Matrices cell to be robust and clearer
        if "OPTIONAL: CONFUSION MATRICES" in text or "📊 OPTIONAL: CONFUSION MATRICES" in text:
            new_src_conf = [
                "# 📊 OPTIONAL: CONFUSION MATRICES & DETAILED EXAMPLES\n",
                "print(\n\"\\n📊 GENERATING DETAILED ANALYSIS...\")\n",
                "\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "from sklearn.metrics import confusion_matrix, classification_report\n",
                "\n",
                "# Ensure reporting context exists (robust to partial runs)\n",
                "if 'approaches' not in globals():\n",
                "    approaches = {}\n",
                "    if 'approach2_results' in globals():\n",
                "        a2_metrics, a2_cat = compute_approach_metrics(approach2_results, \"Pure Semantic Clustering\")\n",
                "        approaches['Approach 2 (Semantic)'] = {\n",
                "            'metrics': a2_metrics, 'results': approach2_results, 'categorized': a2_cat\n",
                "        }\n",
                "    if 'approach4_results' in globals():\n",
                "        a4_metrics, a4_cat = compute_approach_metrics(approach4_results, \"Pure Zero-Shot Classification\")\n",
                "        approaches['Approach 4 (Zero-Shot)'] = {\n",
                "            'metrics': a4_metrics, 'results': approach4_results, 'categorized': a4_cat\n",
                "        }\n",
                "    if 'hybrid_results' in globals():\n",
                "        h_metrics, h_cat = compute_approach_metrics(hybrid_results, \"Hybrid (Best of Both)\")\n",
                "        approaches['Hybrid (Best of Both)'] = {\n",
                "            'metrics': h_metrics, 'results': hybrid_results, 'categorized': h_cat\n",
                "        }\n",
                "\n",
                "# Helper: derive y_true/y_pred safely (prefer DataFrame true_category)\n",
                "def _get_truth_and_pred(df: pd.DataFrame):\n",
                "    if 'true_category' in df.columns:\n",
                "        mask = df['true_category'].notna() & (df['predicted_category'] != 'Uncategorized')\n",
                "        y_true = df.loc[mask, 'true_category'].astype(str)\n",
                "        y_pred = df.loc[mask, 'predicted_category'].astype(str)\n",
                "        return y_true, y_pred\n",
                "    if 'ground_truth' in globals() and ground_truth:\n",
                "        truths, preds = [], []\n",
                "        for _, r in df.iterrows():\n",
                "            key = (r.get('name') or '').lower()\n",
                "            if key in ground_truth and r['predicted_category'] != 'Uncategorized':\n",
                "                truths.append(ground_truth[key])\n",
                "                preds.append(r['predicted_category'])\n",
                "        return pd.Series(truths), pd.Series(preds)\n",
                "    return pd.Series(dtype=str), pd.Series(dtype=str)\n",
                "\n",
                "# Build confusion matrices if truth available\n",
                "print(\n\"\\n🎯 CONFUSION MATRICES (when ground truth available):\")\n",
                "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
                "fig.suptitle('🎯 Confusion Matrices for All Approaches', fontsize=16, fontweight='bold')\n",
                "\n",
                "axes_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]\n",
                "for ax in axes_list:\n",
                "    ax.axis('off')\n",
                "\n",
                "for idx, (name, data) in enumerate(list(approaches.items())[:3]):\n",
                "    ax = axes_list[idx]\n",
                "    ax.axis('on')\n",
                "    results_df = data['results']\n",
                "    y_true, y_pred = _get_truth_and_pred(results_df)\n",
                "\n",
                "    if len(y_true) > 0:\n",
                "        labels = MAIN_CATEGORIES\n",
                "        cm = confusion_matrix(y_true, y_pred, labels=labels)\n",
                "        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n",
                "                    xticklabels=labels, yticklabels=labels, ax=ax)\n",
                "        acc = (cm.trace() / cm.sum()) if cm.sum() else 0.0\n",
                "        ax.set_title(f\"{name.split('(')[0].strip()}\\nAccuracy: {acc:.1%}\", fontweight='bold')\n",
                "        ax.set_xlabel('Predicted', fontweight='bold')\n",
                "        ax.set_ylabel('Actual', fontweight='bold')\n",
                "    else:\n",
                "        ax.text(0.5, 0.5, 'No ground truth\\nmatches found', ha='center', va='center', transform=ax.transAxes)\n",
                "        ax.set_title(f\"{name.split('(')[0].strip()}\\nNo Data\", fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Save confusion matrices as artifacts (robust to missing context)\n",
                "if 'SAVE_ARTIFACTS' in globals() and SAVE_ARTIFACTS:\n",
                "    import os\n",
                "    from datetime import datetime\n",
                "    if 'timestamp' not in globals():\n",
                "        timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
                "    if 'artifacts_dir' not in globals():\n",
                "        artifacts_dir = \"artifacts\"\n",
                "    os.makedirs(artifacts_dir, exist_ok=True)\n",
                "\n",
                "    for name, data in approaches.items():\n",
                "        y_true, y_pred = _get_truth_and_pred(data['results'])\n",
                "        if len(y_true) > 0:\n",
                "            cm = confusion_matrix(y_true, y_pred, labels=MAIN_CATEGORIES)\n",
                "            safe_name = name.lower().replace(\" \", \"_\").replace(\"(\", \"\").replace(\")\", \"\")\n",
                "            cm_file = f\"{artifacts_dir}/confusion_matrix_{safe_name}_{timestamp}.csv\"\n",
                "            pd.DataFrame(cm, index=MAIN_CATEGORIES, columns=MAIN_CATEGORIES).to_csv(cm_file)\n",
                "            print(f\"   📊 Saved confusion matrix: {cm_file}\")\n",
                "\n",
                "# Generate top examples for each approach\n",
                "print(\n\"\\n✨ TOP EXAMPLES BY APPROACH:\")\n",
                "for name, data in approaches.items():\n",
                "    categorized = data['categorized']\n",
                "    if len(categorized) > 0:\n",
                "        print(f\"\\n📋 {name}:\")\n",
                "        top_confident = categorized.nlargest(3, 'confidence')\n",
                "        print(\"   🏆 Highest confidence predictions:\")\n",
                "        for _, row in top_confident.iterrows():\n",
                "            print(f\"      • '{row['name'][:50]}...' → {row['predicted_category']} (conf: {row['confidence']:.3f})\")\n",
                "\n",
                "        if 'true_category' in categorized.columns:\n",
                "            correct_mask = (categorized['true_category'] == categorized['predicted_category'])\n",
                "            correct_examples = categorized[correct_mask].head(2)\n",
                "            incorrect_examples = categorized[~correct_mask].head(2)\n",
                "\n",
                "            if len(correct_examples) > 0:\n",
                "                print(\"   ✅ Correct predictions (sample):\")\n",
                "                for _, row in correct_examples.iterrows():\n",
                "                    print(f\"      • '{row['name'][:50]}...' → {row['predicted_category']} ✓\")\n",
                "\n",
                "            if len(incorrect_examples) > 0:\n",
                "                print(\"   ❌ Incorrect predictions (sample):\")\n",
                "                for _, row in incorrect_examples.iterrows():\n",
                "                    print(f\"      • '{row['name'][:50]}...' → {row['predicted_category']} (should be {row['true_category']})\")\n",
                "\n",
                "print(\n\"\\n✅ DETAILED ANALYSIS Complete!\")\n",
            ]
            cell["source"] = new_src_conf
            changed = True

        # Patch Approach 4 per-row inference to true batch classify_batch
        if "APPROACH 4: PURE ZERO-SHOT CLASSIFICATION" in text:
            try:
                loop_start = next(
                    i for i, l in enumerate(src) if "for _, row in batch.iterrows()" in l
                )
                # find an anchor near end of that processing loop; use first occurrence of 'approach4_time'
                loop_end = next(i for i, l in enumerate(src) if "approach4_time" in l)
                new = [
                    "    # True batch classification for each batch\n",
                    "    enhanced_texts = [f\"Product: {row['name']} | Type: office/business item\" for _, row in batch.iterrows()]\n",
                    "    batch_results = zero_shot.classify_batch(enhanced_texts, enhanced_categories, batch_size=batch_size)\n",
                    "    for name, result in zip(batch['name'].tolist(), batch_results):\n",
                    "        try:\n",
                    "            if result.get('labels') and result.get('scores'):\n",
                    "                pred_category = result['labels'][0]\n",
                    "                confidence = result['scores'][0]\n",
                    "            else:\n",
                    "                pred_category, confidence = 'Unclassified', 0.0\n",
                    "            if pred_category == 'Unclassified' or confidence < 0.2:\n",
                    "                pred_category, confidence = 'Uncategorized', 0.0\n",
                    "            elif confidence < 0.4:\n",
                    "                confidence = confidence * 1.4\n",
                    "            elif confidence < 0.6:\n",
                    "                confidence = confidence * 1.2\n",
                    "            approach4_predictions.append(pred_category)\n",
                    "            approach4_confidences.append(min(confidence, 1.0))\n",
                    "            processed += 1\n",
                    "        except Exception as e:\n",
                    "            print(f\"   ⚠️  Error processing '{name[:30]}...': {str(e)[:50]}...\")\n",
                    "            approach4_predictions.append('Uncategorized')\n",
                    "            approach4_confidences.append(0.0)\n",
                    "            processed += 1\n",
                ]
                cell["source"] = src[:loop_start] + new + src[loop_end:]
                changed = True
            except StopIteration:
                pass

    # Remove exact duplicate code cells
    seen = set()
    unique = []
    removed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            unique.append(cell)
            continue
        sig = "".join(cell.get("source") or [])
        key = ("code", sig)
        if key in seen and len(sig) > 0:
            removed += 1
            changed = True
            continue
        seen.add(key)
        unique.append(cell)
    if removed:
        print(f"Removed {removed} duplicate code cells")
    if changed:
        nb["cells"] = unique
        nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Notebook updated")
    else:
        print("No changes needed")
    return changed


if __name__ == "__main__":
    nb_path = Path("notebooks/enhanced_demo.ipynb")
    if not nb_path.exists():
        raise SystemExit("Notebook not found: notebooks/enhanced_demo.ipynb")
    patch_notebook(nb_path)


