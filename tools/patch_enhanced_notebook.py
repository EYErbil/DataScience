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
        print("✅ Notebook updated")
    else:
        print("ℹ️ No changes needed")
    return changed


if __name__ == "__main__":
    nb_path = Path("notebooks/enhanced_demo.ipynb")
    if not nb_path.exists():
        raise SystemExit("Notebook not found: notebooks/enhanced_demo.ipynb")
    patch_notebook(nb_path)


