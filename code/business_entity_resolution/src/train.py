"""Train the entity-matching classifier and tune the decision threshold.

Usage (from code/business_entity_resolution/):
    python -m src.train --data-dir "../.." --sample 50000   # quick smoke test
    python -m src.train --data-dir "../.."                  # full run

Pipeline:
  1. Load train_source1/2/3 + train_ground_truth, normalize text fields.
  2. Split source1 entities into train/val (entity-level split -> no leakage).
  3. Blocking: generate candidate (source1, other) pairs for both splits
     against the FULL source2/source3 pools (a true match can be anywhere).
  4. Label candidates from the ground truth; compute similarity features.
  5. Train a LightGBM binary classifier (match probability).
  6. On the val split, sweep thresholds and pick the one maximizing macro
     F_0.5 (the actual competition metric) -- not accuracy or AUC.
  7. Persist model + threshold + feature list + metrics to models/.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from . import blocking, features, io_utils, metrics

REPO_ROOT = Path(__file__).resolve().parents[3]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_gt_dict(gt_df):
    gt = {}
    for sid, matches in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        gt[sid] = set(m for m in matches.split(",") if m) if matches else set()
    return gt


def label_pairs(pairs_df, gt_dict):
    labels = [
        1 if o in gt_dict.get(s, set()) else 0
        for s, o in zip(pairs_df["source1_entity_id"], pairs_df["other_entity_id"])
    ]
    return np.array(labels, dtype=np.int8)


def candidate_recall(pairs_df, s1_ids, gt_dict):
    have = pairs_df.groupby("source1_entity_id")["other_entity_id"].apply(set)
    total_true, found_true = 0, 0
    for sid in s1_ids:
        true_set = gt_dict.get(sid, set())
        if not true_set:
            continue
        cand = have.get(sid, set())
        total_true += len(true_set)
        found_true += len(true_set & cand)
    return found_true / total_true if total_true else 1.0


def build_candidate_features(s1_df, s2_df, s3_df, s2_index, s3_index):
    cand2 = blocking.generate_candidates_batched(s1_df, other_index=s2_index)
    cand3 = blocking.generate_candidates_batched(s1_df, other_index=s3_index)
    log(f"  blocking: {len(cand2):,} candidates vs source2, {len(cand3):,} vs source3")

    feat2 = features.compute_features(cand2, s1_df, s2_df, other_source=2)
    feat3 = features.compute_features(cand3, s1_df, s3_df, other_source=3)
    feat = pd.concat([feat2, feat3], ignore_index=True)
    return feat, pd.concat([cand2, cand3], ignore_index=True)


def tune_threshold(val_pairs, val_probs, val_s1_ids, gt_dict):
    val_pairs = val_pairs.copy()
    val_pairs["prob"] = val_probs
    best_t, best_f = 0.5, -1.0
    results = []
    for t in np.arange(0.30, 0.96, 0.02):
        matched = val_pairs[val_pairs["prob"] >= t]
        pred = matched.groupby("source1_entity_id")["other_entity_id"].apply(set).to_dict()
        f = metrics.f_beta_macro(gt_dict, pred, val_s1_ids, beta=0.5)
        results.append((round(float(t), 2), round(f, 5)))
        if f > best_f:
            best_f, best_t = f, round(float(t), 2)
    return best_t, best_f, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT))
    ap.add_argument("--sample", type=int, default=None, help="subsample N source1 rows for a quick run")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-out", default=str(REPO_ROOT / "models" / "model.pkl"))
    ap.add_argument("--meta-out", default=str(REPO_ROOT / "models" / "model_meta.json"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    log("Loading + normalizing source1 (uncached -- may be subsampled below)...")
    s1 = io_utils.load_source(data_dir / "train_source1.tsv")
    gt = io_utils.load_ground_truth(data_dir / "train_ground_truth.tsv")
    log(f"  source1={len(s1):,} ground_truth={len(gt):,}")

    if args.sample:
        s1 = s1.sample(n=min(args.sample, len(s1)), random_state=args.seed).reset_index(drop=True)
        log(f"  sampled source1 down to {len(s1):,} rows")
    s1 = io_utils.add_normalized_columns(s1)

    log("Loading + normalizing source2/source3 (cached to parquet after first run)...")
    s2 = io_utils.load_source_normalized_cached(data_dir / "train_source2.tsv")
    s3 = io_utils.load_source_normalized_cached(data_dir / "train_source3.tsv")
    log(f"  source2={len(s2):,} source3={len(s3):,}")
    gt_dict = build_gt_dict(gt)

    train_ids, val_ids = train_test_split(
        s1["entity_id"].tolist(), test_size=args.val_frac, random_state=args.seed
    )
    train_ids, val_ids = set(train_ids), set(val_ids)
    s1_train = s1[s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)
    log(f"  train entities={len(s1_train):,}  val entities={len(s1_val):,}")

    log("Building the source2/source3 blocking index once (reused for train + val)...")
    s2_index = blocking.build_other_index(s2)
    s3_index = blocking.build_other_index(s3)

    log("Building TRAIN candidates + features...")
    train_feat, train_pairs = build_candidate_features(s1_train, s2, s3, s2_index, s3_index)
    train_labels = label_pairs(train_pairs, gt_dict)
    log(f"  train pairs={len(train_feat):,}  positives={train_labels.sum():,} "
        f"({100*train_labels.mean():.3f}%)")
    log(f"  train candidate recall = {candidate_recall(train_pairs, s1_train['entity_id'], gt_dict):.4f}")

    log("Building VAL candidates + features...")
    val_feat, val_pairs = build_candidate_features(s1_val, s2, s3, s2_index, s3_index)
    val_recall = candidate_recall(val_pairs, s1_val["entity_id"], gt_dict)
    log(f"  val pairs={len(val_feat):,}  val candidate recall (blocking ceiling) = {val_recall:.4f}")

    X_train = train_feat[features.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    X_val = val_feat[features.FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    log("Training LightGBM classifier...")
    clf = LGBMClassifier(
        n_estimators=400,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(X_train, train_labels)

    log("Scoring val set and tuning threshold for macro F_0.5...")
    val_probs = clf.predict_proba(X_val)[:, 1]
    best_t, best_f, sweep = tune_threshold(val_pairs, val_probs, s1_val["entity_id"].tolist(), gt_dict)
    log(f"  BEST threshold={best_t}  val macro F_0.5={best_f:.5f}")
    log(f"  (blocking recall ceiling was {val_recall:.4f} -- F_0.5 cannot exceed this)")

    importances = sorted(
        zip(features.FEATURE_COLUMNS, clf.feature_importances_.tolist()),
        key=lambda x: -x[1],
    )
    log("Feature importances: " + ", ".join(f"{n}={v}" for n, v in importances))

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump(clf, f)
    meta = {
        "threshold": best_t,
        "val_f0.5": best_f,
        "val_candidate_recall_ceiling": val_recall,
        "feature_columns": features.FEATURE_COLUMNS,
        "threshold_sweep": sweep,
        "feature_importances": importances,
        "sample": args.sample,
        "seed": args.seed,
    }
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"Saved model to {args.model_out} and metadata to {args.meta_out}")


if __name__ == "__main__":
    main()
