"""Generate candidate_pairs.tsv and matching_results.tsv for the test set.

Processes test_source1 in batches and writes both output files
incrementally, rather than building one giant in-memory table for all 1.7M+
test entities -- a full-size single blocking merge on a skewed key (e.g. a
common address word) can fail outright with an out-of-memory error on this
machine's 16GB RAM, not just run slowly (this bit us once already on the
training side; see blocking.generate_candidates_batched).

Usage (from code/business_entity_resolution/):
    python -m src.predict --data-dir "../.." --output-dir "../../output"
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import blocking, features, io_utils

REPO_ROOT = Path(__file__).resolve().parents[3]
BATCH_SIZE = 20_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT))
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "output"))
    ap.add_argument("--model", default=str(REPO_ROOT / "models" / "model.pkl"))
    ap.add_argument("--meta", default=str(REPO_ROOT / "models" / "model_meta.json"))
    ap.add_argument("--threshold", type=float, default=None, help="override the tuned threshold")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.meta) as f:
        meta = json.load(f)
    threshold = args.threshold if args.threshold is not None else meta["threshold"]
    feature_cols = meta["feature_columns"]
    with open(args.model, "rb") as f:
        clf = pickle.load(f)
    log(f"Loaded model. Using threshold={threshold}")

    log("Loading + normalizing test data (source2/3 cached to parquet after first run)...")
    s1 = io_utils.load_source(data_dir / "test_source1.tsv")
    s1 = io_utils.add_normalized_columns(s1)
    s2 = io_utils.load_source_normalized_cached(data_dir / "test_source2.tsv")
    s3 = io_utils.load_source_normalized_cached(data_dir / "test_source3.tsv")
    log(f"  test_source1={len(s1):,} test_source2={len(s2):,} test_source3={len(s3):,}")

    log("Building source2/source3 blocking indices...")
    s2_index = blocking.build_other_index(s2)
    s3_index = blocking.build_other_index(s3)

    cand_path = out_dir / "candidate_pairs.tsv"
    match_path = out_dir / "matching_results.tsv"
    cand_f = open(cand_path, "w", encoding="utf-8", newline="")
    match_f = open(match_path, "w", encoding="utf-8", newline="")
    cand_f.write("source1_entity_id\tcandidate_entity_ids\n")
    match_f.write("source1_entity_id\tmatched_entity_ids\n")

    n = len(s1)
    total_matched_entities = 0
    total_candidates = 0
    t0 = time.time()
    for start in range(0, n, BATCH_SIZE):
        batch = s1.iloc[start:start + BATCH_SIZE].reset_index(drop=True)
        batch_ids = batch["entity_id"].tolist()

        cand2 = blocking.generate_candidates(batch, other_index=s2_index)
        cand3 = blocking.generate_candidates(batch, other_index=s3_index)
        feat2 = features.compute_features(cand2, batch, s2, other_source=2)
        feat3 = features.compute_features(cand3, batch, s3, other_source=3)
        feat = pd.concat([feat2, feat3], ignore_index=True)

        if len(feat):
            X = feat[feature_cols].to_numpy(dtype=np.float32)
            feat["prob"] = clf.predict_proba(X)[:, 1]
            cand_lists = feat.groupby("source1_entity_id")["other_entity_id"].apply(
                lambda s: sorted(set(s))
            ).to_dict()
            matched = feat[feat["prob"] >= threshold]
            match_lists = matched.groupby("source1_entity_id")["other_entity_id"].apply(
                lambda s: sorted(set(s))
            ).to_dict()
        else:
            cand_lists, match_lists = {}, {}

        for eid in batch_ids:
            c = cand_lists.get(eid, [])
            m = match_lists.get(eid, [])
            cand_f.write(f"{eid}\t{','.join(c)}\n")
            match_f.write(f"{eid}\t{','.join(m)}\n")
            total_candidates += len(c)
            if m:
                total_matched_entities += 1

        done = min(start + BATCH_SIZE, n)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta_min = (n - done) / rate / 60 if rate > 0 else float("nan")
        log(f"  {done:,}/{n:,} test entities ({100*done/n:.1f}%) -- "
            f"{rate:.1f} entities/sec, ETA {eta_min:.1f} min")

    cand_f.close()
    match_f.close()
    log(f"Done. {total_matched_entities:,}/{n:,} test entities got at least one match "
        f"({total_candidates:,} total candidates written).")
    log(f"Outputs: {match_path} and {cand_path}")


if __name__ == "__main__":
    main()
