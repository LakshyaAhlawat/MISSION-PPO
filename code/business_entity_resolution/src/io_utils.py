"""Loading and normalization helpers for the source/ground-truth TSVs.

Normalization results for source2/source3 (the large, 5M+ row files) are
cached to parquet next to the raw TSV so repeated dev/train runs don't pay
the text-normalization cost every time.

Memory note: on a 16GB-RAM machine, holding both the raw and normalized text
columns (plus token-set columns) for 10M+ rows at once is what caused the
pipeline to thrash. `add_normalized_columns` therefore returns a SLIM frame
-- entity_id, country_norm, norm_name, norm_address only -- and drops the
raw business_name/business_address/country columns once they've been used.
"""
from pathlib import Path

import pandas as pd

from . import normalize as norm


def load_source(path):
    df = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_values=[],
        quoting=3,  # csv.QUOTE_NONE -- addresses/names may contain stray quote chars
        engine="c",
    )
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("")
    return df


def load_ground_truth(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[], quoting=3)
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")
    return df


def add_normalized_columns(df):
    """Returns a new SLIM frame: entity_id, country_norm, norm_name, norm_address."""
    out = pd.DataFrame({
        "entity_id": df["entity_id"].to_numpy(),
        "country_norm": df["country"].str.strip().str.lower().astype("category"),
        "norm_name": df["business_name"].map(norm.normalize_name),
        "norm_address": df["business_address"].map(norm.normalize_address),
    })
    return out


def load_source_normalized_cached(raw_path, cache_dir=None):
    """Load + normalize a source file, caching the (slim) normalized result
    as parquet. Re-normalizes automatically if the raw TSV is newer."""
    raw_path = Path(raw_path)
    cache_dir = Path(cache_dir) if cache_dir else raw_path.parent / "data" / "normalized_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (raw_path.stem + ".parquet")

    if cache_path.exists() and cache_path.stat().st_mtime >= raw_path.stat().st_mtime:
        return pd.read_parquet(cache_path)

    df = load_source(raw_path)
    normalized = add_normalized_columns(df)
    del df
    normalized.to_parquet(cache_path, index=False)
    return normalized
