"""Pairwise similarity feature computation for candidate (source1, other) pairs."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from . import normalize as norm

FEATURE_COLUMNS = [
    "name_ratio", "name_token_sort_ratio", "name_partial_ratio", "compact_name_ratio",
    "name_token_jaccard", "addr_ratio", "addr_token_sort_ratio", "addr_token_jaccard",
    "digit_jaccard", "country_match", "name_len_1", "name_len_2", "name_len_diff",
    "addr_len_1", "addr_len_2", "addr_missing_1", "addr_missing_2", "other_source",
]

_S1_COLS = ["entity_id", "norm_name", "norm_address", "country_norm"]


def _prep_side(df, suffix):
    return df[_S1_COLS].rename(columns={c: f"{c}_{suffix}" if c != "entity_id" else f"entity_id_{suffix}"
                                         for c in _S1_COLS})


def compute_features(pairs_df, s1_df, other_df, other_source):
    """pairs_df: [source1_entity_id, other_entity_id]. other_source: 2 or 3,
    identifying whether other_df is source2 or source3. Returns pairs_df with
    feature columns appended (row order preserved). Token/digit sets are
    derived per-row inside the loop below -- not stored as columns on the
    (multi-million row) source frames, to keep memory usage bounded."""
    left = _prep_side(s1_df, "1")
    right = _prep_side(other_df, "2")

    merged = pairs_df.merge(left, left_on="source1_entity_id", right_on="entity_id_1", how="left")
    merged = merged.merge(right, left_on="other_entity_id", right_on="entity_id_2", how="left")

    n = len(merged)
    name1 = merged["norm_name_1"].to_numpy()
    name2 = merged["norm_name_2"].to_numpy()
    addr1 = merged["norm_address_1"].to_numpy()
    addr2 = merged["norm_address_2"].to_numpy()

    name_ratio = np.empty(n, dtype=np.float32)
    name_token_sort_ratio = np.empty(n, dtype=np.float32)
    name_partial_ratio = np.empty(n, dtype=np.float32)
    compact_name_ratio = np.empty(n, dtype=np.float32)
    addr_ratio = np.empty(n, dtype=np.float32)
    addr_token_sort_ratio = np.empty(n, dtype=np.float32)
    name_token_jaccard = np.empty(n, dtype=np.float32)
    addr_token_jaccard = np.empty(n, dtype=np.float32)
    digit_jaccard = np.empty(n, dtype=np.float32)

    for i in range(n):
        n1, n2 = name1[i], name2[i]
        a1, a2 = addr1[i], addr2[i]
        name_ratio[i] = fuzz.ratio(n1, n2)
        name_token_sort_ratio[i] = fuzz.token_sort_ratio(n1, n2)
        name_partial_ratio[i] = fuzz.partial_ratio(n1, n2)
        compact_name_ratio[i] = fuzz.ratio(norm.compact(n1), norm.compact(n2))
        addr_ratio[i] = fuzz.ratio(a1, a2)
        addr_token_sort_ratio[i] = fuzz.token_sort_ratio(a1, a2)
        name_token_jaccard[i] = norm.jaccard(norm.tokens(n1), norm.tokens(n2))
        addr_token_jaccard[i] = norm.jaccard(norm.tokens(a1), norm.tokens(a2))
        digit_jaccard[i] = norm.jaccard(norm.digit_tokens(a1), norm.digit_tokens(a2))

    merged["name_ratio"] = name_ratio / 100.0
    merged["name_token_sort_ratio"] = name_token_sort_ratio / 100.0
    merged["name_partial_ratio"] = name_partial_ratio / 100.0
    merged["compact_name_ratio"] = compact_name_ratio / 100.0
    merged["addr_ratio"] = addr_ratio / 100.0
    merged["addr_token_sort_ratio"] = addr_token_sort_ratio / 100.0
    merged["name_token_jaccard"] = name_token_jaccard
    merged["addr_token_jaccard"] = addr_token_jaccard
    merged["digit_jaccard"] = digit_jaccard
    merged["country_match"] = (merged["country_norm_1"] == merged["country_norm_2"]).astype(np.int8)

    merged["name_len_1"] = merged["norm_name_1"].str.len().fillna(0)
    merged["name_len_2"] = merged["norm_name_2"].str.len().fillna(0)
    merged["name_len_diff"] = (merged["name_len_1"] - merged["name_len_2"]).abs()
    merged["addr_len_1"] = merged["norm_address_1"].str.len().fillna(0)
    merged["addr_len_2"] = merged["norm_address_2"].str.len().fillna(0)
    merged["addr_missing_1"] = (merged["addr_len_1"] == 0).astype(np.int8)
    merged["addr_missing_2"] = (merged["addr_len_2"] == 0).astype(np.int8)
    merged["other_source"] = np.int8(other_source)

    keep = ["source1_entity_id", "other_entity_id"] + FEATURE_COLUMNS
    return merged[keep]
