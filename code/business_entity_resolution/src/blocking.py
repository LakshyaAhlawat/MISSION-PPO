"""Candidate generation (blocking) via inverted-index joins.

Three complementary blocking signals, unioned:
  1. Name-token blocking: candidates share at least one normalized name word,
     within the same country. Works for same-script typo/reorder noise.
  2. Address-token blocking: candidates share at least one address word,
     within the same country. Catches landmark-based addresses with no
     digits, and cases where the name is unhelpful (cross-script).
  3. Address-digit blocking: candidates share at least one numeric token
     (house/street number, PIN code, ...) from the address. This is the one
     that rescues cross-script matches (e.g. an Indian entity in English vs.
     Tamil/Hindi/Kannada script) because street numbers and PIN codes stay
     largely intact across scripts even when the business name is fully
     transliterated.

Ranking (important -- this is the fix for a real recall bug found by
diagnosis): a diagnostic over 20,000 real training matches showed that
99.95% of true matches share SOME token/digit with their source1 entity --
so blocking's raw recall ceiling is basically 100%. The ~58% recall actually
observed in early runs was self-inflicted: candidates were ranked by raw
COUNT of shared keys, which lets a false candidate sharing several common
words (e.g. "street", "road", "private") outrank a true match that only
shares one rare, distinctive token (a surname, a house number) -- and the
true match then gets cut by the per-entity candidate cap. Candidates are now
ranked by summed INVERSE-DOCUMENT-FREQUENCY weight instead of raw count, so
sharing one rare token beats sharing three common ones, matching how a human
would judge it.

Scale design (this runs on a 16GB-RAM single machine over 2.2M x 10M+ rows):
  - All three signals use pandas merges (vectorized hash joins), not
    per-row Python loops.
  - The "other" side (source2/source3, fixed at ~5M rows) is expensive to
    explode + weight. That work is independent of which source1 entities are
    being queried, so it's done ONCE per other_df via `build_other_index` and
    reused across train/val calls.
  - A hard MAX_KEY_FREQUENCY still exists purely as an OOM safety valve
    (dropping only truly pathological, near-universal keys) -- quality
    control is now the IDF weighting, not this threshold.
  - `generate_candidates_batched` processes source1 in chunks so no single
    merge() has to materialize a huge intermediate result at once (a
    full-size unbatched merge on a skewed key previously crashed with an
    out-of-memory error).
"""
import numpy as np
import pandas as pd

from . import normalize as norm

MAX_KEY_FREQUENCY = 2000  # OOM safety valve only -- ranking quality now comes from IDF weighting
MIN_TOKEN_LEN = 3
MAX_CANDIDATES = 60  # keep only the top-N other-side candidates per source1 entity, by IDF score
S1_BATCH_SIZE = 10_000  # process source1 rows in batches to bound peak merge memory


def _token_keys(text_series):
    return text_series.map(
        lambda s: [t for t in s.split() if len(t) >= MIN_TOKEN_LEN] if s else []
    )


def _digit_keys(norm_address_series):
    return norm_address_series.map(lambda s: list(norm.digit_tokens(s)))


def _explode(df, key_lists):
    tmp = pd.DataFrame({"entity_id": df["entity_id"].to_numpy(),
                         "country": df["country_norm"].to_numpy(),
                         "key": key_lists})
    tmp = tmp.explode("key")
    return tmp.dropna(subset=["key"])


def _weight_and_prune(exploded, max_freq=MAX_KEY_FREQUENCY):
    """Drop only pathologically common keys (OOM safety valve); attach an
    inverse-frequency weight to every surviving (country, key) so rarer,
    more distinctive keys count for more when ranking candidates."""
    freq = exploded.groupby(["country", "key"]).size().rename("freq").reset_index()
    freq = freq[freq["freq"] <= max_freq]
    freq["weight"] = 1.0 / np.log1p(freq["freq"].to_numpy() + 1.0)
    return exploded.merge(freq[["country", "key", "weight"]], on=["country", "key"], how="inner")


def build_other_index(other_df):
    """Precompute + weight the "other" side's exploded key tables once, so
    they can be reused across multiple s1 subsets (e.g. train and val)."""
    return {
        "name": _weight_and_prune(_explode(other_df, _token_keys(other_df["norm_name"]))),
        "addr": _weight_and_prune(_explode(other_df, _token_keys(other_df["norm_address"]))),
        "digit": _weight_and_prune(_explode(other_df, _digit_keys(other_df["norm_address"]))),
    }


def _join_and_rename(s1_exploded, other_exploded):
    pairs = s1_exploded.merge(other_exploded, on=["country", "key"], suffixes=("_1", "_2"))
    return pairs[["entity_id_1", "entity_id_2", "weight"]].rename(
        columns={"entity_id_1": "source1_entity_id", "entity_id_2": "other_entity_id"}
    )


def generate_candidates(s1_df, other_df=None, other_index=None):
    """Provide either other_df (index built fresh) or a precomputed
    other_index from build_other_index (preferred when calling this more
    than once against the same other_df, e.g. train then val).
    Returns unique [source1_entity_id, other_entity_id], capped to the top
    MAX_CANDIDATES per source1 entity by summed IDF weight of shared keys."""
    if other_index is None:
        other_index = build_other_index(other_df)

    s1_name = _explode(s1_df, _token_keys(s1_df["norm_name"]))
    s1_addr = _explode(s1_df, _token_keys(s1_df["norm_address"]))
    s1_digit = _explode(s1_df, _digit_keys(s1_df["norm_address"]))

    name_pairs = _join_and_rename(s1_name, other_index["name"])
    addr_pairs = _join_and_rename(s1_addr, other_index["addr"])
    digit_pairs = _join_and_rename(s1_digit, other_index["digit"])

    all_pairs = pd.concat([name_pairs, addr_pairs, digit_pairs], ignore_index=True)
    scores = (
        all_pairs.groupby(["source1_entity_id", "other_entity_id"])["weight"]
        .sum()
        .rename("score")
        .reset_index()
    )
    scores = scores.sort_values(["source1_entity_id", "score"], ascending=[True, False])
    top = scores.groupby("source1_entity_id", sort=False).head(MAX_CANDIDATES)
    return top[["source1_entity_id", "other_entity_id"]].reset_index(drop=True)


def generate_candidates_batched(s1_df, other_index, batch_size=S1_BATCH_SIZE, on_batch=None):
    """Same contract as generate_candidates, but processes s1_df in chunks so
    no single merge() has to materialize a huge intermediate result at once
    -- a full-size (2M+ row) single merge on a skewed key (e.g. a common
    address word) can fail outright with an out-of-memory error, not just
    run slowly. If `on_batch(batch_result)` is given, each batch's result is
    passed to it instead of being accumulated (use this for streaming output
    in prediction, where nothing needs to be held in memory across batches);
    otherwise all batches are concatenated and returned."""
    results = [] if on_batch is None else None
    n = len(s1_df)
    for start in range(0, n, batch_size):
        batch = s1_df.iloc[start:start + batch_size]
        batch_result = generate_candidates(batch, other_index=other_index)
        if on_batch is not None:
            on_batch(batch_result)
        else:
            results.append(batch_result)
    if on_batch is not None:
        return None
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["source1_entity_id", "other_entity_id"]
    )
