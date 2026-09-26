#!/usr/bin/env python3
"""Validate matching_results.tsv / candidate_pairs.tsv against the challenge's
submission rules, using only the standard library.

Not the official grader (it wasn't distributed with this copy of the
dataset) -- this re-implements every rule stated in the problem statement so
a broken submission can be caught locally before it's spent on the leaderboard.

Usage:
    python3 validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir .
"""
import argparse
import csv
import sys
from pathlib import Path


def read_ids(path):
    ids = set()
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r, None)  # header
        for row in r:
            if row:
                ids.add(row[0])
    return ids


def read_id_list_file(path, expected_header):
    """Returns (rows: list[(id, [matches])], header_ok: bool, dup_id_rows: list[str])."""
    rows = []
    seen = set()
    dup_ids = []
    header_ok = True
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r, None)
        if header != expected_header:
            header_ok = False
        for line_no, row in enumerate(r, start=2):
            if not row or row == [""]:
                continue
            if len(row) == 1:
                row = [row[0], ""]
            entity_id, id_list_str = row[0], row[1]
            if entity_id in seen:
                dup_ids.append(entity_id)
            seen.add(entity_id)
            id_list = [x for x in id_list_str.split(",") if x] if id_list_str else []
            rows.append((entity_id, id_list))
    return rows, header_ok, dup_ids


def validate_file(path, expected_header, list_col_name, test_s1_ids, test_s2_ids, test_s3_ids):
    errors = []
    warnings = []

    if not Path(path).exists():
        return [f"{path}: file does not exist"], [], {}

    rows, header_ok, dup_ids = read_id_list_file(path, expected_header)
    if not header_ok:
        errors.append(f"{path}: header must be exactly {expected_header!r}")
    if dup_ids:
        errors.append(f"{path}: {len(dup_ids)} duplicate source1_entity_id rows, e.g. {dup_ids[:5]}")

    row_ids = set(eid for eid, _ in rows)
    missing = test_s1_ids - row_ids
    extra = row_ids - test_s1_ids
    if missing:
        errors.append(f"{path}: {len(missing)} test source1 entities missing from submission, "
                       f"e.g. {list(sorted(missing))[:5]}")
    if extra:
        errors.append(f"{path}: {len(extra)} source1_entity_id values not in the test set, "
                       f"e.g. {list(sorted(extra))[:5]}")

    id_lists = {}
    bad_prefix, unknown_id, dup_in_list, self_ref = [], [], [], []
    for eid, id_list in rows:
        id_lists[eid] = id_list
        if len(id_list) != len(set(id_list)):
            dup_in_list.append(eid)
        for mid in id_list:
            if mid == eid:
                self_ref.append(eid)
            elif mid.startswith("S2-"):
                if mid not in test_s2_ids:
                    unknown_id.append((eid, mid))
            elif mid.startswith("S3-"):
                if mid not in test_s3_ids:
                    unknown_id.append((eid, mid))
            else:
                bad_prefix.append((eid, mid))

    if bad_prefix:
        errors.append(f"{path}: {len(bad_prefix)} {list_col_name} entries with a non-S2-/S3- id, "
                       f"e.g. {bad_prefix[:5]}")
    if unknown_id:
        errors.append(f"{path}: {len(unknown_id)} {list_col_name} entries reference an id not in "
                       f"the test set, e.g. {unknown_id[:5]}")
    if dup_in_list:
        errors.append(f"{path}: {len(dup_in_list)} rows have duplicate ids within their own list, "
                       f"e.g. {dup_in_list[:5]}")
    if self_ref:
        errors.append(f"{path}: {len(self_ref)} rows reference their own source1_entity_id, "
                       f"e.g. {self_ref[:5]}")

    return errors, warnings, id_lists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    def find(name):
        direct = test_dir / name
        nested = test_dir / "test" / name
        return direct if direct.exists() else nested

    test_s1_ids = read_ids(find("test_source1.tsv"))
    test_s2_ids = read_ids(find("test_source2.tsv"))
    test_s3_ids = read_ids(find("test_source3.tsv"))

    all_errors = []
    all_warnings = []

    m_errors, m_warnings, match_lists = validate_file(
        args.matching, ["source1_entity_id", "matched_entity_ids"], "matched_entity_ids",
        test_s1_ids, test_s2_ids, test_s3_ids,
    )
    c_errors, c_warnings, cand_lists = validate_file(
        args.candidate, ["source1_entity_id", "candidate_entity_ids"], "candidate_entity_ids",
        test_s1_ids, test_s2_ids, test_s3_ids,
    )
    all_errors += m_errors + c_errors
    all_warnings += m_warnings + c_warnings

    if not m_errors and not c_errors:
        not_a_candidate = 0
        for eid, matches in match_lists.items():
            cand_set = set(cand_lists.get(eid, []))
            for mid in matches:
                if mid not in cand_set:
                    not_a_candidate += 1
        if not_a_candidate:
            all_warnings.append(
                f"{not_a_candidate} matched ids never appeared as a candidate for their source1 "
                f"entity -- this signals a pipeline bug (matches should be a subset of candidates)"
            )

    if all_warnings:
        print("WARNINGS:")
        for i, w in enumerate(all_warnings, 1):
            print(f"  {i}. {w}")

    if all_errors:
        print("ISSUES TO FIX:")
        for i, e in enumerate(all_errors, 1):
            print(f"  {i}. {e}")
        sys.exit(1)

    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
