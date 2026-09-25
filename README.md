# ML Challenge — Entity Resolution Dataset

Business entity matching / record-linkage task: match business entities across three
independent sources (`source1`, `source2`, `source3`) using name, address, and country.

## Files

All files are tab-separated (`.tsv`) with a header row.

| File | Columns | Lines (incl. header) |
|---|---|---|
| `train_source1.tsv` | `entity_id, business_name, business_address, country` | 2,206,822 |
| `train_source2.tsv` | `entity_id, business_name, business_address, country` | 5,034,617 |
| `train_source3.tsv` | `entity_id, business_name, business_address, country` | 5,285,604 |
| `train_ground_truth.tsv` | `source1_entity_id, matched_entity_ids` | 2,206,822 |
| `test_source1.tsv` | `entity_id, business_name, business_address, country` | 1,732,545 |
| `test_source2.tsv` | `entity_id, business_name, business_address, country` | 4,887,274 |
| `test_source3.tsv` | `entity_id, business_name, business_address, country` | 5,082,317 |

- `entity_id` is prefixed by source, e.g. `S1-965667`, `S2-681193310`, `S3-11291185`.
- `train_ground_truth.tsv` maps each `source1_entity_id` to its matching entity id(s) in
  source2/source3 as a comma-separated list in `matched_entity_ids`
  (e.g. `S2-681193310,S3-775321672,S3-11291185`).
- The task: for each `test_source1.tsv` entity, predict its matching entities in
  `test_source2.tsv` / `test_source3.tsv`.
- Addresses/names may be in non-English scripts (e.g. Hindi) for entities based in India.

## Data storage (why it's chunked)

The raw `.tsv` files are 127MB–510MB each, over GitHub's 100MB per-file limit, so they are
**not stored directly in this repo**. Instead each file is split into <100MB parts under
`data/chunks/<name>/<name>.part_NNN`, split on line boundaries (no row is broken across parts).

To reconstruct the full `.tsv` files after cloning:

```bash
python scripts/reassemble.py
```

This writes the full files to `data/full/*.tsv` (gitignored — regenerate locally, don't commit them).
