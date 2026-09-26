# Running this pipeline on Kaggle

This repo's data is chunked (see `README.md`) to fit GitHub's 100MB file limit, and
the pipeline in `code/business_entity_resolution/` was built/tuned on a 16GB-RAM
laptop, which forced some memory-saving compromises (batched blocking, a capped
training sample). Kaggle's free CPU notebooks give ~30GB RAM, which removes most of
that ceiling -- e.g. training on the full 2.2M source1 entities instead of a subsample.

## Setup (in a new Kaggle Notebook)

```bash
!git clone https://github.com/LakshyaAhlawat/MISSION-PPO.git
%cd MISSION-PPO
!python scripts/reassemble.py
```
This reconstructs the full `.tsv` files into `data/full/` from the committed chunks.

Then install dependencies and point the pipeline at the reassembled data:
```bash
!pip install -q -r code/business_entity_resolution/requirements.txt
%cd code/business_entity_resolution
!python -m src.train --data-dir "../../data/full"   # omitting --sample uses ALL 2.2M entities
!python -m src.predict --data-dir "../../data/full" --output-dir "../../output"
```

Note: `train.py`'s `--sample` flag subsamples source1 training entities for memory
safety on constrained machines. On Kaggle's larger RAM, you can likely omit `--sample`
entirely to train on all 2,206,821 entities -- try it, and back off to e.g.
`--sample 1000000` if you still hit a memory error.

## What to watch for

- `src/blocking.py` has `MAX_KEY_FREQUENCY` (currently 2000) as an OOM safety valve for
  the blocking join, and `MAX_CANDIDATES` (60) capping candidates per entity. Both can
  likely be raised on Kaggle's extra RAM for better recall -- that's the main lever for
  pushing macro F_0.5 higher than what was achieved locally (~0.79 validated F_0.5 at
  82% blocking-recall-ceiling on a 3K-entity sample; not yet confirmed at full scale).
- Prediction over the full 1.73M test entities takes multiple hours on a CPU laptop;
  Kaggle CPUs may be faster or slower depending on the instance -- benchmark on a small
  `--sample` first before committing to a full run.
- `utils/validate_submission.py` checks the output format before you submit --
  always run it before uploading `matching_results.tsv`.
