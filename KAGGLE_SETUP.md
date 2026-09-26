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

Before training, raise the blocking settings in `src/blocking.py` to use Kaggle's extra
RAM -- training-data volume plateaus fast (200k vs 400k entities barely moved the score
locally), so the better use of that RAM is a wider blocking net, not a bigger sample:

```python
MAX_KEY_FREQUENCY = 3000      # was 2000 -- OOM safety valve only, ranking quality comes from IDF weighting
MAX_CANDIDATES = 100          # was 60 -- more headroom per entity for the classifier
S1_BATCH_SIZE = 20_000        # was 10,000 -- fewer batches, less looping overhead
```

Then install dependencies and run. Test on a small sample first to sanity check timing
and recall before committing to a multi-hour full run:
```bash
!pip install -q -r code/business_entity_resolution/requirements.txt
%cd code/business_entity_resolution
!python -m src.train --data-dir "../../data/full" --sample 50000   # quick check first
```
Look at the logged `train candidate recall` and elapsed time. If it looks good, scale up
training (400k-800k entities is plenty -- diminishing returns beyond that locally) and
then run the full, mandatory, un-subsampled prediction:
```bash
!python -m src.train --data-dir "../../data/full" --sample 800000
!python -m src.predict --data-dir "../../data/full" --output-dir "../../output"
```

## What to watch for

- Prediction over the full 1.73M test entities takes multiple hours on a CPU laptop;
  Kaggle CPUs may be faster or slower depending on the instance -- benchmark on a small
  `--sample` first before committing to a full run.
- Run this as **Save Version -> Save & Run All (Commit)**, not an interactive session --
  a multi-hour job will get killed if you close the browser tab on an interactive run.
- `utils/validate_submission.py` checks the output format before you submit --
  always run it before uploading `matching_results.tsv`.
- Locally (16GB RAM, MAX_KEY_FREQUENCY=2000, MAX_CANDIDATES=60), best validated result
  was ~0.79 F_0.5 at an 82% blocking-recall-ceiling on a 3K-entity sample -- not yet
  confirmed at full scale. The settings above aim to beat that using Kaggle's RAM headroom.
