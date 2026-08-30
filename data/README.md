# Data

This directory separates small, reproducible demo inputs from large external
datasets and runtime resources.

- `demo2/`: two attributed, compressed videos used for local pipeline tests.
- `dexycb/`: download and selection notes. The subject-07 archive is downloaded
  directly on RunPod and is never committed.
- `resources/`: robot models and hand-landmark assets required by the pipeline.
  Upstream license files are retained beside the assets.

Generated trajectories, datasets, checkpoints, and rollouts belong under
`artifacts/`, which is ignored by Git. DexYCB is CC BY-NC 4.0, so the demo does
not present its output as a commercial training dataset.
