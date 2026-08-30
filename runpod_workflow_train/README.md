# RunPod workflow training

This package is the GPU half of the demo. It creates an ephemeral Secure
RunPod, prepares two verified DexYCB subject-07 RGB seeds, generates 500
physics-validated Franka episodes, trains and evaluates the behavior-cloning
policy, downloads the artifacts, and deletes the Pod in a `finally` block.

The package contains no API keys, `.env` files, DexYCB archives, captured
videos, generated datasets, checkpoints, or output artifacts. DexYCB is
downloaded directly inside the temporary Pod and is subject to CC BY-NC 4.0.

## Dry-run

From the repository root:

```bash
python -m runpod_workflow_train
```

This validates `runpod_workflow_train/config/training_request.json` and prints a credential-free
deployment plan. It does not create a Pod.

## Execute

Set the key only in the process environment, then opt in explicitly:

```bash
export RUNPOD_API_KEY='...'
python -m runpod_workflow_train \
  --execute \
  --output-dir artifacts/runpod-final \
  --timeout 1800
```

Use `--ssh-key /path/to/private-key` only when the registered RunPod key is not
one of the SSH client's defaults. The key and API token are never added to the
request archive or manifest.

GPU preference is RTX 4090, then A40, then L4. The request uses one GPU, a
50 GB container disk, and a 100 GB workspace volume. The command exits `0`
only when all acceptance gates pass, including CUDA metadata, 500 validated
episodes, at least 10 successes in 20 held-out trials, and two rollout videos.

The remote command exports `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` before
MuJoCo is imported. If infrastructure fails after the checkpoint is saved, the
held-out evaluation can be resumed without retraining:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m runpod_workflow_train.resume_evaluation \
  --checkpoint /workspace/output/policy.pt \
  --out /workspace/output \
  --training-log /workspace/output/training.log \
  --seed-dir /workspace/output/dexycb/seeds \
  --trials 20
```

## Package layout

- `runner.py`: RunPod REST API, SSH/SCP transport, archive and artifact gates.
- `dexycb_prepare.py`: selects two verified subject-07 sequences and produces
  provenance-preserving hybrid trajectories using `dataminer.dexycb_pipeline`.
- `train_policy.py`: physics augmentation, phase-conditioned BC, evaluation.
- `mimic/`: MuJoCo generation, rollout, validation, BC, and video utilities.
- `config/`: non-secret training request and Franka workspace calibration.
- `download_dexycb.sh`: resumable in-Pod Hugging Face/Xet dataset download.
- `requirements.txt`: dependencies installed in the Pod image.

The upload archive also includes the repository's shared, non-training-data
resources at `data/resources/franka_emika_panda` and
`data/resources/hand_landmarker.task`. They are referenced in place rather than
duplicated inside this package.

## Tests

```bash
python -m unittest \
  tests.test_runpod_training_runner \
  tests.test_runpod_dexycb_runner \
  tests.test_demo2_train_policy \
  tests.test_resume_evaluation -v
```
