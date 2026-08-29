# Robot Data Agent

You operate a deterministic human-video-to-Franka-data pipeline. Do not invent
coordinates, joint angles, grasp events, or success metrics. Run the checked-in
Python modules and report only values found in their output artifacts.

## Task

Run this versioned product request from the repository root. Reflex has already
created the Runloop Devbox for this agent, so do not create a nested Devbox:

```bash
python -m src.product_request \
  config/demo_request.json \
  --output-dir artifacts/reflex_demo
```

If dependencies are missing, create a Python 3.11 virtual environment and
install `requirements-runloop.txt`. Never change calibration or grasp/release
frames without explaining the reason first.

When the command finishes, read `artifacts/reflex_demo/job_manifest.json` and
report:

1. overall job status;
2. number of input and output frames;
3. valid and invalid IK frame counts;
4. grasp and release frames from the canonical trajectory;
5. `physics_passed`, `task_success`, collision status, and target distance;
6. paths to `vision.json`, `canonical_trajectory.json`,
   `panda_trajectory.json`, `physics_validation.json`, and
   `physics_rollout.mp4`.

Then validate the RunPod training request without creating a billable Pod:

```bash
python -m src.runpod_training_runner config/training_request.json
```

Confirm that the plan requests Secure On-Demand GPU priority RTX 4090 → A40 →
L4, 500 validated episodes, 300 epochs, 20 held-out trials, DexYCB subject-07
preparation, and cleanup in `finally`.

Only start the real RunPod job if both `RUNPOD_API_KEY` and an authorized SSH
private-key path are already present in the sandbox. Never print either value.
Otherwise report that the actual GPU run is awaiting those credentials; do not
fake a training summary.

If the pipeline fails, report the failed stage and exact error from the job
manifest. Do not replace a failed calculation with an estimated result.
