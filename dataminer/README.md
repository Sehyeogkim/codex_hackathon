# DataMiner core

`dataminer` contains the submission's video-to-robot-data path. One command
turns one ordinary RGB video into auditable Franka artifacts:

```text
video.mp4
  -> vision.py              frame-aligned right-hand observations
  -> retarget.py            canonical end-effector trajectory
  -> panda_sim.py           Franka 7-joint inverse kinematics
  -> physics_validation.py  MuJoCo mustard-bottle validation and rollout
  -> pipeline.py            artifacts, events and final manifest
```

## Run one video

From the repository root:

```bash
python -m dataminer \
  data/demo2/do_as_i_do_pick_place_preview.mp4 \
  --config dataminer/config/demo_config.json \
  --output-dir artifacts/dataminer_demo \
  --grasp-frame 30 \
  --release-frame 70
```

The command writes:

- `vision.json`: one observation for every source frame, including null
  observations when no right hand is visible;
- `canonical_trajectory.json`: robot-independent EE position, phase and
  gripper width;
- `panda_trajectory.json`: seven Franka joint targets and per-frame IK status;
- `physics_validation.json`: measured simulation gates;
- `physics_rollout.mp4`: bottle motion rendered in MuJoCo;
- `job_manifest.json`: inputs, stage status, artifact paths and summary.

The physics stage uses the human video for task timing and explicit
grasp/release events, then anchors execution to the configured bottle and target
poses. It is physical task validation, not metric 3D reconstruction of the
person or object from one monocular camera.

## DexYCB seed preparation

`dataminer.dexycb_pipeline` contains the RGB-only seed preparation path used by
the training workflow. It only accepts sequences whose metadata proves
`ycb_ids[ycb_grasp_ind] == 5` (mustard bottle) and preserves the distinction
between the RGB-derived human segment and generated carry/place/release frames.

## Package layout

- `pipeline.py`: local end-to-end CLI and artifact manifest.
- `product_request.py` / `product_runloop.py`: validated customer request entry
  points for local and Runloop execution.
- `runloop_runner.py`: minimal remote runtime packaging and Devbox lifecycle.
- `simulation/`: MuJoCo scene, rollout, export, and small BC utilities.
- `tools/`: optional capture and presentation helpers; each module is runnable
  with `python -m dataminer.tools.<name>`.

Runtime models are stored once under `data/resources/`; the package does not
depend on the removed root `src`, `mimic`, `models`, or `vendor` trees.
