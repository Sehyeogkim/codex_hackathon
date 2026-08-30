# Dataminer

Dataminer turns ordinary human task videos into auditable, robot-specific
training data. The submission targets a Franka Panda performing a mustard
bottle pick-and-place task.

The core idea is simple: a customer provides a robot specification and a task;
workers demonstrate that task in front of a fixed camera; Dataminer extracts a
canonical end-effector path, compiles it into the robot's joint space, and
rejects trajectories that fail physics validation. Validated seeds can then be
scaled and used for policy training without teleoperating the target robot for
every demonstration.

## Verified pipeline

```text
RGB task video (fixed third-person capture recommended)
  -> MediaPipe hand observations
  -> canonical end-effector trajectory
  -> Franka Panda 7-DoF inverse kinematics
  -> MuJoCo task and safety gates
  -> JSON trajectories + rollout video
  -> validated episode generation
  -> phase-conditioned behavior cloning on RunPod
```

The checked Demo 1 run processes 91 source frames, solves IK for all 91 frames,
passes collision and joint-limit gates, moves the simulated bottle to within
0.38 mm of its target, and reports `task_success=true`. Its source is a public
Do As I Do preview used as a reproducible smoke test; it is not represented as
a newly recorded proprietary demonstration.

Demo 2 uses two verified right-hand, mustard-bottle sequences from DexYCB
subject-07. RGB-derived pickup segments are kept distinct from MuJoCo-generated
carry/place/release segments. DexYCB pose and depth annotations are used only
for selection and evaluation, not as trajectory inputs.

## Repository map

| Path | Purpose |
| --- | --- |
| [`persona/`](persona/) | Four Reflex agent personas, gate contracts, and session configuration |
| [`data/`](data/) | Small reproducible inputs and instructions for large external datasets/resources |
| [`dataminer/`](dataminer/) | One-video RGB-to-Franka trajectory and MuJoCo validation pipeline |
| [`runpod_workflow_train/`](runpod_workflow_train/) | Ephemeral RunPod provisioning, DexYCB preparation, episode generation, BC training, and evaluation |
| [`presentation/`](presentation/) | Self-contained Korean/English HTML deck and compressed demo media |
| [`document/`](document/) | Product reasoning, capture guide, workflow, dataset, robot, and demo notes |
| [`tests/`](tests/) | Unit, contract, and integration tests |

Secrets live only in the local `.env`; Git tracks only [`.env.example`](.env.example).
Raw DexYCB archives, generated datasets, checkpoints, and runtime artifacts are
also excluded.

## Quick start: one video

Create a Python environment and install the local conversion dependencies, then
run:

```bash
python -m dataminer \
  data/demo2/do_as_i_do_pick_place_preview.mp4 \
  --config dataminer/config/demo_config.json \
  --output-dir artifacts/dataminer_demo \
  --grasp-frame 30 \
  --release-frame 70
```

The output contains frame-aligned observations, a canonical trajectory, Franka
joint targets, physics metrics, a MuJoCo rollout, and an auditable job manifest.

## RunPod training

Validate the non-secret request without creating billable infrastructure:

```bash
python -m runpod_workflow_train
```

Launch the temporary GPU workflow only when explicitly intended:

```bash
set -a
source .env
set +a
python -m runpod_workflow_train \
  --execute \
  --output-dir artifacts/runpod-final \
  --timeout 1800
```

The runner prefers RTX 4090, then A40, then L4; verifies CUDA; downloads DexYCB
inside the Pod; trains and evaluates the policy; recovers artifacts; and deletes
the Pod in a `finally` block on both success and failure. A training run passes
only with exactly 500 validated episodes and at least 10 successes in 20 held-out
trials.

## Reflex and Runloop

The orchestration contract uses four long-lived role sessions, each backed by
its own Reflex-provided Runloop Devbox:

```text
Reconstruction -> Retargeting -> Physical Validation -> Data Scaling
```

The role definitions and honest implementation boundary are documented in
[`persona/README.md`](persona/README.md). The current repository implements and
tests the individual stages; it does not claim that cross-Devbox artifact queues
are already production-complete.

## Presentation

Open [`presentation/index.html`](presentation/index.html) for Korean or
[`presentation/en.html`](presentation/en.html) for English. The deck is static
and can be presented without installing dependencies or running live training.

## Data and licensing

- DexYCB is CC BY-NC 4.0 and is used only for research/hackathon demonstration.
- The included EgoVerse-derived clip is CC BY-SA 4.0; attribution is recorded in
  [`data/demo2/manifest.md`](data/demo2/manifest.md).
- The Do As I Do preview is retained only for attributed pipeline testing; its
  individual redistribution terms are not asserted beyond the source record.
- MuJoCo Menagerie Franka assets retain their upstream license notices.

This prototype demonstrates a data-generation workflow, not guaranteed
zero-shot deployment on a physical robot. Real deployment still requires camera
calibration, robot-specific safety review, and validation with a small amount of
real-robot data.
