"""Phase-conditioned behaviour cloning on the generated dataset.

Two design choices matter more than anything else here, and both are standard
practice in modern manipulation policies:

* **Relative observations.** The policy is handed the vectors from the gripper to
  the object and to the target, not just their absolute coordinates. Regressing on
  absolutes forces the network to learn forward kinematics before it can even begin
  the task.
* **Explicit phase.** A one-hot task phase and within-phase progress remove the
  ambiguity between approach, carry, release, and retreat.
* **Closed-loop absolute targets.** The policy predicts one absolute joint target
  and is queried again every control step. This avoids accumulated delta error.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn as nn

CHUNK = 1       # re-plan every control step
ACT_DIM = 8     # 7 joint deltas + gripper command
PHASE_COUNT = 7
PHASE_NAMES = (
    "approach", "grasp", "lift", "carry", "place", "release", "retreat"
)
PHASE_EDGES = np.asarray([0.0, 0.20, 0.30, 0.45, 0.70, 0.85, 0.93, 1.0])


def phase_at(step: int, total_steps: int) -> tuple[int, float]:
    """Return deterministic phase id and normalized progress for one episode step."""

    progress = float(np.clip(step / max(total_steps - 1, 1), 0.0, 1.0))
    phase = min(int(np.searchsorted(PHASE_EDGES[1:], progress, side="right")),
                PHASE_COUNT - 1)
    lo, hi = PHASE_EDGES[phase], PHASE_EDGES[phase + 1]
    phase_progress = float(np.clip((progress - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
    return phase, phase_progress


def make_obs(joints, grip_width, ee_pos, obj_pos, target_xy,
             phase_id: int = 0, phase_progress: float = 0.0) -> np.ndarray:
    """The policy's view of the world -- absolute state plus task-relative vectors."""
    joints = np.asarray(joints, np.float32)
    ee_pos = np.asarray(ee_pos, np.float32)
    obj_pos = np.asarray(obj_pos, np.float32)
    target_xy = np.asarray(target_xy, np.float32)
    if not 0 <= int(phase_id) < PHASE_COUNT:
        raise ValueError(f"phase_id must be in [0, {PHASE_COUNT})")
    phase = np.zeros(PHASE_COUNT, np.float32)
    phase[int(phase_id)] = 1.0
    return np.concatenate([
        joints, [grip_width], ee_pos, obj_pos, target_xy,
        obj_pos - ee_pos,                       # gripper -> object
        target_xy - ee_pos[:2],                 # gripper -> target
        obj_pos[:2] - target_xy,                # object -> target
        phase, [float(np.clip(phase_progress, 0.0, 1.0))],
    ]).astype(np.float32)


OBS_DIM = 7 + 1 + 3 + 3 + 2 + 3 + 2 + 2 + PHASE_COUNT + 1  # 31


def build_arrays(episodes, chunk=CHUNK) -> tuple[np.ndarray, np.ndarray]:
    """Flatten episodes into (obs, action-chunk) pairs.

    The label is the delta from the *achieved* joint state at time t to the joint
    target *commanded* at t+1 -- which is exactly what the policy applies at run
    time. Regressing on achieved-to-achieved deltas instead teaches the policy to
    under-command by however much the position controller lags, and the arm then
    stops short of the object on every rollout.
    """
    obs, act = [], []
    for ep in episodes:
        tgt = np.asarray(ep.scene["target_xy"], np.float32)
        n = len(ep) - 1
        if n < chunk + 2:
            continue
        qtarget = ep.qcmd[1:n + 1]                              # absolute target
        grip = ep.ee_command[1:n + 1, 7:8]
        step = np.concatenate([qtarget, grip], axis=1)           # (n, 8)

        for t in range(n - chunk):
            phase_id, phase_progress = phase_at(t, n)
            obs.append(make_obs(ep.qpos[t, :7], ep.qpos[t, 7],
                                ep.ee_actual[t, :3], ep.bottle_pos[t], tgt,
                                phase_id, phase_progress))
            act.append(step[t:t + chunk].reshape(-1))
    if not obs:
        raise ValueError("no episodes long enough to train on")
    return np.asarray(obs, np.float32), np.asarray(act, np.float32)


class MLP(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM * CHUNK, hidden=512, depth=3):
        super().__init__()
        layers, d = [], obs_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, act_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclasses.dataclass
class Policy:
    """A trained policy plus the normalisation it was trained under."""

    model: MLP
    obs_mean: np.ndarray
    obs_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray
    chunk: int = CHUNK
    action_mode: str = "absolute_joint"
    horizon: int = 187

    def chunk_of(self, obs: np.ndarray) -> np.ndarray:
        """Return the next `chunk` actions as (chunk, 8)."""
        device = next(self.model.parameters()).device
        x = torch.from_numpy(((obs - self.obs_mean) / self.obs_std).astype(np.float32)).to(device)
        with torch.no_grad():
            y = self.model(x.unsqueeze(0)).squeeze(0).cpu().numpy()
        return (y * self.act_std + self.act_mean).reshape(self.chunk, ACT_DIM)

    def save(self, path):
        torch.save({"state_dict": self.model.state_dict(), "chunk": self.chunk,
                    "horizon": self.horizon,
                    "obs_mean": self.obs_mean, "obs_std": self.obs_std,
                    "act_mean": self.act_mean, "act_std": self.act_std}, path)


def train(episodes, epochs=200, batch_size=256, lr=1e-3, hidden=512, depth=3,
          val_frac=0.1, seed=0, chunk=CHUNK, obs_noise=0.02,
          log_every=25, device="auto") -> tuple[Policy, dict]:
    """Fit the policy. `obs_noise` perturbs observations in normalised units so the
    policy can recover from its own drift instead of stalling once a rollout leaves
    the demonstrated distribution."""
    torch.manual_seed(seed)
    obs, act = build_arrays(episodes, chunk)
    obs_mean, obs_std = obs.mean(0), obs.std(0) + 1e-6
    act_mean, act_std = act.mean(0), act.std(0) + 1e-6
    resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if device == "auto" else str(device)
    X = torch.from_numpy((obs - obs_mean) / obs_std).to(resolved_device)
    Y = torch.from_numpy((act - act_mean) / act_std).to(resolved_device)

    n_val = max(1, int(len(X) * val_frac))
    perm = torch.randperm(len(X))
    tr, va = perm[n_val:], perm[:n_val]

    model = MLP(X.shape[1], Y.shape[1], hidden, depth).to(resolved_device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    history = {"train": [], "val": [], "device": resolved_device}

    for ep in range(epochs):
        model.train()
        idx = tr[torch.randperm(len(tr))]
        tot = 0.0
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            xb = X[b]
            if obs_noise:
                xb = xb + torch.randn_like(xb) * obs_noise
            loss = nn.functional.mse_loss(model(xb), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(model(X[va]), Y[va]).item()
        history["train"].append(tot / len(tr)); history["val"].append(vl)
        if log_every and (ep + 1) % log_every == 0:
            print(f"      epoch {ep+1:4d}/{epochs}  train {tot/len(tr):.5f}  val {vl:.5f}")

    horizon = int(np.median([len(episode) for episode in episodes]))
    return Policy(model.eval(), obs_mean, obs_std, act_mean, act_std, chunk,
                  "absolute_joint", horizon), history
