"""Run the Franka behaviour-cloning job on an ephemeral RunPod GPU Pod.

The command is intentionally safe by default: without ``--execute`` it only
prints a redacted deployment plan.  Credentials are read from
``RUNPOD_API_KEY`` and are never serialized into archives, manifests, commands,
or event payloads.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import math
import os
import shlex
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUEST_PATH = Path(__file__).resolve().parent / "config/training_request.json"
API_BASE_URL = "https://rest.runpod.io/v1"
REMOTE_ROOT = "/workspace/robot-training"
REMOTE_OUTPUT = "/workspace/output"
GPU_PREFERENCE = (
    "NVIDIA GeForce RTX 4090",
    "NVIDIA A40",
    "NVIDIA L4",
)
RUNTIME_PATHS = (
    "runpod_workflow_train",
    "dataminer",
    "data/resources/franka_emika_panda",
    "data/resources/hand_landmarker.task",
)


@dataclasses.dataclass(frozen=True)
class TrainingRequest:
    """Validated, versioned input contract for one training job."""

    schema_version: int
    job_name: str
    seed_trajectories: tuple[Path, ...]
    seed_trajectory_sources: tuple[str, ...]
    prepare_dexycb_on_pod: bool
    dexycb_sequence_limit: int
    episodes: int
    epochs: int
    eval_trials: int
    random_seed: int
    image_name: str


def _require_int(data: Mapping[str, Any], key: str, *, minimum: int) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def load_training_request(
    request_path: str | Path, *, require_seed_files: bool = True
) -> TrainingRequest:
    """Load and validate a request without ever accepting embedded secrets."""

    request_path = Path(request_path).resolve()
    data = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("training request must be a JSON object")

    forbidden = {
        key
        for key in data
        if any(part in key.lower() for part in ("api_key", "token", "secret", "password"))
    }
    if forbidden:
        raise ValueError(
            "credentials are forbidden in training requests; use environment variables"
        )
    if data.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")

    name = data.get("job_name")
    if not isinstance(name, str) or not name.strip() or len(name) > 80:
        raise ValueError("job_name must be a non-empty string of at most 80 characters")
    allowed_name = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    if any(character not in allowed_name for character in name):
        raise ValueError("job_name may contain only letters, numbers, '-' and '_'")

    prepare_dexycb = data.get("prepare_dexycb_on_pod", False)
    if not isinstance(prepare_dexycb, bool):
        raise ValueError("prepare_dexycb_on_pod must be a boolean")
    raw_trajectories = data.get("seed_trajectories", [])
    if not isinstance(raw_trajectories, list):
        raise ValueError("seed_trajectories must be a list")
    if not all(isinstance(item, str) and item.strip() for item in raw_trajectories):
        raise ValueError("every seed trajectory must be a non-empty path string")
    if not prepare_dexycb and not raw_trajectories:
        raise ValueError(
            "seed_trajectories must be non-empty unless prepare_dexycb_on_pod is true"
        )
    seed_paths = tuple((request_path.parent / item).resolve() for item in raw_trajectories)
    if require_seed_files:
        missing = [str(path) for path in seed_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"seed trajectories not found: {', '.join(missing)}")

    image_name = data.get(
        "image_name",
        "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    )
    if not isinstance(image_name, str) or not image_name.strip():
        raise ValueError("image_name must be a non-empty string")

    sequence_limit = (
        _require_int(data, "dexycb_sequence_limit", minimum=1)
        if prepare_dexycb
        else 0
    )
    if prepare_dexycb and sequence_limit != 2:
        raise ValueError("dexycb_sequence_limit must be 2 for the verified subject-07 set")

    return TrainingRequest(
        schema_version=1,
        job_name=name,
        seed_trajectories=seed_paths,
        seed_trajectory_sources=tuple(raw_trajectories),
        prepare_dexycb_on_pod=prepare_dexycb,
        dexycb_sequence_limit=sequence_limit,
        episodes=_require_int(data, "episodes", minimum=1),
        epochs=_require_int(data, "epochs", minimum=1),
        eval_trials=_require_int(data, "eval_trials", minimum=1),
        random_seed=_require_int(data, "random_seed", minimum=0),
        image_name=image_name,
    )


def pod_create_payload(request: TrainingRequest) -> dict[str, Any]:
    """Return the fixed-cost, Secure On-Demand Pod specification."""

    return {
        "name": request.job_name,
        "cloudType": "SECURE",
        "computeType": "GPU",
        "interruptible": False,
        "gpuCount": 1,
        "gpuTypeIds": list(GPU_PREFERENCE),
        "gpuTypePriority": "custom",
        "containerDiskInGb": 50,
        "volumeInGb": 100,
        "volumeMountPath": "/workspace",
        "imageName": request.image_name,
        "ports": ["22/tcp"],
        "supportPublicIp": True,
    }


def build_training_archive(
    request: TrainingRequest,
    archive_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Package runtime code and seed trajectories without credentials."""

    project_root = Path(project_root).resolve()
    archive_path = Path(archive_path).resolve()
    missing = [path for path in RUNTIME_PATHS if not (project_root / path).exists()]
    if missing:
        raise FileNotFoundError(f"missing runtime paths: {', '.join(missing)}")
    missing_seeds = [str(path) for path in request.seed_trajectories if not path.is_file()]
    if missing_seeds:
        raise FileNotFoundError(f"seed trajectories not found: {', '.join(missing_seeds)}")

    public_request = {
        "schema_version": request.schema_version,
        "job_name": request.job_name,
        "seed_trajectories": [
            f"job/seeds/{index:03d}.json"
            for index in range(len(request.seed_trajectories))
        ],
        "seed_trajectory_sources": list(request.seed_trajectory_sources),
        "prepare_dexycb_on_pod": request.prepare_dexycb_on_pod,
        "dexycb_sequence_limit": request.dexycb_sequence_limit,
        "episodes": request.episodes,
        "epochs": request.epochs,
        "eval_trials": request.eval_trials,
        "random_seed": request.random_seed,
    }
    request_bytes = json.dumps(public_request, indent=2).encode("utf-8")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for relative in RUNTIME_PATHS:
            archive.add(
                project_root / relative,
                arcname=relative,
                recursive=True,
                filter=_archive_filter,
            )
        info = tarfile.TarInfo("job/training_request.json")
        info.size = len(request_bytes)
        info.mode = 0o600
        archive.addfile(info, fileobj=io.BytesIO(request_bytes))
        for index, seed_path in enumerate(request.seed_trajectories):
            archive.add(seed_path, arcname=f"job/seeds/{index:03d}.json")
    return archive_path


def _archive_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Exclude caches and local environment files from the remote payload."""

    parts = Path(info.name).parts
    if (
        "__pycache__" in parts
        or info.name.endswith((".pyc", ".pyo"))
        or any(part == ".env" or part.startswith(".env.") for part in parts)
    ):
        return None
    return info


def remote_training_command(request: TrainingRequest) -> str:
    """Build the deterministic command executed over SSH."""

    if request.prepare_dexycb_on_pod:
        return remote_dexycb_training_command(request)

    args = [
        "python",
        "-m",
        "runpod_workflow_train.train_policy",
    ]
    for index in range(len(request.seed_trajectories)):
        args.extend(
            ["--trajectory-json", f"{REMOTE_ROOT}/job/seeds/{index:03d}.json"]
        )
    args.extend(
        [
        "--episodes",
        str(request.episodes),
        "--epochs",
        str(request.epochs),
        "--eval-trials",
        str(request.eval_trials),
        "--seed",
        str(request.random_seed),
        "--out",
        REMOTE_OUTPUT,
        ]
    )
    train = " ".join(shlex.quote(part) for part in args)
    root = shlex.quote(REMOTE_ROOT)
    output = shlex.quote(REMOTE_OUTPUT)
    body = (
        "set -euo pipefail; "
        f"mkdir -p {root} {output}; "
        "trap 'tar -czf /tmp/robot-training-output.tar.gz -C /workspace output >/dev/null 2>&1 || true' EXIT; "
        f"tar --no-same-owner -xzf /tmp/robot-training.tar.gz -C {root}; "
        f"cd {root}; "
        "apt-get update -qq; "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
        "libegl1 libgles2 libgl1 >/dev/null; "
        f"python -m pip install -r runpod_workflow_train/requirements.txt 2>&1 | tee {output}/dependencies.log; "
        "python -c 'import torch; assert torch.cuda.is_available(), \"CUDA unavailable\"'; "
        f"{train} --require-cuda 2>&1 | tee {output}/training.log; "
        "tar -czf /tmp/robot-training-output.tar.gz -C /workspace output; trap - EXIT"
    )
    return f"bash -lc {shlex.quote(body)}"


def remote_dexycb_training_command(request: TrainingRequest) -> str:
    """Download, prepare two verified DexYCB seeds, and train in one Pod/workspace."""

    root = shlex.quote(REMOTE_ROOT)
    output = shlex.quote(REMOTE_OUTPUT)
    dataset = "/workspace/dexycb"
    extracted = f"{dataset}/extracted"
    prepared = "/workspace/dexycb-prepared"
    train_args = [
        "--episodes",
        str(request.episodes),
        "--epochs",
        str(request.epochs),
        "--eval-trials",
        str(request.eval_trials),
        "--seed",
        str(request.random_seed),
        "--out",
        REMOTE_OUTPUT,
        "--require-cuda",
    ]
    fixed_train_args = " ".join(shlex.quote(part) for part in train_args)
    body = (
        "set -euo pipefail; "
        f"mkdir -p {root} {output} {dataset} {extracted} {prepared}; "
        "trap 'tar -czf /tmp/robot-training-output.tar.gz -C /workspace output >/dev/null 2>&1 || true' EXIT; "
        f"tar --no-same-owner -xzf /tmp/robot-training.tar.gz -C {root}; "
        f"cd {root}; "
        "apt-get update -qq; "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
        "libegl1 libgles2 libgl1 >/dev/null; "
        f"python -m pip install -r runpod_workflow_train/requirements.txt 2>&1 | tee {output}/dependencies.log; "
        "python -c 'import torch; assert torch.cuda.is_available(), \"CUDA unavailable\"'; "
        f"bash runpod_workflow_train/download_dexycb.sh {dataset} 2>&1 | tee {output}/dexycb_download.log; "
        "python -c 'import hashlib,json,pathlib; "
        f"p=pathlib.Path(\"{dataset}/subject-07.tar.gz\"); "
        "h=hashlib.sha256(); f=p.open(\"rb\"); "
        "all(h.update(chunk) is None for chunk in iter(lambda:f.read(1048576),b\"\")); f.close(); "
        "d={\"dataset\":\"DexYCB\",\"subject\":\"07\",\"license\":\"CC BY-NC 4.0\","
        "\"source\":\"UCBProject/DexYCB Hugging Face mirror\","
        "\"archive_bytes\":p.stat().st_size,\"sha256\":h.hexdigest()}; "
        f"pathlib.Path(\"{REMOTE_OUTPUT}/dexycb_download.json\").write_text(json.dumps(d,indent=2))'; "
        f"tar --no-same-owner -xzf {dataset}/subject-07.tar.gz -C {extracted}; "
        "python -m runpod_workflow_train.dexycb_prepare "
        f"{extracted} --output-dir {prepared} --config runpod_workflow_train/config/demo_config.json "
        f"--limit {request.dexycb_sequence_limit} 2>&1 | tee {output}/dexycb_prepare.log; "
        f"mkdir -p {output}/dexycb/seeds; "
        f"cp {prepared}/dexycb_manifest.json {output}/dexycb/; "
        f"cp {prepared}/runpod_dexycb_stage.json {output}/dexycb/; "
        f"mapfile -t seeds < <(find {prepared} -name hybrid_trajectory.json -type f | sort); "
        f"[[ ${{#seeds[@]}} -eq {request.dexycb_sequence_limit} ]]; "
        "trajectory_args=(); seed_index=0; "
        "for seed_path in \"${seeds[@]}\"; do "
        "trajectory_args+=(--trajectory-json \"$seed_path\"); "
        f"printf -v seed_name 'seed_%03d.json' \"$seed_index\"; cp \"$seed_path\" {output}/dexycb/seeds/\"$seed_name\"; "
        "seed_index=$((seed_index+1)); done; "
        "python -m runpod_workflow_train.train_policy \"${trajectory_args[@]}\" "
        f"{fixed_train_args} 2>&1 | tee {output}/training.log; "
        "tar -czf /tmp/robot-training-output.tar.gz -C /workspace output; trap - EXIT"
    )
    return f"bash -lc {shlex.quote(body)}"


class RunPodAPI(Protocol):
    def create_pod(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def get_pod(self, pod_id: str) -> Mapping[str, Any]: ...

    def terminate_pod(self, pod_id: str) -> None: ...


class RunPodRestClient:
    """Small REST client that never includes its bearer token in errors or reprs."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = API_BASE_URL,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        if not api_key:
            raise ValueError("RunPod API key cannot be empty")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._opener = opener

    def __repr__(self) -> str:
        return f"RunPodRestClient(base_url={self.base_url!r}, api_key=<redacted>)"

    def _request(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with self._opener(request, timeout=60) as response:
                content = response.read()
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"RunPod API {method} {path} failed with HTTP {error.code}"
            ) from None
        except urllib.error.URLError as error:
            raise RuntimeError(f"RunPod API {method} {path} was unreachable") from error
        if not content:
            return {}
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"RunPod API {method} {path} returned invalid JSON")
        return parsed

    def create_pod(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._request("POST", "/pods", payload)

    def get_pod(self, pod_id: str) -> Mapping[str, Any]:
        return self._request("GET", f"/pods/{pod_id}")

    def terminate_pod(self, pod_id: str) -> None:
        self._request("DELETE", f"/pods/{pod_id}")


class RemoteExecutor(Protocol):
    def upload(self, local: Path, *, host: str, port: int, remote: str) -> None: ...

    def run(self, command: str, *, host: str, port: int) -> None: ...

    def download(self, remote: str, *, host: str, port: int, local: Path) -> None: ...


class SSHExecutor:
    """Non-interactive SSH/SCP transport using the caller's RunPod SSH key."""

    def __init__(
        self,
        *,
        private_key: str | Path | None = None,
        user: str = "root",
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.private_key = Path(private_key).expanduser() if private_key else None
        self.user = user
        self._run_command = command_runner

    def _options(self, port: int) -> list[str]:
        options = [
            "-P",
            str(port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=30",
        ]
        if self.private_key:
            options.extend(["-i", str(self.private_key)])
        return options

    def _execute(self, command: list[str]) -> None:
        # Remote training can run for hours and emit large download/progress logs.
        # Keep stdout on the remote artifact logs instead of buffering it locally,
        # but retain a bounded stderr tail so transport/shell failures are actionable.
        completed = self._run_command(
            command,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            detail = f": {stderr[-4000:]}" if stderr else ""
            raise RuntimeError(
                f"remote transport failed with exit code {completed.returncode}{detail}"
            )

    def upload(self, local: Path, *, host: str, port: int, remote: str) -> None:
        self._execute(
            ["scp", *self._options(port), str(local), f"{self.user}@{host}:{remote}"]
        )

    def run(self, command: str, *, host: str, port: int) -> None:
        options = self._options(port)
        options[0] = "-p"
        self._execute(["ssh", *options, f"{self.user}@{host}", command])

    def download(self, remote: str, *, host: str, port: int, local: Path) -> None:
        self._execute(
            ["scp", *self._options(port), f"{self.user}@{host}:{remote}", str(local)]
        )


def wait_for_ssh(
    api: RunPodAPI,
    pod_id: str,
    *,
    timeout_seconds: float = 900,
    poll_seconds: float = 5,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[str, int, Mapping[str, Any]]:
    """Poll until the Pod is running and its public SSH mapping is available."""

    deadline = clock() + timeout_seconds
    while clock() < deadline:
        pod = api.get_pod(pod_id)
        status = str(pod.get("desiredStatus", ""))
        if status in {"EXITED", "TERMINATED"}:
            raise RuntimeError(f"RunPod Pod entered terminal status {status} before SSH")
        public_ip = pod.get("publicIp")
        mappings = pod.get("portMappings") or {}
        port = mappings.get("22") or mappings.get("22/tcp")
        if status == "RUNNING" and isinstance(public_ip, str) and public_ip and port:
            return public_ip, int(port), pod
        sleeper(poll_seconds)
    raise TimeoutError(f"RunPod Pod {pod_id} did not expose SSH before timeout")


def _validate_member(member: tarfile.TarInfo, destination: Path) -> None:
    if member.issym() or member.islnk():
        raise ValueError(f"archive links are not allowed: {member.name}")
    target = (destination / member.name).resolve()
    if destination.resolve() not in (target, *target.parents):
        raise ValueError(f"unsafe archive member: {member.name}")


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            _validate_member(member, destination)
        # Every member was validated above. Avoid the Python 3.12-only
        # ``filter=`` argument so the submission CLI also runs on Python 3.9.
        archive.extractall(destination)


def assess_training_output(
    output_dir: str | Path, request: TrainingRequest
) -> dict[str, Any]:
    """Evaluate output against the demo's explicit acceptance gates."""

    output_dir = Path(output_dir).resolve()
    summary_path = output_dir / "output" / "summary.json"
    policy_path = output_dir / "output" / "policy.pt"
    if not summary_path.is_file():
        raise RuntimeError("RunPod job did not return output/summary.json")
    if not policy_path.is_file():
        raise RuntimeError("RunPod job did not return output/policy.pt")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise RuntimeError("RunPod summary.json must contain an object")

    eval_trials = int(summary.get("eval_trials", 0))
    eval_results = summary.get("eval_results", [])
    if isinstance(eval_results, list):
        successes = sum(result is True or result == 1 for result in eval_results)
    else:
        successes = 0
    videos = sorted((output_dir / "output").glob("*.mp4"))
    gates = {
        "cuda_available": summary.get("cuda_available") is True,
        "gpu_name_recorded": bool(summary.get("gpu_name")),
        "episodes_validated": int(summary.get("episodes_validated", 0))
        >= request.episodes,
        "held_out_trials": eval_trials >= request.eval_trials,
        "success_threshold": successes >= math.ceil(request.eval_trials / 2),
        "rollout_videos": len(videos) >= 2,
    }
    if request.prepare_dexycb_on_pod:
        dexycb_root = output_dir / "output" / "dexycb"
        hybrid_seeds = sorted((dexycb_root / "seeds").glob("seed_*.json"))
        gates.update(
            {
                "dexycb_download_audit": (
                    output_dir / "output" / "dexycb_download.json"
                ).is_file(),
                "dexycb_manifest": (dexycb_root / "dexycb_manifest.json").is_file(),
                "dexycb_hybrid_seeds": len(hybrid_seeds)
                == request.dexycb_sequence_limit,
            }
        )
    return {
        "accepted": all(gates.values()),
        "gates": gates,
        "eval_successes": successes,
        "eval_trials": eval_trials,
        "summary": str(summary_path),
        "policy": str(policy_path),
        "rollout_videos": [str(path) for path in videos],
        "job_summary": summary,
    }


def run_training_on_runpod(
    request_path: str | Path,
    output_dir: str | Path,
    *,
    api_key: str | None = None,
    api: RunPodAPI | None = None,
    remote: RemoteExecutor | None = None,
    private_key: str | Path | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    timeout_seconds: float = 900,
    poll_seconds: float = 5,
    project_root: str | Path = PROJECT_ROOT,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Create, use, and always terminate one RunPod training Pod."""

    request = load_training_request(request_path)
    if api is None:
        token = api_key or os.environ.get("RUNPOD_API_KEY")
        if not token:
            raise RuntimeError("RUNPOD_API_KEY is required for an executed run")
        api = RunPodRestClient(token)
    remote = remote or SSHExecutor(private_key=private_key)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def emit(event: str, **details: Any) -> None:
        payload = {"event": event, **details}
        if event_callback:
            event_callback(payload)
        else:
            print(json.dumps(payload, ensure_ascii=False), flush=True)

    pod_id: str | None = None
    terminated = False
    with tempfile.TemporaryDirectory(prefix="franka-runpod-") as temp_name:
        temp_dir = Path(temp_name)
        request_archive = build_training_archive(
            request, temp_dir / "request.tar.gz", project_root=project_root
        )
        try:
            emit("pod.creating", gpu_preference=list(GPU_PREFERENCE))
            pod = api.create_pod(pod_create_payload(request))
            value = pod.get("id")
            if not isinstance(value, str) or not value:
                raise RuntimeError("RunPod create response did not contain a Pod ID")
            pod_id = value
            emit("pod.created", pod_id=pod_id)
            host, port, ready_pod = wait_for_ssh(
                api,
                pod_id,
                timeout_seconds=timeout_seconds,
                poll_seconds=poll_seconds,
                clock=clock,
                sleeper=sleeper,
            )
            emit("pod.ready", pod_id=pod_id)

            remote.upload(
                request_archive,
                host=host,
                port=port,
                remote="/tmp/robot-training.tar.gz",
            )
            emit(
                "input.uploaded",
                seed_count=(
                    request.dexycb_sequence_limit
                    if request.prepare_dexycb_on_pod
                    else len(request.seed_trajectories)
                ),
                seed_mode=(
                    "dexycb_remote_prepare"
                    if request.prepare_dexycb_on_pod
                    else "local_upload"
                ),
            )
            try:
                remote.run(remote_training_command(request), host=host, port=port)
            except Exception:
                partial_archive = temp_dir / "partial-response.tar.gz"
                try:
                    remote.download(
                        "/tmp/robot-training-output.tar.gz",
                        host=host,
                        port=port,
                        local=partial_archive,
                    )
                    _safe_extract(partial_archive, output_dir)
                    emit("output.partial_downloaded", output_dir=str(output_dir))
                except Exception:
                    emit("output.partial_download_failed")
                raise
            emit("training.completed")

            response_archive = temp_dir / "response.tar.gz"
            remote.download(
                "/tmp/robot-training-output.tar.gz",
                host=host,
                port=port,
                local=response_archive,
            )
            _safe_extract(response_archive, output_dir)
            assessment = assess_training_output(output_dir, request)
            gpu = ready_pod.get("gpu") or {}
            result = {
                "mode": "runpod",
                "pod_id": pod_id,
                "gpu_requested": list(GPU_PREFERENCE),
                "gpu_allocated": gpu.get("displayName") if isinstance(gpu, dict) else None,
                "data_provenance": (
                    {
                        "dataset": "DexYCB",
                        "subject": "07",
                        "license": "CC BY-NC 4.0",
                        "prepared_on_runpod": True,
                    }
                    if request.prepare_dexycb_on_pod
                    else list(request.seed_trajectory_sources)
                ),
                "output_dir": str(output_dir / "output"),
                "assessment": assessment,
            }
        finally:
            if pod_id is not None:
                emit("pod.terminating", pod_id=pod_id)
                try:
                    api.terminate_pod(pod_id)
                except Exception as error:
                    emit("pod.termination_failed", pod_id=pod_id)
                    raise RuntimeError(
                        f"CRITICAL: RunPod Pod {pod_id} could not be terminated"
                    ) from error
                terminated = True
                emit("pod.terminated", pod_id=pod_id)

    result["pod_terminated"] = terminated
    manifest_path = output_dir / "runpod_job_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["manifest"] = str(manifest_path)
    return result


def dry_run_plan(request_path: str | Path) -> dict[str, Any]:
    """Return a non-mutating, credential-free preview of an actual run."""

    request = load_training_request(request_path, require_seed_files=False)
    return {
        "mode": "dry-run",
        "pod": pod_create_payload(request),
        "seed_trajectory_count": (
            request.dexycb_sequence_limit
            if request.prepare_dexycb_on_pod
            else len(request.seed_trajectories)
        ),
        "seed_trajectories": list(request.seed_trajectory_sources),
        "seed_source": (
            "DexYCB subject-07, prepared remotely"
            if request.prepare_dexycb_on_pod
            else "local uploaded trajectories"
        ),
        "remote_command": remote_training_command(request),
        "cleanup": "DELETE /pods/{pod_id} in finally",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Franka behaviour-cloning training on an ephemeral RunPod Pod."
    )
    parser.add_argument(
        "request",
        type=Path,
        nargs="?",
        default=DEFAULT_REQUEST_PATH,
        help="Training request JSON (defaults to the packaged demo request)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runpod"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create a billable Pod; without this flag the command is a dry-run",
    )
    parser.add_argument("--ssh-key", type=Path, help="Private SSH key registered with RunPod")
    parser.add_argument("--timeout", type=float, default=900)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print(json.dumps(dry_run_plan(args.request), indent=2), flush=True)
        return 0
    result = run_training_on_runpod(
        args.request,
        args.output_dir,
        private_key=args.ssh_key,
        timeout_seconds=args.timeout,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["assessment"]["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
