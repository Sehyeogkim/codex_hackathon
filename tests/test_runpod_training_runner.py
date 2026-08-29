from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
import urllib.error
from pathlib import Path

from src import runpod_training_runner as runner


def _write_request(directory: Path, **overrides: object) -> Path:
    for index in range(3):
        (directory / f"seed_{index}.json").write_text("{}", encoding="utf-8")
    payload: dict[str, object] = {
        "schema_version": 1,
        "job_name": "test-training",
        "seed_trajectories": [f"seed_{index}.json" for index in range(3)],
        "episodes": 500,
        "epochs": 300,
        "eval_trials": 20,
        "random_seed": 7,
    }
    payload.update(overrides)
    path = directory / "request.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_cloud_request(directory: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "schema_version": 1,
        "job_name": "dexycb-cloud-training",
        "prepare_dexycb_on_pod": True,
        "dexycb_sequence_limit": 3,
        "seed_trajectories": [],
        "episodes": 500,
        "epochs": 300,
        "eval_trials": 20,
        "random_seed": 0,
    }
    payload.update(overrides)
    path = directory / "cloud_request.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_project(directory: Path) -> Path:
    project = directory / "project"
    for relative in runner.RUNTIME_PATHS:
        path = project / relative
        if Path(relative).suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("placeholder", encoding="utf-8")
        else:
            path.mkdir(parents=True, exist_ok=True)
            (path / "placeholder").write_text("x", encoding="utf-8")
    return project


def _result_archive(*, accepted: bool = True, dexycb: bool = False) -> bytes:
    buffer = io.BytesIO()
    results = [True] * (10 if accepted else 9) + [False] * (10 if accepted else 11)
    summary = {
        "cuda_available": True,
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "episodes_validated": 500,
        "eval_trials": 20,
        "eval_results": results,
    }
    entries = {
        "output/summary.json": json.dumps(summary).encode(),
        "output/policy.pt": b"model",
        "output/policy_trial_00.mp4": b"video-0",
        "output/policy_trial_01.mp4": b"video-1",
    }
    if dexycb:
        entries.update(
            {
                "output/dexycb_download.json": b"{}",
                "output/dexycb/dexycb_manifest.json": b"{}",
                "output/dexycb/runpod_dexycb_stage.json": b"{}",
                "output/dexycb/seeds/seed_000.json": b"{}",
                "output/dexycb/seeds/seed_001.json": b"{}",
                "output/dexycb/seeds/seed_002.json": b"{}",
            }
        )
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class _FakeAPI:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.terminated: list[str] = []
        self.polls = 0

    def create_pod(self, payload: dict[str, object]) -> dict[str, object]:
        self.created.append(dict(payload))
        return {"id": "pod_test"}

    def get_pod(self, pod_id: str) -> dict[str, object]:
        self.polls += 1
        if self.polls == 1:
            return {"id": pod_id, "desiredStatus": "RUNNING", "portMappings": {}}
        return {
            "id": pod_id,
            "desiredStatus": "RUNNING",
            "publicIp": "192.0.2.10",
            "portMappings": {"22": 22022},
            "gpu": {"displayName": "NVIDIA GeForce RTX 4090"},
        }

    def terminate_pod(self, pod_id: str) -> None:
        self.terminated.append(pod_id)


class _FakeRemote:
    def __init__(self, archive: bytes, *, fail_run: bool = False) -> None:
        self.archive = archive
        self.fail_run = fail_run
        self.uploads: list[tuple[Path, str, int, str]] = []
        self.commands: list[str] = []

    def upload(self, local: Path, *, host: str, port: int, remote: str) -> None:
        self.uploads.append((local, host, port, remote))

    def run(self, command: str, *, host: str, port: int) -> None:
        self.commands.append(command)
        if self.fail_run:
            raise RuntimeError("training failed")

    def download(self, remote: str, *, host: str, port: int, local: Path) -> None:
        local.write_bytes(self.archive)


class RunPodTrainingRunnerTests(unittest.TestCase):
    def test_rest_client_redacts_token_from_repr_and_errors(self) -> None:
        token = "super-secret-runpod-token"

        def failing_opener(*_: object, **__: object) -> object:
            raise urllib.error.HTTPError(
                "https://rest.runpod.io/v1/pods", 401, "unauthorized", {}, None
            )

        client = runner.RunPodRestClient(token, opener=failing_opener)
        self.assertNotIn(token, repr(client))
        with self.assertRaises(RuntimeError) as raised:
            client.create_pod({"name": "test"})
        self.assertNotIn(token, str(raised.exception))

    def test_request_and_pod_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            request = runner.load_training_request(_write_request(Path(temp_name)))
        self.assertEqual(request.episodes, 500)
        self.assertEqual(request.eval_trials, 20)
        payload = runner.pod_create_payload(request)
        self.assertEqual(payload["cloudType"], "SECURE")
        self.assertFalse(payload["interruptible"])
        self.assertEqual(payload["gpuCount"], 1)
        self.assertEqual(payload["gpuTypeIds"], list(runner.GPU_PREFERENCE))
        self.assertEqual(payload["gpuTypePriority"], "custom")
        self.assertEqual(payload["containerDiskInGb"], 50)
        self.assertEqual(payload["volumeInGb"], 100)
        self.assertEqual(payload["ports"], ["22/tcp"])

    def test_request_rejects_embedded_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            path = _write_request(Path(temp_name), runpod_api_key="do-not-store")
            with self.assertRaisesRegex(ValueError, "credentials are forbidden"):
                runner.load_training_request(path)

    def test_archive_contains_sources_and_provenance_but_no_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            request = runner.load_training_request(_write_request(temp))
            project = _make_project(temp)
            archive = runner.build_training_archive(
                request, temp / "job.tar.gz", project_root=project
            )
            with tarfile.open(archive, "r:gz") as tar:
                names = tar.getnames()
                public = json.load(tar.extractfile("job/training_request.json"))
        self.assertIn("mimic", names)
        self.assertIn("job/seeds/000.json", names)
        self.assertEqual(len(public["seed_trajectories"]), 3)
        self.assertNotIn("api_key", json.dumps(public).lower())

    def test_remote_command_uses_all_seeds_and_fixed_training_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            request = runner.load_training_request(_write_request(Path(temp_name)))
        command = runner.remote_training_command(request)
        self.assertEqual(command.count("--trajectory-json"), 3)
        self.assertIn("job/seeds/000.json", command)
        self.assertIn("job/seeds/001.json", command)
        self.assertIn("job/seeds/002.json", command)
        self.assertIn("--episodes 500", command)
        self.assertIn("--epochs 300", command)
        self.assertIn("--eval-trials 20", command)
        self.assertIn("--seed 7", command)
        self.assertIn("--require-cuda", command)
        self.assertNotIn("API", command)

    def test_dexycb_cloud_command_downloads_prepares_and_trains_all_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            request = runner.load_training_request(
                _write_cloud_request(Path(temp_name))
            )
        command = runner.remote_training_command(request)
        self.assertIn("scripts/download_dexycb.sh", command)
        self.assertIn("tar -xzf /workspace/dexycb/subject-07.tar.gz", command)
        self.assertIn("python -m src.runpod_dexycb_runner", command)
        self.assertIn("mapfile -t seeds", command)
        self.assertIn('trajectory_args+=(--trajectory-json "$seed_path")', command)
        self.assertIn("dexycb_download.json", command)
        self.assertIn("runpod_dexycb_stage.json", command)

    def test_cloud_request_archive_requires_no_local_12gb_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            request = runner.load_training_request(_write_cloud_request(temp))
            archive = runner.build_training_archive(
                request, temp / "cloud.tar.gz", project_root=_make_project(temp)
            )
            with tarfile.open(archive, "r:gz") as tar:
                public = json.load(tar.extractfile("job/training_request.json"))
                names = tar.getnames()
        self.assertTrue(public["prepare_dexycb_on_pod"])
        self.assertEqual(public["seed_trajectories"], [])
        self.assertIn("scripts/download_dexycb.sh", names)
        self.assertFalse(any(name.endswith("subject-07.tar.gz") for name in names))

    def test_dry_run_needs_no_seed_files_or_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            path = _write_request(temp)
            for seed in temp.glob("seed_*.json"):
                seed.unlink()
            plan = runner.dry_run_plan(path)
        self.assertEqual(plan["mode"], "dry-run")
        self.assertEqual(plan["seed_trajectory_count"], 3)
        self.assertEqual(plan["cleanup"], "DELETE /pods/{pod_id} in finally")

    def test_run_downloads_validates_and_always_terminates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            path = _write_request(temp)
            project = _make_project(temp)
            api = _FakeAPI()
            remote = _FakeRemote(_result_archive())
            events: list[dict[str, object]] = []
            result = runner.run_training_on_runpod(
                path,
                temp / "result",
                api=api,
                remote=remote,
                project_root=project,
                poll_seconds=0,
                sleeper=lambda _: None,
                event_callback=events.append,
            )
        self.assertEqual(api.terminated, ["pod_test"])
        self.assertTrue(result["pod_terminated"])
        self.assertTrue(result["assessment"]["accepted"])
        self.assertEqual(events[-1]["event"], "pod.terminated")
        self.assertEqual(api.created[0]["gpuTypePriority"], "custom")
        self.assertEqual(len(remote.uploads), 1)

    def test_remote_failure_still_terminates_pod(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            api = _FakeAPI()
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                runner.run_training_on_runpod(
                    _write_request(temp),
                    temp / "result",
                    api=api,
                    remote=_FakeRemote(_result_archive(), fail_run=True),
                    project_root=_make_project(temp),
                    poll_seconds=0,
                    sleeper=lambda _: None,
                )
        self.assertEqual(api.terminated, ["pod_test"])

    def test_unsuccessful_policy_is_reported_not_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            request = runner.load_training_request(_write_request(temp))
            archive_path = temp / "result.tar.gz"
            archive_path.write_bytes(_result_archive(accepted=False))
            runner._safe_extract(archive_path, temp / "result")
            assessment = runner.assess_training_output(temp / "result", request)
        self.assertFalse(assessment["accepted"])
        self.assertEqual(assessment["eval_successes"], 9)
        self.assertFalse(assessment["gates"]["success_threshold"])

    def test_cloud_assessment_requires_download_manifest_and_three_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            request = runner.load_training_request(_write_cloud_request(temp))
            archive_path = temp / "result.tar.gz"
            archive_path.write_bytes(_result_archive(dexycb=True))
            runner._safe_extract(archive_path, temp / "result")
            assessment = runner.assess_training_output(temp / "result", request)
        self.assertTrue(assessment["accepted"])
        self.assertTrue(assessment["gates"]["dexycb_download_audit"])
        self.assertTrue(assessment["gates"]["dexycb_manifest"])
        self.assertTrue(assessment["gates"]["dexycb_hybrid_seeds"])


if __name__ == "__main__":
    unittest.main()
