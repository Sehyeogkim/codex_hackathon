from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from persona import reflex_api


def _manifest(directory: Path, **overrides: object) -> Path:
    prompt = directory / "prompt.md"
    prompt.write_text("Run deterministic checks and report artifacts only.", encoding="utf-8")
    payload: dict[str, object] = {
        "schema_version": 1,
        "base_url": "https://reflex.runloop.ai/api",
        "organization_id": "test-org",
        "repo_slug": "Sehyeogkim/codex_hackathon",
        "repo_branch": "codex/demo-runloop-reflex",
        "agent_type": "opencode",
        "prompt_mode": "implement",
        "sandbox": {"resource_size": "SMALL", "idle_time_minutes": 15},
        "agents": [
            {
                "name": f"role-{index}",
                "prompt_file": "prompt.md",
                "role_context": f"Focus on stage {index}.",
            }
            for index in range(4)
        ],
    }
    payload.update(overrides)
    path = directory / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _Response:
    def __init__(self, payload: object) -> None:
        self.content = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.content


class _Opener:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, **_: object) -> _Response:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _Response(response)


class _LaunchClient:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.created: list[dict[str, object]] = []
        self.stopped: list[str] = []

    def create_agent(self, payload):
        if self.fail_at is not None and len(self.created) == self.fail_at:
            raise RuntimeError("provider failed with hidden details")
        self.created.append(dict(payload))
        index = len(self.created)
        return {
            "id": f"agent_{index}",
            "name": payload["name"],
            "agentType": payload["agentType"],
            "status": "starting",
            "devboxId": None,
        }

    def stop_agent(self, agent_id: str):
        self.stopped.append(agent_id)
        return {"id": agent_id, "status": "stopped"}


class ReflexAPITests(unittest.TestCase):
    def test_repository_manifest_has_four_small_idle_roles(self) -> None:
        path = Path(__file__).resolve().parents[1] / "persona" / "reflex_agents.json"
        manifest = reflex_api.load_manifest(path)
        self.assertEqual(manifest.organization_id, "sehyeog-workspace-1")
        self.assertEqual(manifest.repo_slug, "Sehyeogkim/codex_hackathon")
        self.assertEqual(manifest.repo_branch, "codex/demo-runloop-reflex")
        self.assertEqual(manifest.resource_size, "SMALL")
        self.assertEqual(manifest.idle_time_minutes, 15)
        self.assertEqual(len(manifest.agents), 4)
        self.assertEqual(len({agent.name for agent in manifest.agents}), 4)
        self.assertEqual(
            [agent.name for agent in manifest.agents],
            [
                "robot-data-reconstruction",
                "robot-data-retargeting",
                "robot-data-physical-validation",
                "robot-data-scaling",
            ],
        )
        self.assertEqual(
            [agent.prompt_file.name for agent in manifest.agents],
            [
                "reconstruction.prompt.md",
                "retargeting.prompt.md",
                "physical_validation.prompt.md",
                "data_scaling.prompt.md",
            ],
        )
        manifest_text = path.read_text(encoding="utf-8")
        self.assertIn("subject-07", manifest_text)
        self.assertNotIn("subject-01", manifest_text)
        self.assertIn("requests 3 matching sequences", manifest_text)
        self.assertIn("currently verifies 2", manifest_text)

        role_manifest_path = (
            Path(__file__).resolve().parents[1] / "persona" / "manifest.json"
        )
        role_manifest = json.loads(role_manifest_path.read_text(encoding="utf-8"))
        dataset = role_manifest["demo_dataset"]
        self.assertEqual(dataset["subject"], "subject-07")
        self.assertEqual(dataset["requested_sequence_count"], 3)
        self.assertEqual(dataset["available_matching_sequence_count"], 2)
        self.assertEqual(dataset["selected_sequence_count"], 2)
        self.assertEqual(dataset["verified_sequence_count"], 2)
        self.assertNotIn("sequence_count", dataset)

    def test_default_cli_manifest_is_inside_persona_package(self) -> None:
        args = reflex_api.build_parser().parse_args([])
        self.assertEqual(args.manifest, reflex_api.DEFAULT_MANIFEST_PATH)
        self.assertEqual(args.manifest.parent.name, "persona")
        self.assertTrue(args.manifest.is_file())

    def test_default_plan_validates_without_api_key_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            manifest = reflex_api.load_manifest(_manifest(Path(temp_name)))
            plan = reflex_api.dry_run_plan(manifest)
        self.assertTrue(plan["valid"])
        self.assertEqual(plan["mode"], "dry-run")
        self.assertEqual(plan["agentCount"], 4)
        self.assertFalse(plan["provisionsDevboxes"])
        self.assertNotIn(
            "Run deterministic checks and report artifacts only.", json.dumps(plan)
        )
        self.assertNotIn("rfx_", json.dumps(plan))

    def test_manifest_rejects_credentials_and_non_small_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            with self.assertRaisesRegex(ValueError, "credentials are forbidden"):
                reflex_api.load_manifest(_manifest(temp, api_key="rfx_should_not_exist"))
            with self.assertRaisesRegex(ValueError, "resource_size=SMALL"):
                reflex_api.load_manifest(
                    _manifest(
                        temp,
                        sandbox={"resource_size": "LARGE", "idle_time_minutes": 15},
                    )
                )

    def test_create_list_get_and_stop_use_official_routes_and_headers(self) -> None:
        opener = _Opener(
            [
                {"id": "agent_1", "status": "starting"},
                {"agents": [], "nextCursor": None},
                {"id": "agent_1", "status": "running"},
                {"id": "agent_1", "status": "stopped"},
            ]
        )
        client = reflex_api.ReflexClient("rfx-private-value", "test-org", opener=opener)
        client.create_agent({"agentType": "opencode", "prompt": "work"})
        client.list_agents(limit=50)
        client.get_agent("agent_1")
        client.stop_agent("agent_1")

        self.assertEqual(
            [(request.method, request.full_url) for request in opener.requests],
            [
                ("POST", "https://reflex.runloop.ai/api/agents"),
                ("GET", "https://reflex.runloop.ai/api/agents?limit=50"),
                ("GET", "https://reflex.runloop.ai/api/agents/agent_1"),
                ("POST", "https://reflex.runloop.ai/api/agents/agent_1/stop"),
            ],
        )
        headers = {key.casefold(): value for key, value in opener.requests[0].header_items()}
        self.assertEqual(headers["authorization"], "Bearer rfx-private-value")
        self.assertEqual(headers["x-organization-id"], "test-org")

    def test_client_repr_and_http_errors_redact_api_key(self) -> None:
        token = "rfx_super_secret_private_token"

        def fail(*_: object, **__: object) -> object:
            raise urllib.error.HTTPError(
                "https://reflex.runloop.ai/api/agents",
                401,
                "unauthorized",
                {},
                io.BytesIO(token.encode()),
            )

        client = reflex_api.ReflexClient(token, "test-org", opener=fail)
        self.assertNotIn(token, repr(client))
        with self.assertRaises(RuntimeError) as raised:
            client.list_agents()
        self.assertNotIn(token, str(raised.exception))

    def test_launch_builds_four_required_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            manifest = reflex_api.load_manifest(_manifest(Path(temp_name)))
            client = _LaunchClient()
            result = reflex_api.launch_agents(manifest, client)
        self.assertEqual(len(result), 4)
        self.assertEqual(len(client.created), 4)
        for payload in client.created:
            self.assertEqual(payload["agentType"], "opencode")
            self.assertTrue(payload["prompt"])
            self.assertEqual(payload["repoSlug"], "Sehyeogkim/codex_hackathon")
            self.assertEqual(payload["repoBranch"], "codex/demo-runloop-reflex")
            self.assertEqual(payload["sandboxOptions"]["resourceSize"], "SMALL")
            self.assertEqual(payload["sandboxOptions"]["idleTimeMinutes"], 15)
        self.assertNotIn("prompt", result[0])

    def test_partial_launch_failure_stops_created_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            manifest = reflex_api.load_manifest(_manifest(Path(temp_name)))
            client = _LaunchClient(fail_at=2)
            with self.assertRaises(reflex_api.ReflexLaunchError) as raised:
                reflex_api.launch_agents(manifest, client)
        self.assertEqual(raised.exception.launched_ids, ("agent_1", "agent_2"))
        self.assertEqual(client.stopped, ["agent_2", "agent_1"])
        self.assertEqual(raised.exception.cleanup_failed_ids, ())

    def test_agent_id_rejects_path_injection(self) -> None:
        client = reflex_api.ReflexClient("private", "test-org", opener=_Opener([]))
        with self.assertRaisesRegex(ValueError, "unsupported characters"):
            client.stop_agent("../other-agent")


if __name__ == "__main__":
    unittest.main()
