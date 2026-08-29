"""Minimal, secret-safe Reflex Agent Sessions REST client.

The CLI is a validation-only dry-run unless an explicit action such as
``--launch`` or ``--stop`` is supplied.  API keys are accepted only through the
``REFLEX_API_KEY`` environment variable and are never included in payloads,
manifests, summaries, or error text.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://reflex.runloop.ai/api"
FORBIDDEN_KEY_PARTS = ("api_key", "token", "secret", "password", "authorization")
SECRET_VALUE_PATTERN = re.compile(
    r"(?:^Bearer\s+\S+|^sk-[A-Za-z0-9_-]{12,}|^rfx_[A-Za-z0-9_-]{12,})"
)


@dataclasses.dataclass(frozen=True)
class AgentSpec:
    name: str
    prompt_file: Path
    role_context: str


@dataclasses.dataclass(frozen=True)
class ReflexManifest:
    schema_version: int
    base_url: str
    organization_id: str
    repo_slug: str
    repo_branch: str
    agent_type: str
    prompt_mode: str
    resource_size: str
    idle_time_minutes: int
    agents: tuple[AgentSpec, ...]


def _reject_secrets(value: Any, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold()
            if any(part in lowered for part in FORBIDDEN_KEY_PARTS):
                raise ValueError(f"credentials are forbidden in {path}")
            _reject_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secrets(child, f"{path}[{index}]")
    elif isinstance(value, str) and SECRET_VALUE_PATTERN.search(value.strip()):
        raise ValueError(f"credential-like value is forbidden in {path}")


def _require_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def load_manifest(path: str | Path) -> ReflexManifest:
    """Load the four-role launch manifest and resolve its prompt files."""

    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Reflex manifest must be a JSON object")
    _reject_secrets(data)
    allowed_top = {
        "schema_version",
        "base_url",
        "organization_id",
        "repo_slug",
        "repo_branch",
        "agent_type",
        "prompt_mode",
        "sandbox",
        "agents",
    }
    unknown = sorted(set(data) - allowed_top)
    if unknown:
        raise ValueError(f"unknown manifest fields: {', '.join(unknown)}")
    if data.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")

    base_url = _require_string(data, "base_url").rstrip("/")
    if not base_url.startswith("https://"):
        raise ValueError("base_url must use HTTPS")
    organization_id = _require_string(data, "organization_id")
    repo_slug = _require_string(data, "repo_slug")
    repo_branch = _require_string(data, "repo_branch")
    agent_type = _require_string(data, "agent_type")
    prompt_mode = _require_string(data, "prompt_mode")
    if prompt_mode not in {"implement", "plan", "review"}:
        raise ValueError("prompt_mode must be implement, plan, or review")

    sandbox = data.get("sandbox")
    if not isinstance(sandbox, dict):
        raise ValueError("sandbox must be an object")
    if set(sandbox) != {"resource_size", "idle_time_minutes"}:
        raise ValueError("sandbox requires only resource_size and idle_time_minutes")
    resource_size = sandbox.get("resource_size")
    if resource_size != "SMALL":
        raise ValueError("this demo requires sandbox.resource_size=SMALL")
    idle_time = sandbox.get("idle_time_minutes")
    if isinstance(idle_time, bool) or not isinstance(idle_time, int) or idle_time <= 0:
        raise ValueError("sandbox.idle_time_minutes must be a positive integer")

    raw_agents = data.get("agents")
    if not isinstance(raw_agents, list) or len(raw_agents) != 4:
        raise ValueError("agents must contain exactly four role definitions")
    agents: list[AgentSpec] = []
    for index, raw in enumerate(raw_agents):
        if not isinstance(raw, dict):
            raise ValueError(f"agents[{index}] must be an object")
        if set(raw) != {"name", "prompt_file", "role_context"}:
            raise ValueError(
                f"agents[{index}] requires name, prompt_file, and role_context"
            )
        name = _require_string(raw, "name")
        role = _require_string(raw, "role_context")
        prompt_value = _require_string(raw, "prompt_file")
        prompt_path = (path.parent / prompt_value).resolve()
        if prompt_path.name == ".env" or prompt_path.name.startswith(".env."):
            raise ValueError("environment files cannot be used as agent prompts")
        if not prompt_path.is_file():
            raise FileNotFoundError(f"agent prompt not found: {prompt_path}")
        agents.append(AgentSpec(name, prompt_path, role))
    if len({agent.name for agent in agents}) != 4:
        raise ValueError("all four agent names must be unique")

    return ReflexManifest(
        schema_version=1,
        base_url=base_url,
        organization_id=organization_id,
        repo_slug=repo_slug,
        repo_branch=repo_branch,
        agent_type=agent_type,
        prompt_mode=prompt_mode,
        resource_size=resource_size,
        idle_time_minutes=idle_time,
        agents=tuple(agents),
    )


def _agent_prompt(spec: AgentSpec) -> str:
    base_prompt = spec.prompt_file.read_text(encoding="utf-8").strip()
    if not base_prompt:
        raise ValueError(f"agent prompt is empty: {spec.prompt_file}")
    return (
        f"Role assignment: {spec.name}\n\n"
        f"Your specific focus:\n{spec.role_context}\n\n"
        "Shared operating instructions:\n"
        f"{base_prompt}\n"
    )


def build_agent_payload(manifest: ReflexManifest, spec: AgentSpec) -> dict[str, Any]:
    """Build exactly the public POST /agents fields used by this demo."""

    return {
        "name": spec.name,
        "agentType": manifest.agent_type,
        "prompt": _agent_prompt(spec),
        "promptMode": manifest.prompt_mode,
        "repoSlug": manifest.repo_slug,
        "repoBranch": manifest.repo_branch,
        "sandboxOptions": {
            "resourceSize": manifest.resource_size,
            "idleTimeMinutes": manifest.idle_time_minutes,
        },
    }


def dry_run_plan(manifest: ReflexManifest) -> dict[str, Any]:
    """Return a prompt-fingerprinted plan without provisioning a Devbox."""

    agents = []
    for spec in manifest.agents:
        payload = build_agent_payload(manifest, spec)
        prompt = payload.pop("prompt")
        agents.append(
            {
                **payload,
                "promptFile": str(spec.prompt_file),
                "promptCharacters": len(prompt),
                "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
    return {
        "mode": "dry-run",
        "valid": True,
        "baseUrl": manifest.base_url,
        "organizationId": manifest.organization_id,
        "agentCount": len(agents),
        "agents": agents,
        "provisionsDevboxes": False,
        "launchRequires": "--launch and REFLEX_API_KEY",
    }


class ReflexClient:
    """Small synchronous client for create/list/get/stop Agent operations."""

    def __init__(
        self,
        api_key: str,
        organization_id: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        if not api_key:
            raise ValueError("Reflex API key cannot be empty")
        if not organization_id:
            raise ValueError("Reflex organization id cannot be empty")
        self._api_key = api_key
        self.organization_id = organization_id
        self.base_url = base_url.rstrip("/")
        self._opener = opener

    def __repr__(self) -> str:
        return (
            "ReflexClient("
            f"base_url={self.base_url!r}, organization_id={self.organization_id!r}, "
            "api_key=<redacted>)"
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "x-organization-id": self.organization_id,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with self._opener(request, timeout=60) as response:
                content = response.read()
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"Reflex API {method} {path} failed with HTTP {error.code}"
            ) from None
        except urllib.error.URLError as error:
            raise RuntimeError(f"Reflex API {method} {path} was unreachable") from error
        if not content:
            return {}
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Reflex API {method} {path} returned invalid JSON")
        return parsed

    def create_agent(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = self._request("POST", "/agents", payload)
        if not isinstance(response.get("id"), str) or not response["id"]:
            raise RuntimeError("Reflex create response did not contain an agent id")
        return response

    def list_agents(
        self, *, limit: int = 200, cursor: str | None = None
    ) -> dict[str, Any]:
        if not 1 <= limit <= 200:
            raise ValueError("list limit must be between 1 and 200")
        query: dict[str, Any] = {"limit": limit}
        if cursor:
            query["cursor"] = cursor
        return self._request("GET", "/agents", query=query)

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        return self._request("GET", f"/agents/{_safe_agent_id(agent_id)}")

    def stop_agent(self, agent_id: str) -> dict[str, Any]:
        return self._request("POST", f"/agents/{_safe_agent_id(agent_id)}/stop")


def _safe_agent_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("agent id contains unsupported characters")
    return urllib.parse.quote(value, safe="")


def summarize_agent(agent: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude prompts, tunnel keys, and any provider details from CLI output."""

    return {
        key: agent.get(key)
        for key in ("id", "name", "agentType", "status", "turnState", "devboxId")
        if key in agent
    }


class ReflexLaunchError(RuntimeError):
    def __init__(
        self,
        launched_ids: Sequence[str],
        stopped_ids: Sequence[str],
        cleanup_failed_ids: Sequence[str],
    ) -> None:
        super().__init__(
            "Reflex launch failed; already-created agents were stopped where possible"
        )
        self.launched_ids = tuple(launched_ids)
        self.stopped_ids = tuple(stopped_ids)
        self.cleanup_failed_ids = tuple(cleanup_failed_ids)


def launch_agents(
    manifest: ReflexManifest, client: ReflexClient
) -> list[dict[str, Any]]:
    """Launch four sessions, stopping earlier sessions if a later launch fails."""

    launched: list[dict[str, Any]] = []
    try:
        for spec in manifest.agents:
            launched.append(client.create_agent(build_agent_payload(manifest, spec)))
    except Exception as error:
        stopped: list[str] = []
        cleanup_failed: list[str] = []
        for agent in reversed(launched):
            agent_id = str(agent["id"])
            try:
                client.stop_agent(agent_id)
                stopped.append(agent_id)
            except Exception:
                cleanup_failed.append(agent_id)
        raise ReflexLaunchError(
            [str(agent["id"]) for agent in launched], stopped, cleanup_failed
        ) from error
    return [summarize_agent(agent) for agent in launched]


def _client_from_environment(manifest: ReflexManifest) -> ReflexClient:
    api_key = os.environ.get("REFLEX_API_KEY")
    if not api_key:
        raise RuntimeError("REFLEX_API_KEY is required for API operations")
    return ReflexClient(
        api_key,
        manifest.organization_id,
        base_url=manifest.base_url,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or operate the four Reflex demo Agent Sessions."
    )
    parser.add_argument("manifest", type=Path, nargs="?", default=Path("config/reflex_agents.json"))
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--validate", action="store_true", help="Validate and print a dry-run")
    actions.add_argument("--launch", action="store_true", help="Provision all four sessions")
    actions.add_argument("--list", action="store_true", help="List organization sessions")
    actions.add_argument("--get", metavar="AGENT_ID", help="Get one session")
    actions.add_argument("--stop", metavar="AGENT_ID", help="Stop one session and its Devbox")
    parser.add_argument("--limit", type=int, default=200, help="Maximum sessions for --list")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    if not any((args.launch, args.list, args.get, args.stop)):
        print(json.dumps(dry_run_plan(manifest), indent=2), flush=True)
        return 0

    client = _client_from_environment(manifest)
    if args.launch:
        result: Any = {"launched": launch_agents(manifest, client)}
    elif args.list:
        response = client.list_agents(limit=args.limit)
        agents = response.get("agents", [])
        if not isinstance(agents, list):
            raise RuntimeError("Reflex list response did not contain an agents list")
        result = {
            "agents": [summarize_agent(agent) for agent in agents],
            "nextCursor": response.get("nextCursor"),
        }
    elif args.get:
        result = summarize_agent(client.get_agent(args.get))
    else:
        result = summarize_agent(client.stop_agent(args.stop))
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
