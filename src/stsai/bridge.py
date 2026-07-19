"""Localhost client and trajectory recorder for the MIT STS2MCP bridge.

This module has no dependency on the mod's Python MCP wrapper. Keeping the wire
adapter small lets STSAI version and test its learning data independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class BridgeError(RuntimeError):
    """Raised for an unavailable bridge or an invalid bridge response."""


def _validate_local_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "http":
        raise ValueError("The game bridge URL must use http")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The game bridge must be bound to localhost")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("The game bridge URL must not contain credentials, a query, or a fragment")
    return value.rstrip("/")


@dataclass(frozen=True)
class BridgeConfig:
    base_url: str = "http://127.0.0.1:15526"
    timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _validate_local_url(self.base_url))
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


class STS2MCPClient:
    """Minimal JSON client for the STS2MCP localhost REST API."""

    def __init__(self, config: BridgeConfig | None = None):
        self.config = config or BridgeConfig()

    def _request(
        self, method: str, path: str = "/", payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            self.config.base_url + path, data=body, headers=headers, method=method
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read()
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise BridgeError(f"Bridge returned HTTP {exc.code}: {details}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise BridgeError(f"Unable to reach the local STS2 bridge: {exc}") from exc
        try:
            value = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise BridgeError("Bridge returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise BridgeError("Bridge response must be a JSON object")
        return value

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/")

    def get_state(self, mode: str = "singleplayer") -> dict[str, Any]:
        if mode not in {"singleplayer", "multiplayer"}:
            raise ValueError("mode must be singleplayer or multiplayer")
        return self._request("GET", f"/api/v1/{mode}?format=json")

    def get_latest_replay(self) -> dict[str, Any]:
        """Decode the latest native combat replay inside the running game."""

        return self._request("GET", "/api/v1/replay/latest")

    def act(self, action: Mapping[str, Any], mode: str = "singleplayer") -> dict[str, Any]:
        if mode not in {"singleplayer", "multiplayer"}:
            raise ValueError("mode must be singleplayer or multiplayer")
        if not isinstance(action.get("action"), str) or not action["action"]:
            raise ValueError("action payload requires a non-empty 'action' string")
        return self._request("POST", f"/api/v1/{mode}", action)


def state_digest(state: Mapping[str, Any]) -> str:
    canonical = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class TransitionRecorder:
    """Wrap bridge actions with before/after state capture in append-only NDJSON."""

    client: STS2MCPClient
    output: Path
    mode: str = "singleplayer"
    settle_timeout_seconds: float = 3.0
    poll_interval_seconds: float = 0.05
    trajectory_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.mode not in {"singleplayer", "multiplayer"}:
            raise ValueError("mode must be singleplayer or multiplayer")
        if self.settle_timeout_seconds < 0 or self.poll_interval_seconds <= 0:
            raise ValueError("settle timeout must be nonnegative and poll interval positive")
        self.output = Path(self.output)
        self.trajectory_id = self._existing_trajectory_id() or uuid.uuid4().hex

    def _existing_trajectory_id(self) -> str | None:
        try:
            with self.output.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    value = event.get("trajectory_id") if isinstance(event, dict) else None
                    return str(value) if value else None
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return None

    def _settled_state(self, previous_digest: str) -> tuple[dict[str, Any], bool]:
        deadline = time.monotonic() + self.settle_timeout_seconds
        latest = self.client.get_state(self.mode)
        while state_digest(latest) == previous_digest and time.monotonic() < deadline:
            time.sleep(self.poll_interval_seconds)
            latest = self.client.get_state(self.mode)
        return latest, state_digest(latest) != previous_digest

    def act(self, action: Mapping[str, Any]) -> dict[str, Any]:
        before = self.client.get_state(self.mode)
        before_digest = state_digest(before)
        response = self.client.act(action, self.mode)
        after, changed = self._settled_state(before_digest)
        event = {
            "schema_version": 1,
            "event_type": "bridge_transition",
            "trajectory_id": self.trajectory_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "bridge": "STS2MCP",
            "mode": self.mode,
            "observation": before,
            "action": dict(action),
            "bridge_response": response,
            "next_observation": after,
            "state_changed": changed,
            "terminal": after.get("state_type") == "game_over",
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n"
        with self.output.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return event
