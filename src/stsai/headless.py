"""Process adapter for the pinned sts2-cli native headless runtime."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
from threading import RLock, Thread
from typing import Any, Mapping, Sequence


class HeadlessRuntimeError(RuntimeError):
    """Raised when the native runtime exits or violates its JSON protocol."""


class HeadlessSnapshotError(HeadlessRuntimeError):
    """Raised when an exact reconstructive snapshot cannot be created."""


class HeadlessSnapshotDivergence(HeadlessSnapshotError):
    """Raised when a fresh native process does not reproduce a snapshot prefix."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _json_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HeadlessSnapshotStep:
    """One native mutation and the response required to prove exact replay."""

    request_json: str = field(repr=False)
    response_sha256: str

    def request(self) -> dict[str, Any]:
        value = json.loads(self.request_json)
        if not isinstance(value, dict):
            raise HeadlessSnapshotError("Snapshot request is not a JSON object")
        return value


@dataclass(frozen=True)
class HeadlessSnapshot:
    """Immutable in-memory recipe for an exact, process-isolated native fork."""

    schema_version: int
    protocol_version: str
    runtime_sha256: str
    information_scope: str
    steps: tuple[HeadlessSnapshotStep, ...]
    state_json: str = field(repr=False)
    state_sha256: str
    checkpoint_json: str = field(repr=False)
    checkpoint_sha256: str

    def state(self) -> dict[str, Any]:
        value = json.loads(self.state_json)
        if not isinstance(value, dict):
            raise HeadlessSnapshotError("Snapshot state is not a JSON object")
        return value


def default_runtime_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "build"
        / "sts2-cli"
        / "src"
        / "Sts2Headless"
        / "bin"
        / "Release"
        / "net9.0"
        / "Sts2Headless.dll"
    )


@dataclass(frozen=True)
class HeadlessConfig:
    runtime: Path = field(default_factory=default_runtime_path)
    dotnet: str = "dotnet"
    startup_timeout: float = 15.0
    command_timeout: float = 120.0
    command: tuple[str, ...] | None = None
    environment: Mapping[str, str] = field(default_factory=dict)


class Sts2HeadlessClient:
    """Own one isolated sts2-cli process and exchange one JSON command at a time."""

    def __init__(self, config: HeadlessConfig | None = None):
        self.config = config or HeadlessConfig()
        self._responses: Queue[Mapping[str, Any] | BaseException] = Queue()
        self._logs: deque[str] = deque(maxlen=200)
        self._lock = RLock()
        self._closed = False
        self._snapshot_steps: list[HeadlessSnapshotStep] = []
        self._snapshot_state: dict[str, Any] | None = None
        self._snapshot_taint: str | None = None
        self._fork_replay_mismatches: tuple[int, ...] = ()

        command: Sequence[str]
        if self.config.command is not None:
            command = self.config.command
        else:
            runtime = self.config.runtime.resolve()
            if not runtime.is_file():
                raise FileNotFoundError(
                    f"Headless runtime not found: {runtime}. "
                    "Run scripts/setup-sts2-cli.ps1 first."
                )
            command = (self.config.dotnet, str(runtime))

        env = os.environ.copy()
        env.update(self.config.environment)
        creationflags = 0x08000000 if os.name == "nt" else 0
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=creationflags,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self._process.kill()
            raise HeadlessRuntimeError("Could not open headless runtime pipes")

        self._stdout_thread = Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        ready = self._next_response(self.config.startup_timeout)
        if ready.get("type") != "ready":
            self.close()
            raise HeadlessRuntimeError(f"Expected ready response, received: {ready}")
        self.version = str(ready.get("version", "unknown"))
        self._runtime_sha256 = self._runtime_fingerprint()

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def recent_logs(self) -> tuple[str, ...]:
        return tuple(self._logs)

    def command(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise HeadlessRuntimeError("Headless runtime is closed")
        request = dict(payload)
        if (
            request.get("cmd") == "load_save"
            and "json" not in request
            and request.get("path")
        ):
            save_path = Path(str(request.pop("path"))).resolve()
            request["json"] = save_path.read_text(encoding="utf-8-sig")
        encoded = json.dumps(request, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            if self._process.poll() is not None:
                raise self._exited_error()
            assert self._process.stdin is not None
            try:
                self._process.stdin.write(encoded + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise self._exited_error() from exc
            response = dict(self._next_response(self.config.command_timeout))
            self._track_snapshot_command(request, response)
            return response

    def start_run(
        self,
        *,
        character: str,
        ascension: int,
        seed: str | None = None,
        language: str = "en",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cmd": "start_run",
            "character": character,
            "ascension": ascension,
            "lang": language,
        }
        if seed is not None:
            payload["seed"] = seed
        return self.command(payload)

    def action(self, action: str, **arguments: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"cmd": "action", "action": action}
        if arguments:
            payload["args"] = arguments
        return self.command(payload)

    def load_save(self, path: str | Path, *, language: str = "en") -> dict[str, Any]:
        # Inline the exact bytes in the in-memory transcript. A path-only recipe could
        # silently fork a different state if the save is replaced between branches.
        save_json = Path(path).resolve().read_text(encoding="utf-8-sig")
        return self.command(
            {"cmd": "load_save", "json": save_json, "lang": language}
        )

    @property
    def reconstructed_state(self) -> Mapping[str, Any]:
        if self._snapshot_state is None:
            raise HeadlessSnapshotError("No reconstructive native state is active")
        return json.loads(_canonical_json(self._snapshot_state))

    @property
    def fork_replay_mismatches(self) -> tuple[int, ...]:
        """Transient response steps that differed before the final exact checkpoint."""

        return self._fork_replay_mismatches

    def snapshot(self) -> HeadlessSnapshot:
        """Freeze the current decision as a deterministic native replay recipe."""

        with self._lock:
            if self._closed:
                raise HeadlessSnapshotError("Cannot snapshot a closed native runtime")
            if self._snapshot_taint is not None:
                raise HeadlessSnapshotError(
                    f"Native command history is not snapshot-safe: {self._snapshot_taint}"
                )
            if not self._snapshot_steps or self._snapshot_state is None:
                raise HeadlessSnapshotError("No seeded run or loaded save is available")
            first = self._snapshot_steps[0].request()
            if first.get("cmd") == "start_run" and not first.get("seed"):
                raise HeadlessSnapshotError(
                    "start_run must use an explicit seed before it can be forked"
                )
            checkpoint = self.command({"cmd": "snapshot_checkpoint"})
            if checkpoint.get("type") != "snapshot_checkpoint":
                raise HeadlessSnapshotError(
                    f"Native runtime did not return a snapshot checkpoint: {checkpoint}"
                )
            if checkpoint.get("schema_version") != 1:
                raise HeadlessSnapshotError(
                    "Native runtime returned an unsupported checkpoint schema"
                )
            if checkpoint.get("snapshot_safe") is not True:
                reasons = checkpoint.get("reasons") or ["unknown native reason"]
                raise HeadlessSnapshotError(
                    "Native decision is not safe to fork: " + ", ".join(map(str, reasons))
                )
            state_json = _canonical_json(self._snapshot_state)
            checkpoint_json = _canonical_json(checkpoint)
            return HeadlessSnapshot(
                schema_version=1,
                protocol_version=self.version,
                runtime_sha256=self._runtime_sha256,
                information_scope="native_reconstructive_semantic_v1",
                steps=tuple(self._snapshot_steps),
                state_json=state_json,
                state_sha256=hashlib.sha256(state_json.encode("utf-8")).hexdigest(),
                checkpoint_json=checkpoint_json,
                checkpoint_sha256=hashlib.sha256(
                    checkpoint_json.encode("utf-8")
                ).hexdigest(),
            )

    def fork(self, snapshot: HeadlessSnapshot | None = None) -> Sts2HeadlessClient:
        """Reconstruct a snapshot in a fresh native process and prove every step."""

        frozen = snapshot or self.snapshot()
        if frozen.schema_version != 1:
            raise HeadlessSnapshotError(
                f"Unsupported snapshot schema: {frozen.schema_version}"
            )
        if frozen.protocol_version != self.version:
            raise HeadlessSnapshotError(
                "Snapshot protocol does not match the source native runtime"
            )
        if frozen.runtime_sha256 != self._runtime_sha256:
            raise HeadlessSnapshotError(
                "Snapshot runtime fingerprint does not match the source runtime"
            )
        if frozen.information_scope != "native_reconstructive_semantic_v1":
            raise HeadlessSnapshotError(
                f"Unsupported snapshot information scope: {frozen.information_scope}"
            )

        child = Sts2HeadlessClient(self.config)
        try:
            if child.version != frozen.protocol_version:
                raise HeadlessSnapshotError(
                    "Forked native process reported a different protocol version"
                )
            if child._runtime_sha256 != frozen.runtime_sha256:
                raise HeadlessSnapshotError(
                    "Forked native process has a different runtime fingerprint"
                )
            transient_mismatches: list[int] = []
            for index, step in enumerate(frozen.steps):
                response = child.command(step.request())
                if response.get("type") in {"error", "save_error"}:
                    raise HeadlessSnapshotDivergence(
                        f"Native fork failed transcript step {index}: {response}"
                    )
                actual = _json_digest(response)
                if actual != step.response_sha256:
                    if index == len(frozen.steps) - 1:
                        raise HeadlessSnapshotDivergence(
                            "Native fork diverged at its final transcript step "
                            f"{index}: expected {step.response_sha256}, received {actual}"
                        )
                    transient_mismatches.append(index)
            checkpoint = child.command({"cmd": "snapshot_checkpoint"})
            if checkpoint.get("snapshot_safe") is not True:
                raise HeadlessSnapshotDivergence(
                    "Forked native decision failed its quiescence gate: "
                    + ", ".join(map(str, checkpoint.get("reasons") or []))
                )
            actual_checkpoint_sha = _json_digest(checkpoint)
            if actual_checkpoint_sha != frozen.checkpoint_sha256:
                raise HeadlessSnapshotDivergence(
                    "Forked native semantic checkpoint diverged: "
                    f"expected {frozen.checkpoint_sha256}, "
                    f"received {actual_checkpoint_sha}"
                )
            child._fork_replay_mismatches = tuple(transient_mismatches)
            actual_state = dict(child.reconstructed_state)
            actual_state_sha = _json_digest(actual_state)
            if actual_state_sha != frozen.state_sha256:
                raise HeadlessSnapshotDivergence(
                    "Native fork reproduced individual commands but not the final state: "
                    f"expected {frozen.state_sha256}, received {actual_state_sha}"
                )
            return child
        except BaseException:
            child.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        if self._process.poll() is None:
            try:
                self.command({"cmd": "quit"})
            except (HeadlessRuntimeError, TimeoutError):
                pass
        self._closed = True
        if self._process.poll() is None:
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
        for stream in (
            self._process.stdin,
            self._process.stdout,
            self._process.stderr,
        ):
            if stream is not None:
                stream.close()
        self._stdout_thread.join(timeout=1)
        self._stderr_thread.join(timeout=1)

    def __enter__(self) -> Sts2HeadlessClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        try:
            for raw_line in self._process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    self._logs.append(f"stdout: {line}")
                    continue
                if isinstance(value, Mapping):
                    self._responses.put(dict(value))
                else:
                    self._responses.put(
                        HeadlessRuntimeError(f"Non-object JSON response: {value!r}")
                    )
        except BaseException as exc:  # propagate reader failures to the caller
            self._responses.put(exc)
        finally:
            self._responses.put(self._exited_error())

    def _read_stderr(self) -> None:
        if self._process.stderr is None:
            return
        for raw_line in self._process.stderr:
            line = raw_line.rstrip()
            if line:
                self._logs.append(line)

    def _next_response(self, timeout: float) -> Mapping[str, Any]:
        try:
            value = self._responses.get(timeout=timeout)
        except Empty as exc:
            suffix = ""
            if self._logs:
                suffix = "\nRecent runtime logs:\n" + "\n".join(self._logs)
            raise TimeoutError(
                f"Headless runtime did not respond within {timeout:.1f}s{suffix}"
            ) from exc
        if isinstance(value, BaseException):
            raise HeadlessRuntimeError(str(value)) from value
        return value

    def _exited_error(self) -> HeadlessRuntimeError:
        code = self._process.poll()
        suffix = ""
        if self._logs:
            suffix = "\nRecent runtime logs:\n" + "\n".join(self._logs)
        return HeadlessRuntimeError(
            f"Headless runtime exited with code {code}{suffix}"
        )

    def _runtime_fingerprint(self) -> str:
        if self.config.command is None:
            runtime = self.config.runtime.resolve()
            components = {
                "runtime/" + runtime.name: hashlib.sha256(runtime.read_bytes()).hexdigest()
            }
            for sibling in sorted(runtime.parent.glob("*.dll"), key=lambda path: path.name):
                if sibling.resolve() == runtime:
                    continue
                components["runtime/" + sibling.name] = hashlib.sha256(
                    sibling.read_bytes()
                ).hexdigest()

            configured_lib = self.config.environment.get("STS2_LIB") or os.environ.get(
                "STS2_LIB"
            )
            lib_directory = Path(configured_lib).resolve() if configured_lib else None
            if lib_directory is None or not (lib_directory / "sts2.dll").is_file():
                for parent in runtime.parents:
                    candidate = parent / "lib"
                    if (candidate / "sts2.dll").is_file():
                        lib_directory = candidate
                        break
            if lib_directory is None or not (lib_directory / "sts2.dll").is_file():
                raise HeadlessSnapshotError(
                    "Could not fingerprint the native sts2.dll dependency directory"
                )
            for dependency in sorted(
                lib_directory.glob("*.dll"), key=lambda path: path.name
            ):
                components["lib/" + dependency.name] = hashlib.sha256(
                    dependency.read_bytes()
                ).hexdigest()
            return hashlib.sha256(
                _canonical_json(components).encode("utf-8")
            ).hexdigest()
        configuration = {
            "command": list(self.config.command),
            "environment": dict(sorted(self.config.environment.items())),
        }
        return hashlib.sha256(
            _canonical_json(configuration).encode("utf-8")
        ).hexdigest()

    def _track_snapshot_command(
        self, request: dict[str, Any], response: dict[str, Any]
    ) -> None:
        command = str(request.get("cmd", ""))
        genesis = {"start_run", "load_save"}
        mutations = genesis | {"action", "set_player", "enter_room", "set_draw_order"}
        nonmutating = {
            "get_map",
            "snapshot_checkpoint",
            "write_continue_save",
            "quit",
        }

        if command in nonmutating:
            return
        if command not in mutations:
            self._snapshot_taint = f"unclassified command {command or '<missing>'}"
            return
        if response.get("type") in {"error", "save_error"}:
            self._snapshot_taint = f"failed mutating command {command}"
            return

        step = HeadlessSnapshotStep(
            request_json=_canonical_json(request),
            response_sha256=_json_digest(response),
        )
        if command in genesis:
            self._snapshot_steps = [step]
            self._snapshot_taint = None
        else:
            if not self._snapshot_steps:
                self._snapshot_taint = f"{command} occurred before start_run or load_save"
                return
            self._snapshot_steps.append(step)
        self._snapshot_state = json.loads(_canonical_json(response))
