"""Precommitted DAgger collection on native learner-visited combat states."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from stsai.combat import (
    CombatDatasetSummary,
    action_key,
    headless_action_to_combat,
    headless_combat_trajectory_key,
    normalize_headless_combat_observation,
)
from stsai.headless import HeadlessConfig, HeadlessRuntimeError
from stsai.headless_env import HEADLESS_CHARACTERS, Sts2Env, run_headless_rollout
from stsai.headless_policy import HeadlessHybridPolicy
from stsai.ids import canonical_character
from stsai.policy import MaskedMLPPolicy


_CHARACTER_CODES = {
    "Defect": "D",
    "Ironclad": "I",
    "Necrobinder": "N",
    "Regent": "R",
    "Silent": "S",
}
_MANIFEST_NAME = "stsai-headless-combat-dagger"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _objects(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _mixing_seed(*, build_id: str, round_number: int, character: str, seed: str) -> int:
    payload = f"{build_id}\x1f{round_number}\x1f{canonical_character(character)}\x1f{seed}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _policy_name(round_number: int, beta: float) -> str:
    return f"dagger_round_{round_number}_beta_{beta:.3f}"


def create_dagger_manifest(
    output: str | Path,
    *,
    build_id: str = "v0.107.1",
    ascension: int = 10,
    round_number: int = 1,
    beta: float = 0.5,
    aggregate_per_character: int = 20,
    validation_per_character: int = 5,
    learner_name: str = "combat_policy.heuristic.v0.107.1",
    learner_model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Reserve fresh aggregation and leakage-proof validation seeds."""

    if not build_id or not 0 <= ascension <= 10:
        raise ValueError("A build id and ascension from 0 through 10 are required")
    if round_number <= 0:
        raise ValueError("DAgger round number must be positive")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("DAgger beta must be between zero and one")
    if aggregate_per_character <= 0 or validation_per_character <= 0:
        raise ValueError("Aggregation and validation counts must be positive")

    cases: list[dict[str, Any]] = []
    prefix = f"DGR{round_number}"
    for character in HEADLESS_CHARACTERS:
        for index in range(aggregate_per_character + validation_per_character):
            seed = f"{prefix}{_CHARACTER_CODES[character]}{index:06d}"
            pool = "aggregate" if index < aggregate_per_character else "validation"
            cases.append(
                {
                    "character": canonical_character(character),
                    "seed": seed,
                    "pool": pool,
                    "mixing_seed": _mixing_seed(
                        build_id=build_id,
                        round_number=round_number,
                        character=character,
                        seed=seed,
                    ),
                }
            )
    dagger = {
        "round": round_number,
        "beta": beta,
        "teacher": "greedy_rules_v1",
        "learner": learner_name,
    }
    if learner_model_path is not None:
        dagger["learner_model_sha256"] = _sha256(learner_model_path)
    payload = {
        "schema_version": 1,
        "name": _MANIFEST_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {"build_id": build_id, "ascension": ascension},
        "dagger": dagger,
        "selection": {
            "method": "fresh-prefix-seeds-selected-before-outcomes",
            "seed_prefix": prefix,
            "aggregate_per_character": aggregate_per_character,
            "validation_per_character": validation_per_character,
            "validation_aggregation_allowed": False,
            "outcomes_inspected": False,
        },
        "cases": cases,
    }
    _atomic_json(Path(output), payload)
    return payload


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if manifest.get("name") != _MANIFEST_NAME:
        raise ValueError("Not an STSAI DAgger manifest")
    target = _mapping(manifest.get("target"))
    dagger = _mapping(manifest.get("dagger"))
    if not target.get("build_id") or int(dagger.get("round", 0)) <= 0:
        raise ValueError("DAgger manifest is missing its build or round")
    beta = float(dagger.get("beta", -1.0))
    if not 0.0 <= beta <= 1.0:
        raise ValueError("DAgger manifest beta is invalid")
    cases = _objects(manifest.get("cases"))
    identities: set[tuple[str, str]] = set()
    for index, case in enumerate(cases):
        canonical = canonical_character(case.get("character"))
        seed = str(case.get("seed", ""))
        pool = str(case.get("pool", ""))
        if not seed or pool not in {"aggregate", "validation"}:
            raise ValueError(f"Invalid DAgger case at index {index}")
        identity = (canonical, seed)
        if identity in identities:
            raise ValueError(f"Duplicate DAgger case: {canonical}/{seed}")
        identities.add(identity)
        expected = _mixing_seed(
            build_id=str(target["build_id"]),
            round_number=int(dagger["round"]),
            character=canonical,
            seed=seed,
        )
        if int(case.get("mixing_seed", -1)) != expected:
            raise ValueError(f"DAgger mixing seed mismatch for {canonical}/{seed}")
    return manifest


class DaggerMixturePolicy:
    """Query the teacher at every combat state and beta-mix executed actions."""

    def __init__(
        self,
        macro_model: MaskedMLPPolicy,
        combat_model: MaskedMLPPolicy,
        *,
        beta: float,
        mixing_seed: int,
        round_number: int = 1,
        build_id: str = "v0.107.1",
    ):
        if not 0.0 <= beta <= 1.0:
            raise ValueError("DAgger beta must be between zero and one")
        self.beta = beta
        self.round_number = round_number
        self.rng = random.Random(mixing_seed)
        self.teacher = HeadlessHybridPolicy(
            macro_model, build_id=build_id, use_macro=False
        )
        self.learner = HeadlessHybridPolicy(
            macro_model,
            build_id=build_id,
            use_macro=False,
            combat_model=combat_model,
        )

    def select(self, env: Sts2Env, step: int = 0) -> tuple[int, Mapping[str, Any]]:
        if str(env.raw_state.get("decision", "")) != "combat_play":
            return self.teacher.select(env, step)

        teacher_index, teacher_info = self.teacher.select(env, step)
        learner_index, learner_info = self.learner.select(env, step)
        draw = self.rng.random()
        executed_by = "teacher" if draw < self.beta else "learner"
        action_index = teacher_index if executed_by == "teacher" else learner_index
        teacher_action = env.legal_actions[teacher_index].as_dict()
        learner_action = env.legal_actions[learner_index].as_dict()
        return action_index, {
            "source": "dagger_mixture",
            "dagger": {
                "round": self.round_number,
                "beta": self.beta,
                "mixing_draw": round(draw, 12),
                "executed_by": executed_by,
                "teacher_action_index": teacher_index,
                "teacher_action": teacher_action,
                "learner_action_index": learner_index,
                "learner_action": learner_action,
                "teacher_learner_agree": teacher_action == learner_action,
                "teacher_policy": teacher_info.get("policy", "greedy_rules_v1"),
                "learner_source": learner_info.get("source", "combat_bc"),
            },
        }


def run_dagger_rollout(
    *,
    macro_model_path: str | Path,
    combat_model_path: str | Path,
    output: str | Path,
    character: str,
    ascension: int,
    seed: str,
    beta: float,
    mixing_seed: int,
    round_number: int = 1,
    max_steps: int = 1_000,
    build_id: str = "v0.107.1",
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Run one atomic beta-mixed DAgger episode."""

    macro_model_path = Path(macro_model_path)
    combat_model_path = Path(combat_model_path)
    policy = DaggerMixturePolicy(
        MaskedMLPPolicy.load(macro_model_path),
        MaskedMLPPolicy.load(combat_model_path),
        beta=beta,
        mixing_seed=mixing_seed,
        round_number=round_number,
        build_id=build_id,
    )
    return run_headless_rollout(
        output=output,
        character=character,
        ascension=ascension,
        seed=seed,
        policy_name=_policy_name(round_number, beta),
        select_action=policy.select,
        policy_metadata={
            "build_id": build_id,
            "dagger_round": round_number,
            "beta": beta,
            "mixing_seed": mixing_seed,
            "teacher": "greedy_rules_v1",
            "combat_model": str(combat_model_path.resolve()),
            "combat_model_sha256": _sha256(combat_model_path),
            "macro_model_loaded_for_interface_only": str(macro_model_path.resolve()),
            "macro_model_sha256": _sha256(macro_model_path),
            "validation_data": False,
        },
        max_steps=max_steps,
        config=config,
    )


def _iter_events(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
            if not isinstance(event, Mapping):
                raise ValueError(f"Non-object event in {path} at line {line_number}")
            yield event


def _completed_dagger_rollout(
    path: Path,
    *,
    character: str,
    seed: str,
    policy_name: str,
    expected_metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    events = list(_iter_events(path))
    if not events:
        raise ValueError(f"Published DAgger trajectory is empty: {path}")
    first = events[0]
    last = events[-1]
    metadata = _mapping(first.get("policy_metadata"))
    mismatch = (
        first.get("event_type") != "episode_start"
        or canonical_character(first.get("character")) != canonical_character(character)
        or str(first.get("seed")) != seed
        or first.get("policy") != policy_name
        or any(metadata.get(key) != value for key, value in expected_metadata.items())
        or last.get("event_type") != "episode_end"
    )
    if mismatch:
        raise ValueError(f"Published DAgger trajectory identity mismatch: {path}")
    return dict(last)


def _discard_matching_incomplete(
    path: Path,
    *,
    character: str,
    seed: str,
    policy_name: str,
    expected_metadata: Mapping[str, Any],
) -> bool:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not temporary.is_file():
        return False
    first = next(_iter_events(temporary), None)
    metadata = _mapping(first.get("policy_metadata")) if first else {}
    if (
        first is None
        or first.get("event_type") != "episode_start"
        or canonical_character(first.get("character")) != canonical_character(character)
        or str(first.get("seed")) != seed
        or first.get("policy") != policy_name
        or any(metadata.get(key) != value for key, value in expected_metadata.items())
    ):
        raise FileExistsError(f"Unmatched incomplete trajectory: {temporary}")
    temporary.unlink()
    return True


def _trajectory_stats(path: Path) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    ending: Mapping[str, Any] = {}
    for event in _iter_events(path):
        if event.get("event_type") == "episode_end":
            ending = event
        if event.get("event_type") != "transition":
            continue
        dagger = _mapping(_mapping(event.get("info")).get("policy")).get("dagger")
        dagger = _mapping(dagger)
        if not dagger:
            continue
        stats["combat_decisions"] += 1
        stats[f"executed_{dagger.get('executed_by')}"] += 1
        stats["agreements"] += int(dagger.get("teacher_learner_agree") is True)
        teacher_action = _mapping(dagger.get("teacher_action"))
        stats[f"teacher_{teacher_action.get('action', 'unknown')}"] += 1
    return {
        **dict(stats),
        "steps": int(ending.get("steps", 0)),
        "floor": int(ending.get("floor") or 0),
        "truncated": ending.get("truncated") is True,
        "victory": ending.get("victory") is True,
        "runtime_failure": bool(ending.get("runtime_error")),
    }


def _publish_runtime_failure(path: Path, error: BaseException) -> dict[str, Any]:
    """Atomically retain valid pre-failure DAgger transitions as a truncation."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    if not temporary.is_file():
        raise FileNotFoundError(f"Runtime failure left no trajectory to publish: {temporary}")
    events = list(_iter_events(temporary))
    if not events or events[0].get("event_type") != "episode_start":
        raise ValueError(f"Runtime failure trajectory has no episode metadata: {temporary}")
    transitions = [event for event in events if event.get("event_type") == "transition"]
    last = transitions[-1] if transitions else {}
    state = _mapping(last.get("next_observation")) or _mapping(last.get("observation"))
    context = _mapping(state.get("context"))
    policy_sources: Counter[str] = Counter()
    for transition in transitions:
        policy = _mapping(_mapping(transition.get("info")).get("policy"))
        policy_sources[str(policy.get("source", "unknown"))] += 1
    ending = {
        "schema_version": 1,
        "event_type": "episode_end",
        "trajectory_id": events[0].get("trajectory_id"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "steps": max((int(event.get("step", 0)) for event in transitions), default=0),
        "total_reward": sum(float(event.get("reward", 0.0)) for event in transitions),
        "terminated": False,
        "truncated": True,
        "victory": None,
        "act": context.get("act"),
        "floor": context.get("floor"),
        "policy_sources": dict(policy_sources),
        "runtime_error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }
    with temporary.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(ending, separators=(",", ":"), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return ending


def collect_dagger_round(
    manifest_path: str | Path,
    trajectory_dir: str | Path,
    *,
    macro_model_path: str | Path,
    combat_model_path: str | Path,
    output: str | Path | None = None,
    max_steps: int = 1_000,
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Collect only the manifest's aggregation pool; validation is never run here."""

    manifest_path = Path(manifest_path)
    manifest = _load_manifest(manifest_path)
    target = _mapping(manifest["target"])
    dagger = _mapping(manifest["dagger"])
    build_id = str(target["build_id"])
    ascension = int(target.get("ascension", 10))
    round_number = int(dagger["round"])
    beta = float(dagger["beta"])
    policy_name = _policy_name(round_number, beta)
    macro_sha = _sha256(macro_model_path)
    combat_sha = _sha256(combat_model_path)
    expected_learner_sha = dagger.get("learner_model_sha256")
    if expected_learner_sha and str(expected_learner_sha) != combat_sha:
        raise ValueError("DAgger learner checkpoint does not match the frozen manifest")
    destination = Path(trajectory_dir)
    destination.mkdir(parents=True, exist_ok=True)
    aggregate_cases = [case for case in manifest["cases"] if case["pool"] == "aggregate"]
    validation_cases = [case for case in manifest["cases"] if case["pool"] == "validation"]
    by_character: dict[str, Counter[str]] = {}
    reused = 0
    discarded = 0
    truncations = 0
    runtime_failures = 0

    for index, case in enumerate(aggregate_cases):
        canonical = canonical_character(case["character"])
        character = canonical.rsplit(".", 1)[-1].title()
        seed = str(case["seed"])
        mixing_seed = int(case["mixing_seed"])
        expected_metadata = {
            "build_id": build_id,
            "dagger_round": round_number,
            "beta": beta,
            "mixing_seed": mixing_seed,
            "combat_model_sha256": combat_sha,
            "macro_model_sha256": macro_sha,
            "validation_data": False,
        }
        path = destination / f"{index:03d}.aggregate.{character.lower()}.{seed}.jsonl"
        result = _completed_dagger_rollout(
            path,
            character=character,
            seed=seed,
            policy_name=policy_name,
            expected_metadata=expected_metadata,
        )
        if result is None:
            discarded += int(
                _discard_matching_incomplete(
                    path,
                    character=character,
                    seed=seed,
                    policy_name=policy_name,
                    expected_metadata=expected_metadata,
                )
            )
            try:
                run_dagger_rollout(
                    macro_model_path=macro_model_path,
                    combat_model_path=combat_model_path,
                    output=path,
                    character=character,
                    ascension=ascension,
                    seed=seed,
                    beta=beta,
                    mixing_seed=mixing_seed,
                    round_number=round_number,
                    max_steps=max_steps,
                    build_id=build_id,
                    config=config,
                )
            except (HeadlessRuntimeError, TimeoutError) as error:
                _publish_runtime_failure(path, error)
        else:
            reused += 1
        stats = _trajectory_stats(path)
        truncations += int(stats["truncated"])
        runtime_failures += int(stats["runtime_failure"])
        character_stats = by_character.setdefault(canonical, Counter())
        character_stats["runs"] += 1
        for key in (
            "steps",
            "floor",
            "combat_decisions",
            "executed_teacher",
            "executed_learner",
            "agreements",
            "teacher_play_card",
            "teacher_end_turn",
            "teacher_use_potion",
            "truncated",
            "victory",
            "runtime_failure",
        ):
            character_stats[key] += stats.get(key, 0)

    character_results: dict[str, dict[str, Any]] = {}
    for character, stats in sorted(by_character.items()):
        runs = int(stats["runs"])
        decisions = int(stats["combat_decisions"])
        character_results[character] = {
            **{key: int(value) for key, value in stats.items()},
            "mean_floor": float(stats["floor"] / runs) if runs else 0.0,
            "teacher_execution_rate": (
                float(stats["executed_teacher"] / decisions) if decisions else 0.0
            ),
            "teacher_learner_agreement_rate": (
                float(stats["agreements"] / decisions) if decisions else 0.0
            ),
        }
    payload = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "trajectory_dir": str(destination.resolve()),
        "round": round_number,
        "beta": beta,
        "runs": len(aggregate_cases),
        "validation_cases_reserved_not_collected": len(validation_cases),
        "truncations": truncations,
        "runtime_failures": runtime_failures,
        "reused_completed_runs": reused,
        "discarded_matching_incomplete_runs": discarded,
        "by_character": character_results,
    }
    if output is not None:
        _atomic_json(Path(output), payload)
    return payload


def _discover_trajectories(sources: Sequence[str | os.PathLike[str]]) -> list[Path]:
    paths: list[Path] = []
    for raw in sources:
        source = Path(raw)
        if source.is_file():
            paths.append(source)
        elif source.is_dir():
            paths.extend(
                path
                for path in source.rglob("*")
                if path.is_file() and path.suffix.lower() in {".jsonl", ".ndjson"}
            )
        else:
            raise FileNotFoundError(source)
    return sorted(set(paths))


def build_dagger_dataset(
    sources: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
    *,
    manifest_path: str | Path,
    build_id: str,
) -> CombatDatasetSummary:
    """Build teacher-labeled records while structurally excluding validation cases."""

    manifest = _load_manifest(manifest_path)
    target = _mapping(manifest["target"])
    dagger_config = _mapping(manifest["dagger"])
    if str(target["build_id"]) != build_id:
        raise ValueError("DAgger manifest build does not match requested build")
    aggregate = {
        (canonical_character(case["character"]), str(case["seed"]))
        for case in manifest["cases"]
        if case["pool"] == "aggregate"
    }
    validation = {
        (canonical_character(case["character"]), str(case["seed"]))
        for case in manifest["cases"]
        if case["pool"] == "validation"
    }
    paths = _discover_trajectories(sources)
    if not paths:
        raise ValueError("No DAgger trajectory files were found")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    summary = CombatDatasetSummary()
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for path in paths:
                summary.files_seen += 1
                episode: Mapping[str, Any] = {}
                identity: tuple[str, str] | None = None
                for event in _iter_events(path):
                    summary.events_seen += 1
                    if event.get("event_type") == "episode_start":
                        episode = event
                        identity = (
                            canonical_character(event.get("character")),
                            str(event.get("seed", "")),
                        )
                        if identity in validation:
                            raise ValueError(
                                f"Validation trajectory cannot be aggregated: {path}"
                            )
                        if identity not in aggregate:
                            raise ValueError(f"Trajectory is not in the aggregation pool: {path}")
                        learner_sha = dagger_config.get("learner_model_sha256")
                        metadata = _mapping(event.get("policy_metadata"))
                        if learner_sha and metadata.get("combat_model_sha256") != learner_sha:
                            raise ValueError(
                                f"Trajectory learner does not match DAgger manifest: {path}"
                            )
                        continue
                    if event.get("event_type") != "transition":
                        continue
                    if identity is None:
                        raise ValueError(f"Transition precedes episode metadata: {path}")
                    observation = _mapping(event.get("observation"))
                    if observation.get("decision_type") != "combat_play":
                        summary.skipped_noncombat += 1
                        continue
                    policy = _mapping(_mapping(event.get("info")).get("policy"))
                    dagger = _mapping(policy.get("dagger"))
                    if not dagger:
                        summary.skipped_wrong_source += 1
                        continue
                    if (
                        int(dagger.get("round", -1)) != int(dagger_config["round"])
                        or float(dagger.get("beta", -1.0))
                        != float(dagger_config["beta"])
                        or dagger.get("executed_by") not in {"teacher", "learner"}
                    ):
                        raise ValueError(f"Transition does not match DAgger manifest: {path}")
                    try:
                        legal_actions = [
                            headless_action_to_combat(action)
                            for action in _objects(event.get("legal_actions"))
                        ]
                        executed_action = headless_action_to_combat(
                            _mapping(event.get("action"))
                        )
                        teacher_action = headless_action_to_combat(
                            _mapping(dagger.get("teacher_action"))
                        )
                        learner_action = headless_action_to_combat(
                            _mapping(dagger.get("learner_action"))
                        )
                    except (KeyError, TypeError, ValueError):
                        summary.skipped_illegal += 1
                        continue
                    legal_keys = {action_key(action) for action in legal_actions}
                    if any(
                        action_key(action) not in legal_keys
                        for action in (executed_action, teacher_action, learner_action)
                    ):
                        summary.skipped_illegal += 1
                        continue
                    character = episode.get("character") or observation.get("character")
                    seed = episode.get("seed") or observation.get("seed")
                    normalized = normalize_headless_combat_observation(observation)
                    next_state = _mapping(event.get("next_observation"))
                    record = {
                        "schema_version": 1,
                        "decision_type": "combat_action",
                        "split": "train",
                        "trajectory_key": headless_combat_trajectory_key(
                            build_id=build_id, character=character, seed=seed
                        ),
                        "patch": {
                            "build_id": build_id,
                            "runtime": "sts2-cli",
                            "label_source": "dagger_heuristic_teacher",
                        },
                        "observation": normalized,
                        "legal_actions": legal_actions,
                        "chosen_action": teacher_action,
                        "teacher_action": teacher_action,
                        "learner_action": learner_action,
                        "executed_action": executed_action,
                        "dagger": {
                            "round": int(dagger_config["round"]),
                            "beta": float(dagger_config["beta"]),
                            "executed_by": dagger.get("executed_by"),
                            "teacher_learner_agree": (
                                dagger.get("teacher_learner_agree") is True
                            ),
                            "mixing_draw": dagger.get("mixing_draw"),
                        },
                        "next_observation": (
                            normalize_headless_combat_observation(next_state)
                            if next_state.get("decision_type") == "combat_play"
                            else None
                        ),
                        "terminal": event.get("terminated") is True,
                    }
                    handle.write(
                        json.dumps(record, separators=(",", ":"), ensure_ascii=False)
                        + "\n"
                    )
                    summary.decisions_written += 1
                    summary.splits["train"] += 1
                    summary.action_types[teacher_action["action"]] += 1
                    summary.characters[normalized["player"]["character"]] += 1
                    summary.sources["headless_dagger_heuristic_teacher"] += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return summary


def merge_dagger_training_dataset(
    base_dataset: str | Path,
    dagger_dataset: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Atomically add DAgger train labels while preserving frozen base splits."""

    base_path = Path(base_dataset)
    dagger_path = Path(dagger_dataset)
    base_records = list(_iter_events(base_path))
    dagger_records = list(_iter_events(dagger_path))
    if not base_records or not dagger_records:
        raise ValueError("Both the base and DAgger datasets must be non-empty")
    allowed_splits = {"train", "validation", "test"}
    if any(record.get("split") not in allowed_splits for record in base_records):
        raise ValueError("Base combat dataset contains an invalid split")
    if any(record.get("split") != "train" for record in dagger_records):
        raise ValueError("DAgger aggregation records must all be training data")
    if any(
        _mapping(record.get("patch")).get("label_source")
        != "dagger_heuristic_teacher"
        for record in dagger_records
    ):
        raise ValueError("DAgger dataset contains an unexpected label source")
    base_keys = {str(record.get("trajectory_key")) for record in base_records}
    dagger_keys = {str(record.get("trajectory_key")) for record in dagger_records}
    overlap = (base_keys & dagger_keys) - {"None", ""}
    if overlap:
        raise ValueError("Base and DAgger datasets contain overlapping trajectories")

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    split_counts: Counter[str] = Counter()
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in (*base_records, *dagger_records):
                handle.write(
                    json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
                )
                split_counts[str(record["split"])] += 1
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "base_dataset": str(base_path.resolve()),
        "base_sha256": _sha256(base_path),
        "dagger_dataset": str(dagger_path.resolve()),
        "dagger_sha256": _sha256(dagger_path),
        "output": str(destination.resolve()),
        "output_sha256": _sha256(destination),
        "base_records": len(base_records),
        "dagger_train_records": len(dagger_records),
        "records": len(base_records) + len(dagger_records),
        "splits": dict(split_counts),
        "base_trajectory_keys": len(base_keys),
        "dagger_trajectory_keys": len(dagger_keys),
        "trajectory_overlap": 0,
    }
