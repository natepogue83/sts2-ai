"""Fail-closed ingestion for opt-in first-party human combat demonstrations."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from stsai.combat import (
    COMBAT_STATE_TYPES,
    action_key,
    canonical_action,
    legal_combat_actions,
    normalize_combat_observation,
)
from stsai.ids import canonical_character


HUMAN_COMBAT_CHARACTERS = (
    "CHARACTER.IRONCLAD",
    "CHARACTER.SILENT",
    "CHARACTER.DEFECT",
    "CHARACTER.NECROBINDER",
    "CHARACTER.REGENT",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Frozen artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def _seed_hash(build_id: str, seed: str) -> str:
    return hashlib.sha256(f"{build_id}\x1f{seed}".encode()).hexdigest()


def _excluded_seed_receipts(
    manifests: Sequence[str | Path], build_id: str
) -> tuple[set[str], list[dict[str, Any]]]:
    hashes: set[str] = set()
    receipts: list[dict[str, Any]] = []
    for raw in manifests:
        path = Path(raw)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        seeds = {
            str(case["seed"])
            for case in payload.get("cases") or []
            if isinstance(case, Mapping) and case.get("seed") is not None
        }
        hashes.update(_seed_hash(build_id, seed) for seed in seeds)
        receipts.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "seed_count": len(seeds),
            }
        )
    return hashes, receipts


def create_human_combat_corpus_manifest(
    output: str | Path,
    *,
    game_assembly_path: str | Path,
    bridge_dll_path: str | Path,
    telemetry_patch_path: str | Path,
    excluded_manifests: Sequence[str | Path] = (),
    build_id: str = "v0.107.1",
    ascension: int = 10,
    decisions_per_character: int = 100,
) -> dict[str, Any]:
    """Freeze source provenance, balance, and promotion gates before collection."""

    if not build_id or not 0 <= ascension <= 10 or decisions_per_character <= 0:
        raise ValueError("Build, ascension, and a positive decision target are required")
    excluded_hashes, exclusion_receipts = _excluded_seed_receipts(
        excluded_manifests, build_id
    )
    payload = {
        "schema_version": 1,
        "name": "stsai-frozen-first-party-human-combat-corpus",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {
            "build_id": build_id,
            "ascension": ascension,
            "mode": "singleplayer",
            "characters": list(HUMAN_COMBAT_CHARACTERS),
            "decisions_per_character": decisions_per_character,
            "total_decisions": decisions_per_character * len(HUMAN_COMBAT_CHARACTERS),
        },
        "source": {
            "ownership": "first_party_opt_in_user_gameplay",
            "collector": "STS2MCP",
            "event_schema_version": 2,
            "event_type": "human_combat_action",
            "source": "game_ui",
            "policy_suggestions_allowed_during_collection": False,
            "bridge_actions_allowed": False,
            "game_assembly_sha256": _sha256(game_assembly_path),
            "bridge_dll_sha256": _sha256(bridge_dll_path),
            "telemetry_patch_sha256": _sha256(telemetry_patch_path),
        },
        "quality_gate": {
            "fail_closed": True,
            "canonical_character_id_required": True,
            "chosen_action_must_be_uniquely_legal": True,
            "run_scoped_anonymous_trajectory_required": True,
            "monotonic_event_step_required": True,
            "duplicate_events_allowed": False,
            "mixed_builds_allowed": False,
            "mixed_game_assemblies_allowed": False,
            "balanced_selection": "stable-hash sample exactly the target per character",
            "output_split": "train",
            "next_observation_semantics": "not_recorded_for_pre_action_behavior_cloning",
        },
        "exclusions": {
            "manifests": exclusion_receipts,
            "reserved_seed_hash_count": len(excluded_hashes),
            "comparison_is_hash_only": True,
        },
        "training": {
            "candidate_count": 1,
            "architecture": "typed character-conditioned Deep Sets combat scorer",
            "base_validation_and_test_splits_remain_frozen": True,
            "human_rows_are_train_only": True,
        },
        "promotion_gate": {
            "evaluation": "fresh paired character-balanced headless seeds after checkpoint freeze",
            "overall_mean_paired_floor_delta_at_least": 1.0,
            "every_character_mean_paired_floor_delta_at_least": -0.5,
            "candidate_victories_at_least_baseline": True,
            "candidate_runtime_failures_at_most_baseline": True,
            "candidate_truncations_at_most_baseline": True,
            "if_failed": "reject the candidate without tuning on its evaluation seeds",
        },
        "cases": [],
    }
    _atomic_json(output, payload)
    return payload


def _discover_sources(sources: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for raw in sources:
        path = Path(raw)
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.lower() in {".jsonl", ".ndjson"}
            )
        else:
            raise FileNotFoundError(path)
    result = sorted(set(paths))
    if not result:
        raise ValueError("No human telemetry files were found")
    return result


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _trajectory_key(
    *, build_id: str, character: str, seed_hash: str, trajectory_id: str
) -> str:
    payload = f"{build_id}\x1f{character}\x1f{seed_hash}\x1f{trajectory_id}"
    return hashlib.sha256(payload.encode()).hexdigest()


def build_human_combat_dataset(
    sources: Sequence[str | Path],
    output: str | Path,
    *,
    manifest_path: str | Path,
    receipt_output: str | Path | None = None,
) -> dict[str, Any]:
    """Build one balanced train-only corpus from strict schema-v2 UI demonstrations."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("name") != "stsai-frozen-first-party-human-combat-corpus":
        raise ValueError("Manifest is not a frozen first-party human combat corpus")
    target = _mapping(manifest.get("target"))
    source_contract = _mapping(manifest.get("source"))
    build_id = str(target.get("build_id") or "")
    ascension = int(target.get("ascension", -1))
    expected_characters = tuple(str(value) for value in target.get("characters") or [])
    decisions_per_character = int(target.get("decisions_per_character") or 0)
    game_assembly_sha256 = str(source_contract.get("game_assembly_sha256") or "")
    if expected_characters != HUMAN_COMBAT_CHARACTERS or decisions_per_character <= 0:
        raise ValueError("Human corpus manifest has an invalid character balance contract")
    if not _SHA256.fullmatch(game_assembly_sha256):
        raise ValueError("Human corpus manifest has no valid game assembly hash")

    reserved_hashes: set[str] = set()
    for receipt in _mapping(manifest.get("exclusions")).get("manifests") or []:
        receipt = _mapping(receipt)
        path = Path(str(receipt.get("path") or ""))
        if _sha256(path) != str(receipt.get("sha256")):
            raise ValueError(f"Reserved-seed manifest changed after freeze: {path}")
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        reserved_hashes.update(
            _seed_hash(build_id, str(case["seed"]))
            for case in payload.get("cases") or []
            if isinstance(case, Mapping) and case.get("seed") is not None
        )

    paths = _discover_sources(sources)
    accepted: dict[str, list[tuple[str, dict[str, Any], str, str]]] = defaultdict(list)
    seen_events: set[tuple[str, int]] = set()
    trajectory_metadata: dict[str, tuple[str, str]] = {}
    source_events = 0
    action_types: Counter[str] = Counter()
    valid_by_character: Counter[str] = Counter()
    trajectory_sets: dict[str, set[str]] = defaultdict(set)
    combat_sets: dict[str, set[str]] = defaultdict(set)

    for path in paths:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid telemetry JSON in {path} line {line_number}") from exc
                if not isinstance(event, Mapping):
                    raise ValueError(f"Telemetry event is not an object in {path} line {line_number}")
                source_events += 1
                required_identity = (
                    event.get("schema_version") == 2
                    and event.get("event_type") == "human_combat_action"
                    and event.get("source") == "game_ui"
                    and event.get("collector") == "STS2MCP"
                    and event.get("mode") == "singleplayer"
                )
                if not required_identity:
                    raise ValueError(f"Unexpected telemetry source/schema in {path} line {line_number}")
                if str(event.get("build_id")) != build_id:
                    raise ValueError(f"Wrong telemetry build in {path} line {line_number}")
                if str(event.get("game_assembly_sha256")) != game_assembly_sha256:
                    raise ValueError(f"Wrong game assembly in {path} line {line_number}")

                trajectory_id = str(event.get("trajectory_id") or "")
                seed_hash = str(event.get("run_seed_sha256") or "")
                step = event.get("step")
                if not trajectory_id or not isinstance(step, int) or step < 0:
                    raise ValueError(f"Missing run trajectory/step in {path} line {line_number}")
                if not _SHA256.fullmatch(seed_hash):
                    raise ValueError(f"Missing run seed hash in {path} line {line_number}")
                if seed_hash in reserved_hashes:
                    raise ValueError(f"Telemetry overlaps a reserved evaluation seed in {path}")
                event_key = (trajectory_id, step)
                if event_key in seen_events:
                    raise ValueError(f"Duplicate telemetry event {trajectory_id}:{step}")
                seen_events.add(event_key)

                observation = _mapping(event.get("observation"))
                run = _mapping(observation.get("run"))
                player = _mapping(observation.get("player"))
                if str(observation.get("state_type")) not in COMBAT_STATE_TYPES:
                    raise ValueError(f"Noncombat telemetry event in {path} line {line_number}")
                if int(run.get("ascension", -1)) != ascension:
                    raise ValueError(f"Wrong telemetry ascension in {path} line {line_number}")
                character_id = str(player.get("character_id") or "")
                character = canonical_character(character_id)
                if character_id == "" or character not in HUMAN_COMBAT_CHARACTERS:
                    raise ValueError(f"Missing or unknown canonical character in {path} line {line_number}")
                metadata = (character, seed_hash)
                previous = trajectory_metadata.setdefault(trajectory_id, metadata)
                if previous != metadata:
                    raise ValueError(f"Trajectory metadata changed within {trajectory_id}")

                legal_actions = legal_combat_actions(observation)
                try:
                    chosen = canonical_action(_mapping(event.get("action")))
                    chosen_key = action_key(chosen)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid chosen action in {path} line {line_number}") from exc
                if sum(action_key(action) == chosen_key for action in legal_actions) != 1:
                    raise ValueError(f"Chosen action is not uniquely legal in {path} line {line_number}")

                trajectory_key = _trajectory_key(
                    build_id=build_id,
                    character=character,
                    seed_hash=seed_hash,
                    trajectory_id=trajectory_id,
                )
                normalized = normalize_combat_observation(observation)
                record = {
                    "schema_version": 1,
                    "decision_type": "combat_action",
                    "split": "train",
                    "trajectory_key": trajectory_key,
                    "patch": {
                        "build_id": build_id,
                        "bridge": "STS2MCP",
                        "label_source": "human_ui_demonstration",
                        "telemetry_schema_version": 2,
                        "game_assembly_sha256": game_assembly_sha256,
                    },
                    "observation": normalized,
                    "legal_actions": legal_actions,
                    "chosen_action": chosen,
                    "next_observation": None,
                    "terminal": False,
                }
                selection_hash = hashlib.sha256(
                    f"{trajectory_key}\x1f{step}".encode()
                ).hexdigest()
                floor = str(_mapping(normalized.get("run")).get("floor"))
                combat_key = f"{trajectory_key}:{floor}"
                accepted[character].append(
                    (selection_hash, record, trajectory_key, combat_key)
                )
                valid_by_character[character] += 1
                action_types[chosen["action"]] += 1

    selected_records: list[dict[str, Any]] = []
    selected_by_character: Counter[str] = Counter()
    for character in HUMAN_COMBAT_CHARACTERS:
        choices = sorted(accepted.get(character, []), key=lambda item: item[0])
        if len(choices) < decisions_per_character:
            raise ValueError(
                f"Human corpus is short for {character}: "
                f"{len(choices)}/{decisions_per_character} valid decisions"
            )
        for _, record, trajectory_key, combat_key in choices[:decisions_per_character]:
            selected_records.append(record)
            selected_by_character[character] += 1
            trajectory_sets[character].add(trajectory_key)
            combat_sets[character].add(combat_key)

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Frozen human combat dataset already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for record in selected_records:
                handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    receipt = {
        "schema_version": 1,
        "dataset": str(destination.resolve()),
        "dataset_sha256": _sha256(destination),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "source_files": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in paths
        ],
        "source_events": source_events,
        "valid_events_by_character": dict(valid_by_character),
        "selected_events_by_character": dict(selected_by_character),
        "selected_trajectories_by_character": {
            character: len(trajectory_sets[character])
            for character in HUMAN_COMBAT_CHARACTERS
        },
        "selected_combats_by_character": {
            character: len(combat_sets[character])
            for character in HUMAN_COMBAT_CHARACTERS
        },
        "valid_action_types": dict(action_types),
        "reserved_seed_hashes_checked": len(reserved_hashes),
        "output_split": "train",
        "privacy": {
            "raw_run_seed_persisted": False,
            "run_seed_hash_persisted": False,
            "player_identity_persisted": False,
            "source_path_persisted_in_dataset": False,
        },
    }
    if receipt_output is not None:
        _atomic_json(receipt_output, receipt)
    return receipt
