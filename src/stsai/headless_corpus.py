"""Deterministic native-headless corpus planning and teacher collection."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from stsai.combat import headless_combat_split
from stsai.headless import HeadlessConfig
from stsai.headless_env import HEADLESS_CHARACTERS
from stsai.headless_policy import run_heuristic_rollout
from stsai.ids import canonical_character


_CHARACTER_CODES = {
    "Defect": "D",
    "Ironclad": "I",
    "Necrobinder": "N",
    "Regent": "R",
    "Silent": "S",
}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _completed_rollout(
    path: Path, *, character: str, seed: str
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    first: Mapping[str, Any] | None = None
    last: Mapping[str, Any] | None = None
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, Mapping):
                return None
            if first is None:
                first = event
            last = event
    if (
        first is None
        or last is None
        or first.get("event_type") != "episode_start"
        or canonical_character(first.get("character")) != canonical_character(character)
        or str(first.get("seed")) != seed
        or last.get("event_type") != "episode_end"
        or last.get("truncated") is True
    ):
        return None
    return dict(last)


def _discard_incomplete(path: Path, *, character: str, seed: str) -> bool:
    """Remove a crash-orphaned .tmp for this exact case so reruns can resume."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    if not temporary.is_file():
        return False
    with temporary.open("r", encoding="utf-8-sig") as handle:
        first_line = handle.readline()
    try:
        first = json.loads(first_line)
    except json.JSONDecodeError:
        # A crash before the episode_start header finished writing leaves an
        # empty or truncated first line; nothing identifiable survives.
        temporary.unlink()
        return True
    if (
        not isinstance(first, Mapping)
        or first.get("event_type") != "episode_start"
        or canonical_character(first.get("character")) != canonical_character(character)
        or str(first.get("seed")) != seed
    ):
        raise ValueError(
            f"Refusing to remove mismatched incomplete trajectory: {temporary}"
        )
    temporary.unlink()
    return True


def create_combat_bootstrap_manifest(
    output: str | Path,
    *,
    build_id: str = "v0.107.1",
    ascension: int = 10,
    train_per_character: int = 12,
    validation_per_character: int = 4,
    test_per_character: int = 4,
) -> dict[str, Any]:
    """Precommit fresh character-balanced seeds with exact split counts."""

    targets = {
        "train": train_per_character,
        "validation": validation_per_character,
        "test": test_per_character,
    }
    if not build_id or not 0 <= ascension <= 10:
        raise ValueError("A build id and ascension from 0 through 10 are required")
    if any(count <= 0 for count in targets.values()):
        raise ValueError("Every per-character split count must be positive")
    cases: list[dict[str, Any]] = []
    for character in HEADLESS_CHARACTERS:
        counts: Counter[str] = Counter()
        candidate = 0
        while any(counts[split] < target for split, target in targets.items()):
            seed = f"CBT{_CHARACTER_CODES[character]}{candidate:06d}"
            candidate += 1
            split = headless_combat_split(
                build_id=build_id, character=character, seed=seed
            )
            if counts[split] >= targets[split]:
                continue
            cases.append(
                {
                    "character": canonical_character(character),
                    "seed": seed,
                    "split": split,
                }
            )
            counts[split] += 1
    payload = {
        "schema_version": 1,
        "name": "stsai-headless-combat-heuristic-bootstrap",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {"build_id": build_id, "ascension": ascension},
        "label_source": "deterministic_heuristic_distillation",
        "selection": {
            "method": "fresh-prefix-seeds-selected-by-stable-split-only",
            "seed_prefix": "CBT",
            "train_per_character": train_per_character,
            "validation_per_character": validation_per_character,
            "test_per_character": test_per_character,
            "outcomes_inspected": False,
        },
        "cases": cases,
    }
    _atomic_json(Path(output), payload)
    return payload


def collect_combat_bootstrap_corpus(
    manifest_path: str | Path,
    trajectory_dir: str | Path,
    *,
    macro_model_path: str | Path,
    max_steps: int = 1_000,
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Run the frozen heuristic teacher on every precommitted bootstrap case."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    target = manifest.get("target") or {}
    build_id = str(target.get("build_id", ""))
    ascension = int(target.get("ascension", 10))
    if not build_id:
        raise ValueError("Corpus manifest is missing target.build_id")
    destination = Path(trajectory_dir)
    destination.mkdir(parents=True, exist_ok=True)
    by_split: Counter[str] = Counter()
    by_character: Counter[str] = Counter()
    total_steps = 0
    truncations = 0
    reused = 0
    discarded_incomplete = 0
    cases = manifest.get("cases") or []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError(f"Invalid corpus case at index {index}")
        canonical = canonical_character(case.get("character"))
        character = canonical.rsplit(".", 1)[-1].title()
        seed = str(case.get("seed", ""))
        split = str(case.get("split", ""))
        actual_split = headless_combat_split(
            build_id=build_id, character=character, seed=seed
        )
        if split != actual_split:
            raise ValueError(
                f"Manifest split mismatch for {canonical}/{seed}: "
                f"recorded {split}, computed {actual_split}"
            )
        path = destination / f"{index:03d}.{split}.{character.lower()}.{seed}.jsonl"
        result = _completed_rollout(path, character=character, seed=seed)
        if result is None:
            discarded_incomplete += int(
                _discard_incomplete(path, character=character, seed=seed)
            )
            result = run_heuristic_rollout(
                model_path=macro_model_path,
                output=path,
                character=character,
                ascension=ascension,
                seed=seed,
                max_steps=max_steps,
                build_id=build_id,
                config=config,
            )
        else:
            reused += 1
        total_steps += int(result["steps"])
        truncations += int(result["truncated"] is True)
        by_split[split] += 1
        by_character[canonical] += 1
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "trajectory_dir": str(destination.resolve()),
        "runs": len(cases),
        "total_steps": total_steps,
        "truncations": truncations,
        "reused_completed_runs": reused,
        "discarded_incomplete": discarded_incomplete,
        "by_split": dict(by_split),
        "by_character": dict(by_character),
        "label_source": "deterministic_heuristic_distillation",
    }
