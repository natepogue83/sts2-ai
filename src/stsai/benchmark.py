"""Fixed-seed benchmark manifests and first-attempt run evaluation."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from stsai.runs import iter_map_points, read_runs


def _character(run: Mapping[str, Any]) -> str | None:
    players = run.get("players")
    if not isinstance(players, list) or len(players) != 1 or not isinstance(players[0], dict):
        return None
    value = players[0].get("character")
    return str(value) if value else None


def create_benchmark(
    sources: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
    *,
    build_id: str,
    ascension: int = 10,
    count_per_character: int = 20,
    source_build_id: str | None = None,
) -> dict[str, Any]:
    """Choose deterministic seed/character cases without looking at outcomes."""

    if count_per_character < 1:
        raise ValueError("count_per_character must be positive")
    available: dict[str, set[str]] = defaultdict(set)
    for source in sources:
        for run in read_runs(source):
            if int(run.get("ascension") or 0) != ascension:
                continue
            if source_build_id and str(run.get("build_id")) != source_build_id:
                continue
            character = _character(run)
            seed = run.get("seed")
            if character and seed:
                available[character].add(str(seed))

    cases: list[dict[str, str]] = []
    for character in sorted(available):
        ranked = sorted(
            available[character],
            key=lambda seed: hashlib.sha256(f"{character}\x1f{seed}".encode()).digest(),
        )
        cases.extend(
            {"character": character, "seed": seed}
            for seed in ranked[:count_per_character]
        )
    if not cases:
        raise ValueError("No eligible seed/character cases were found")

    manifest = {
        "schema_version": 1,
        "name": f"stsai-{build_id}-a{ascension}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_after_unix": int(datetime.now(timezone.utc).timestamp()),
        "target": {"build_id": build_id, "ascension": ascension},
        "selection": {
            "method": "sha256-ranked-without-outcome",
            "count_per_character": count_per_character,
            "source_build_id": source_build_id,
        },
        "cases": cases,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read benchmark manifest {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise ValueError("Benchmark manifest requires a cases list")
    if not isinstance(value.get("target"), dict):
        raise ValueError("Benchmark manifest requires a target object")
    return value


def _metrics(results: Sequence[Mapping[str, Any]], total_cases: int) -> dict[str, Any]:
    completed = len(results)
    wins = sum(int(bool(result.get("win"))) for result in results)
    floors = [int(result.get("floors_reached") or 0) for result in results]
    return {
        "cases": total_cases,
        "completed": completed,
        "missing": total_cases - completed,
        "wins": wins,
        "win_rate_completed": round(wins / completed, 6) if completed else 0.0,
        "average_floor": round(mean(floors), 3) if floors else 0.0,
    }


def evaluate_benchmark(
    manifest_path: str | os.PathLike[str], sources: Sequence[str | os.PathLike[str]]
) -> dict[str, Any]:
    """Evaluate the first post-manifest attempt for each seed/character case."""

    manifest = load_manifest(manifest_path)
    target = manifest["target"]
    build_id = str(target.get("build_id"))
    ascension = int(target.get("ascension") or 0)
    started_after = int(manifest.get("started_after_unix") or 0)
    cases = {
        (str(case["seed"]), str(case["character"]))
        for case in manifest["cases"]
        if isinstance(case, dict) and case.get("seed") and case.get("character")
    }
    attempts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        for run in read_runs(source):
            if str(run.get("build_id")) != build_id:
                continue
            if int(run.get("ascension") or 0) != ascension:
                continue
            if int(run.get("start_time") or 0) < started_after:
                continue
            character = _character(run)
            key = (str(run.get("seed", "")), str(character or ""))
            if key not in cases:
                continue
            attempts[key].append(
                {
                    "seed": key[0],
                    "character": key[1],
                    "start_time": int(run.get("start_time") or 0),
                    "win": bool(run.get("win", False)),
                    "abandoned": bool(run.get("was_abandoned", False)),
                    "floors_reached": sum(1 for _ in iter_map_points(run)),
                    "run_time": int(run.get("run_time") or 0),
                }
            )

    first_attempts: list[dict[str, Any]] = []
    retry_count = 0
    for key in sorted(cases):
        ordered = sorted(attempts.get(key, []), key=lambda value: value["start_time"])
        if ordered:
            first_attempts.append(ordered[0])
            retry_count += max(len(ordered) - 1, 0)

    by_character: dict[str, Any] = {}
    characters = sorted(character for _, character in cases)
    for character in sorted(set(characters)):
        character_results = [
            result for result in first_attempts if result["character"] == character
        ]
        character_cases = sum(1 for _, value in cases if value == character)
        by_character[character] = _metrics(character_results, character_cases)

    completed_keys = {(result["seed"], result["character"]) for result in first_attempts}
    missing = [
        {"seed": seed, "character": character}
        for seed, character in sorted(cases - completed_keys)
    ]
    return {
        "benchmark": manifest.get("name"),
        "target": target,
        "overall": _metrics(first_attempts, len(cases)),
        "by_character": by_character,
        "retry_attempts_ignored": retry_count,
        "missing_cases": missing,
        "first_attempt_results": first_attempts,
    }

