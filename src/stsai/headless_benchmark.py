"""Paired fixed-seed evaluation for native headless policies."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
from math import sqrt
from statistics import fmean, median, stdev
from typing import Any, Callable, Mapping, Sequence

from stsai.headless import HeadlessConfig, HeadlessRuntimeError
from stsai.headless_batch import HeadlessBatchJob, HeadlessBatchManager
from stsai.headless_env import HEADLESS_CHARACTERS, run_random_rollout
from stsai.headless_policy import (
    run_combat_distilled_rollout,
    run_heuristic_rollout,
    run_hybrid_rollout,
)
from stsai.ids import canonical_character


_CHARACTER_CODES = {
    "Defect": "D",
    "Ironclad": "I",
    "Necrobinder": "N",
    "Regent": "R",
    "Silent": "S",
}


def _character_name(value: Any) -> str:
    canonical = canonical_character(value)
    name = canonical.rsplit(".", 1)[-1].title()
    if name not in HEADLESS_CHARACTERS:
        raise ValueError(f"Unsupported benchmark character: {value}")
    return name


def select_benchmark_cases(
    manifest: Mapping[str, Any], count_per_character: int
) -> list[dict[str, str]]:
    """Take the manifest's first N precommitted cases for each character."""

    if count_per_character <= 0:
        raise ValueError("count_per_character must be positive")
    selected: list[dict[str, str]] = []
    counts: dict[str, int] = defaultdict(int)
    for raw in manifest.get("cases") or []:
        if not isinstance(raw, Mapping) or not raw.get("seed"):
            continue
        character = _character_name(raw.get("character"))
        if counts[character] >= count_per_character:
            continue
        selected.append({"character": character, "seed": str(raw["seed"])})
        counts[character] += 1
    missing = [
        character for character in HEADLESS_CHARACTERS if counts[character] < count_per_character
    ]
    if missing:
        raise ValueError("Manifest has too few cases for: " + ", ".join(sorted(missing)))
    return selected


def select_benchmark_pool_cases(
    manifest: Mapping[str, Any], pool: str, count_per_character: int
) -> list[dict[str, str]]:
    """Take only a named precommitted manifest pool, preserving its order."""

    filtered = {
        "cases": [
            case
            for case in manifest.get("cases") or []
            if isinstance(case, Mapping) and case.get("pool") == pool
        ]
    }
    return select_benchmark_cases(filtered, count_per_character)


def _policy_seed(base_seed: int, character: str, game_seed: str) -> int:
    payload = f"{base_seed}|{character}|{game_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _summary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    random_floors = [float(row["random"]["floor"] or 0) for row in rows]
    hybrid_floors = [float(row["hybrid"]["floor"] or 0) for row in rows]
    deltas = [hybrid - random for random, hybrid in zip(random_floors, hybrid_floors)]
    result: dict[str, Any] = {
        "cases": len(rows),
        "random": {
            "mean_floor": round(fmean(random_floors), 4) if rows else 0.0,
            "victories": sum(row["random"].get("victory") is True for row in rows),
            "truncations": sum(row["random"].get("truncated") is True for row in rows),
            "mean_steps": round(fmean(float(row["random"]["steps"]) for row in rows), 4)
            if rows
            else 0.0,
        },
        "hybrid": {
            "mean_floor": round(fmean(hybrid_floors), 4) if rows else 0.0,
            "victories": sum(row["hybrid"].get("victory") is True for row in rows),
            "truncations": sum(row["hybrid"].get("truncated") is True for row in rows),
            "mean_steps": round(fmean(float(row["hybrid"]["steps"]) for row in rows), 4)
            if rows
            else 0.0,
        },
        "paired_floor_delta": round(fmean(deltas), 4) if rows else 0.0,
        "improved": sum(delta > 0 for delta in deltas),
        "tied": sum(delta == 0 for delta in deltas),
        "worse": sum(delta < 0 for delta in deltas),
    }
    if rows and all("heuristic" in row for row in rows):
        heuristic_floors = [float(row["heuristic"]["floor"] or 0) for row in rows]
        heuristic_vs_random = [
            heuristic - random for random, heuristic in zip(random_floors, heuristic_floors)
        ]
        hybrid_vs_heuristic = [
            hybrid - heuristic for heuristic, hybrid in zip(heuristic_floors, hybrid_floors)
        ]
        result["heuristic"] = {
            "mean_floor": round(fmean(heuristic_floors), 4),
            "victories": sum(row["heuristic"].get("victory") is True for row in rows),
            "truncations": sum(row["heuristic"].get("truncated") is True for row in rows),
            "mean_steps": round(fmean(float(row["heuristic"]["steps"]) for row in rows), 4),
        }
        result["heuristic_vs_random"] = {
            "paired_floor_delta": round(fmean(heuristic_vs_random), 4),
            "improved": sum(delta > 0 for delta in heuristic_vs_random),
            "tied": sum(delta == 0 for delta in heuristic_vs_random),
            "worse": sum(delta < 0 for delta in heuristic_vs_random),
        }
        result["hybrid_vs_heuristic"] = {
            "paired_floor_delta": round(fmean(hybrid_vs_heuristic), 4),
            "improved": sum(delta > 0 for delta in hybrid_vs_heuristic),
            "tied": sum(delta == 0 for delta in hybrid_vs_heuristic),
            "worse": sum(delta < 0 for delta in hybrid_vs_heuristic),
        }
    return result


def aggregate_paired_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["character"])].append(row)
    return {
        "overall": _summary_metrics(rows),
        "by_character": {
            character: _summary_metrics(grouped.get(character, []))
            for character in HEADLESS_CHARACTERS
        },
    }


def _policy_metrics(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [row[key] for row in rows]
    return {
        "mean_floor": round(fmean(float(value.get("floor") or 0) for value in values), 4)
        if values
        else 0.0,
        "victories": sum(value.get("victory") is True for value in values),
        "truncations": sum(value.get("truncated") is True for value in values),
        "runtime_failures": sum(value.get("runtime_error") is not None for value in values),
        "mean_steps": round(fmean(float(value["steps"]) for value in values), 4) if values else 0.0,
    }


def _combat_distillation_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    deltas = [
        float(row["combat_bc"].get("floor") or 0) - float(row["heuristic"].get("floor") or 0)
        for row in rows
    ]
    mean_delta = fmean(deltas) if deltas else 0.0
    if len(deltas) > 1:
        margin = 1.96 * stdev(deltas) / sqrt(len(deltas))
        interval = [round(mean_delta - margin, 4), round(mean_delta + margin, 4)]
    else:
        interval = [round(mean_delta, 4), round(mean_delta, 4)]
    return {
        "cases": len(rows),
        "heuristic": _policy_metrics(rows, "heuristic"),
        "combat_bc": _policy_metrics(rows, "combat_bc"),
        "paired_floor_delta": round(mean_delta, 4),
        "descriptive_paired_95_interval": interval,
        "improved": sum(delta > 0 for delta in deltas),
        "tied": sum(delta == 0 for delta in deltas),
        "worse": sum(delta < 0 for delta in deltas),
    }


def aggregate_combat_distillation(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["character"])].append(row)
    return {
        "overall": _combat_distillation_summary(rows),
        "by_character": {
            character: _combat_distillation_summary(grouped.get(character, []))
            for character in HEADLESS_CHARACTERS
        },
    }


def _combat_model_pair_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    deltas = [
        float(row["candidate"].get("floor") or 0) - float(row["baseline"].get("floor") or 0)
        for row in rows
    ]
    mean_delta = fmean(deltas) if deltas else 0.0
    if len(deltas) > 1:
        margin = 1.96 * stdev(deltas) / sqrt(len(deltas))
        interval = [round(mean_delta - margin, 4), round(mean_delta + margin, 4)]
    else:
        interval = [round(mean_delta, 4), round(mean_delta, 4)]
    return {
        "cases": len(rows),
        "baseline": _policy_metrics(rows, "baseline"),
        "candidate": _policy_metrics(rows, "candidate"),
        "paired_floor_delta": round(mean_delta, 4),
        "descriptive_paired_95_interval": interval,
        "improved": sum(delta > 0 for delta in deltas),
        "tied": sum(delta == 0 for delta in deltas),
        "worse": sum(delta < 0 for delta in deltas),
    }


def aggregate_combat_model_pair(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["character"])].append(row)
    return {
        "overall": _combat_model_pair_summary(rows),
        "by_character": {
            character: _combat_model_pair_summary(grouped.get(character, []))
            for character in HEADLESS_CHARACTERS
        },
    }


def _combat_policy_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values = [row["policy"] for row in rows]
    floors = [float(value.get("floor") or 0) for value in values]
    victories = sum(value.get("victory") is True for value in values)
    return {
        "cases": len(rows),
        "victories": victories,
        "win_rate": round(victories / len(rows), 4) if rows else 0.0,
        "mean_floor": round(fmean(floors), 4) if floors else 0.0,
        "median_floor": round(float(median(floors)), 4) if floors else 0.0,
        "min_floor": int(min(floors)) if floors else 0,
        "max_floor": int(max(floors)) if floors else 0,
        "truncations": sum(value.get("truncated") is True for value in values),
        "runtime_failures": sum(value.get("runtime_error") is not None for value in values),
        "mean_steps": round(fmean(float(value.get("steps") or 0) for value in values), 4)
        if values
        else 0.0,
    }


def aggregate_combat_policy_baseline(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize one frozen policy overall and separately for every character."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["character"])].append(row)
    return {
        "overall": _combat_policy_summary(rows),
        "by_character": {
            character: _combat_policy_summary(grouped.get(character, []))
            for character in HEADLESS_CHARACTERS
        },
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded_seed_receipts(
    manifests: Sequence[str | Path],
) -> tuple[set[str], list[dict[str, Any]]]:
    seeds: set[str] = set()
    receipts: list[dict[str, Any]] = []
    for raw_path in manifests:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        manifest_seeds = {
            str(case["seed"])
            for case in payload.get("cases") or []
            if isinstance(case, Mapping) and case.get("seed")
        }
        seeds.update(manifest_seeds)
        receipts.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "seed_count": len(manifest_seeds),
            }
        )
    return seeds, receipts


def create_combat_policy_baseline_manifest(
    output: str | Path,
    *,
    macro_model_path: str | Path,
    combat_model_path: str | Path,
    combat_kind: str = "set",
    seed_prefix: str = "BASEV1",
    count_per_character: int = 10,
    excluded_manifests: Sequence[str | Path] = (),
    build_id: str = "v0.107.1",
    ascension: int = 10,
) -> dict[str, Any]:
    """Freeze a one-shot, character-balanced online policy measurement."""

    if combat_kind not in {"mlp", "set"}:
        raise ValueError("Combat model kind must be mlp or set")
    if not seed_prefix or count_per_character <= 0:
        raise ValueError("A seed prefix and positive per-character count are required")
    if not 0 <= ascension <= 10:
        raise ValueError("Ascension must be between zero and ten")
    cases = [
        {
            "character": canonical_character(character),
            "seed": f"{seed_prefix}{_CHARACTER_CODES[character]}{index:06d}",
        }
        for character in HEADLESS_CHARACTERS
        for index in range(count_per_character)
    ]
    case_seeds = {str(case["seed"]) for case in cases}
    if len(case_seeds) != len(cases):
        raise ValueError("Generated baseline seeds are not unique")
    excluded_seeds, exclusion_receipts = _excluded_seed_receipts(excluded_manifests)
    overlap = sorted(case_seeds & excluded_seeds)
    if overlap:
        raise ValueError("Fresh baseline seeds overlap excluded manifests: " + ", ".join(overlap))
    payload = {
        "schema_version": 1,
        "name": "stsai-frozen-online-combat-policy-baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {"build_id": build_id, "ascension": ascension},
        "models": {
            "macro": {
                "sha256": _sha256(macro_model_path),
                "filename": Path(macro_model_path).name,
            },
            "combat": {
                "kind": combat_kind,
                "sha256": _sha256(combat_model_path),
                "filename": Path(combat_model_path).name,
            },
        },
        "policy_constraints": {
            "safe_potion_ids": ["POTION.ATTACK_POTION"],
        },
        "selection": {
            "method": "fresh-prefix-seeds-selected-before-outcomes",
            "seed_prefix": seed_prefix,
            "count_per_character": count_per_character,
            "outcomes_inspected": False,
            "excluded_manifests": exclusion_receipts,
            "excluded_seed_count": len(excluded_seeds),
            "overlap_count": 0,
        },
        "evaluation_precommit": {
            "purpose": "measurement-only baseline; no policy promotion or tuning",
            "primary_metric": "per-character victory count and win rate",
            "secondary_metrics": [
                "overall victory count and win rate",
                "mean and median final floor",
                "truncation and runtime-failure counts",
                "mean steps",
            ],
            "execution": "one first attempt per frozen case; completed exact resumes only",
            "stop_rule": "open the result only after all frozen cases finish",
        },
        "cases": cases,
    }
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Frozen baseline manifest already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return payload


def create_chip_aware_heuristic_manifest(
    output: str | Path,
    *,
    macro_model_path: str | Path,
    seed_prefix: str = "TACTV1",
    count_per_character: int = 10,
    excluded_manifests: Sequence[str | Path] = (),
    build_id: str = "v0.107.1",
    ascension: int = 10,
) -> dict[str, Any]:
    """Freeze a paired v1-versus-chip-aware-v2 tactical teacher test."""

    if not seed_prefix or count_per_character <= 0:
        raise ValueError("A seed prefix and positive per-character count are required")
    if not 0 <= ascension <= 10:
        raise ValueError("Ascension must be between zero and ten")
    cases = [
        {
            "character": canonical_character(character),
            "seed": f"{seed_prefix}{_CHARACTER_CODES[character]}{index:06d}",
        }
        for character in HEADLESS_CHARACTERS
        for index in range(count_per_character)
    ]
    case_seeds = {str(case["seed"]) for case in cases}
    excluded_seeds, exclusion_receipts = _excluded_seed_receipts(excluded_manifests)
    overlap = sorted(case_seeds & excluded_seeds)
    if len(case_seeds) != len(cases) or overlap:
        raise ValueError("Tactical teacher seeds are not fresh and unique")
    payload = {
        "schema_version": 1,
        "name": "stsai-frozen-chip-aware-heuristic-pair",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {"build_id": build_id, "ascension": ascension},
        "model": {
            "filename": Path(macro_model_path).name,
            "sha256": _sha256(macro_model_path),
            "usage": "interface-only; learned macro scoring disabled",
        },
        "arms": {
            "baseline": "greedy_rules_v1",
            "candidate": "greedy_rules_v2_chip_aware",
        },
        "policy_constraints": {
            "safe_potion_ids": ["POTION.ATTACK_POTION"],
            "all_noncombat_rules_unchanged": True,
            "candidate_change": (
                "raise block-card priority by preventable incoming damage and "
                "give a lethal-damage prevention bonus"
            ),
        },
        "selection": {
            "method": "fresh-prefix-seeds-selected-before-outcomes",
            "seed_prefix": seed_prefix,
            "count_per_character": count_per_character,
            "outcomes_inspected": False,
            "excluded_manifests": exclusion_receipts,
            "excluded_seed_count": len(excluded_seeds),
            "overlap_count": 0,
        },
        "promotion_gate": {
            "overall_mean_paired_floor_delta_at_least": 0.75,
            "every_character_mean_paired_floor_delta_at_least": -0.5,
            "candidate_victories_at_least_baseline": True,
            "candidate_truncations_at_most_baseline": True,
            "if_failed": "stop this tactical rule candidate; do not tune on these cases",
        },
        "cases": cases,
    }
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Frozen tactical manifest already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return payload


def create_combat_model_pair_manifest(
    output: str | Path,
    *,
    baseline_model_path: str | Path,
    candidate_model_path: str | Path,
    baseline_kind: str = "mlp",
    candidate_kind: str = "set",
    seed_prefix: str = "SETV1",
    count_per_character: int = 5,
    build_id: str = "v0.107.1",
    ascension: int = 10,
) -> dict[str, Any]:
    """Bind two frozen checkpoints before creating fresh outcome-blind seeds."""

    if baseline_kind not in {"mlp", "set"} or candidate_kind not in {"mlp", "set"}:
        raise ValueError("Combat model kind must be mlp or set")
    if not seed_prefix or count_per_character <= 0:
        raise ValueError("A seed prefix and positive per-character count are required")
    if not 0 <= ascension <= 10:
        raise ValueError("Ascension must be between zero and ten")
    cases = [
        {
            "character": canonical_character(character),
            "seed": f"{seed_prefix}{_CHARACTER_CODES[character]}{index:06d}",
        }
        for character in HEADLESS_CHARACTERS
        for index in range(count_per_character)
    ]
    payload = {
        "schema_version": 1,
        "name": "stsai-frozen-combat-checkpoint-pair",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {"build_id": build_id, "ascension": ascension},
        "models": {
            "baseline": {
                "kind": baseline_kind,
                "sha256": _sha256(baseline_model_path),
                "filename": Path(baseline_model_path).name,
            },
            "candidate": {
                "kind": candidate_kind,
                "sha256": _sha256(candidate_model_path),
                "filename": Path(candidate_model_path).name,
            },
        },
        "selection": {
            "method": "fresh-prefix-seeds-selected-before-outcomes",
            "seed_prefix": seed_prefix,
            "count_per_character": count_per_character,
            "outcomes_inspected": False,
        },
        "cases": cases,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return payload


def _combat_policy_name(kind: str) -> str:
    if kind == "mlp":
        return "combat_bc_greedy_noncombat_v1"
    if kind == "set":
        return "combat_deepsets_greedy_noncombat_v1"
    raise ValueError(f"Unsupported combat checkpoint kind: {kind}")


def _run_combat_checkpoint(
    kind: str,
    *,
    macro_model_path: str | Path,
    combat_model_path: str | Path,
    output: Path,
    character: str,
    ascension: int,
    seed: str,
    max_steps: int,
    build_id: str,
    config: HeadlessConfig | None,
) -> dict[str, Any]:
    if kind == "mlp":
        return run_combat_distilled_rollout(
            macro_model_path=macro_model_path,
            combat_model_path=combat_model_path,
            output=output,
            character=character,
            ascension=ascension,
            seed=seed,
            max_steps=max_steps,
            build_id=build_id,
            config=config,
        )
    if kind == "set":
        from stsai.set_policy import run_set_combat_rollout

        return run_set_combat_rollout(
            macro_model_path=macro_model_path,
            combat_model_path=combat_model_path,
            output=output,
            character=character,
            ascension=ascension,
            seed=seed,
            max_steps=max_steps,
            build_id=build_id,
            config=config,
            safe_potion_ids=("POTION.ATTACK_POTION",),
        )
    raise ValueError(f"Unsupported combat checkpoint kind: {kind}")


def _combat_action_counts(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, Mapping) or event.get("event_type") != "transition":
                continue
            observation = event.get("observation")
            observation = observation if isinstance(observation, Mapping) else {}
            if observation.get("decision_type") != "combat_play":
                continue
            action = event.get("action")
            action = action if isinstance(action, Mapping) else {}
            counts[str(action.get("action", "unknown"))] += 1
    return dict(counts)


def _pair_combat_action_counts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for arm in ("baseline", "candidate"):
            counts: Counter[str] = Counter()
            for row in selected:
                counts.update(row[arm].get("combat_action_counts") or {})
            result[arm] = dict(counts)
        return result

    return {
        "overall": summarize(rows),
        "by_character": {
            character: summarize([row for row in rows if row.get("character") == character])
            for character in HEADLESS_CHARACTERS
        },
    }


def _completed_rollout(
    path: Path,
    *,
    character: str,
    seed: str,
    ascension: int,
    policy: str,
    build_id: str | None,
    combat_model_sha256: str | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
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
    metadata = first.get("policy_metadata") if first else None
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if (
        first is None
        or last is None
        or first.get("event_type") != "episode_start"
        or last.get("event_type") != "episode_end"
        or canonical_character(first.get("character")) != canonical_character(character)
        or str(first.get("seed")) != seed
        or int(first.get("ascension", -1)) != ascension
        or str(first.get("policy")) != policy
        or (build_id is not None and str(metadata.get("build_id")) != build_id)
        or (
            combat_model_sha256 is not None
            and str(metadata.get("combat_model_sha256")) != combat_model_sha256
        )
        or any(
            str(metadata.get(key)) != str(value)
            for key, value in (expected_metadata or {}).items()
        )
    ):
        return None
    return {
        **{
            key: last.get(key)
            for key in (
                "steps",
                "total_reward",
                "terminated",
                "truncated",
                "victory",
                "act",
                "floor",
                "policy_sources",
                "runtime_error",
            )
        },
        "trajectory": str(path.resolve()),
        "trajectory_id": last.get("trajectory_id"),
    }


def _discard_matching_incomplete(
    path: Path,
    *,
    character: str,
    seed: str,
    policy: str,
) -> bool:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not temporary.is_file():
        return False
    with temporary.open("r", encoding="utf-8-sig") as handle:
        first_line = handle.readline()
    try:
        first = json.loads(first_line)
    except json.JSONDecodeError:
        # A crash before the episode_start header finished writing leaves an
        # empty or truncated first line. Nothing identifiable survives, so the
        # file cannot belong to a different case; remove it so reruns resume.
        temporary.unlink()
        return True
    if (
        not isinstance(first, Mapping)
        or first.get("event_type") != "episode_start"
        or canonical_character(first.get("character")) != canonical_character(character)
        or str(first.get("seed")) != seed
        or str(first.get("policy")) != policy
    ):
        raise ValueError(f"Refusing to remove mismatched incomplete trajectory: {temporary}")
    temporary.unlink()
    return True


def _publish_runtime_failure(path: Path, error: BaseException) -> dict[str, Any]:
    """Turn an atomic partial trajectory into an explicit benchmark failure."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    if not temporary.is_file():
        raise error
    first: Mapping[str, Any] | None = None
    last_transition: Mapping[str, Any] | None = None
    total_reward = 0.0
    policy_sources: dict[str, int] = defaultdict(int)
    with temporary.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, Mapping):
                raise ValueError(f"Invalid partial benchmark trajectory: {temporary}")
            if first is None:
                first = event
            if event.get("event_type") == "transition":
                last_transition = event
                total_reward += float(event.get("reward") or 0.0)
                info = event.get("info")
                info = info if isinstance(info, Mapping) else {}
                policy = info.get("policy")
                policy = policy if isinstance(policy, Mapping) else {}
                source = str(policy.get("source", first.get("policy", "unknown")))
                policy_sources[source] += 1
    if first is None or first.get("event_type") != "episode_start":
        raise ValueError(f"Partial benchmark trajectory has no episode start: {temporary}")
    observation = last_transition.get("next_observation") if last_transition is not None else None
    observation = observation if isinstance(observation, Mapping) else {}
    context = observation.get("context")
    context = context if isinstance(context, Mapping) else {}
    message = str(error).split("\nRecent runtime logs:", 1)[0]
    runtime_error = {"type": type(error).__name__, "message": message}
    summary = {
        "steps": int(last_transition.get("step") or 0) if last_transition else 0,
        "total_reward": total_reward,
        "terminated": False,
        "truncated": True,
        "victory": None,
        "act": context.get("act"),
        "floor": context.get("floor"),
        "policy_sources": dict(policy_sources),
        "runtime_error": runtime_error,
    }
    episode_end = {
        "schema_version": 1,
        "event_type": "episode_end",
        "trajectory_id": first.get("trajectory_id"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        **summary,
    }
    with temporary.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(episode_end, separators=(",", ":"), ensure_ascii=False))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return {
        **summary,
        "trajectory": str(path.resolve()),
        "trajectory_id": first.get("trajectory_id"),
    }


def _run_benchmark_rollout(path: Path, run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return run()
    except (HeadlessRuntimeError, TimeoutError) as exc:
        return _publish_runtime_failure(path, exc)


def evaluate_hybrid_vs_random(
    *,
    manifest_path: str | Path,
    model_path: str | Path,
    output: str | Path,
    trajectory_dir: str | Path,
    count_per_character: int = 5,
    random_policy_seed: int = 7,
    max_steps: int = 1_000,
    build_id: str = "v0.107.1",
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Evaluate both policies once on each precommitted character/seed pair."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    target = manifest.get("target") or {}
    if target.get("build_id") and str(target["build_id"]) != build_id:
        raise ValueError(f"Manifest targets {target['build_id']}; evaluator targets {build_id}")
    ascension = int(target.get("ascension", 10))
    cases = select_benchmark_cases(manifest, count_per_character)
    trajectory_dir = Path(trajectory_dir)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    reused = 0
    discarded_incomplete = 0
    for case_index, case in enumerate(cases):
        character = case["character"]
        seed = case["seed"]
        stem = f"{case_index:03d}.{character.lower()}.{seed}"
        derived_policy_seed = _policy_seed(random_policy_seed, character, seed)
        arms: tuple[
            tuple[str, str, str | None, dict[str, Any] | None, Callable[[Path], dict[str, Any]]],
            ...,
        ] = (
            (
                "random",
                "legal_random",
                None,
                {"policy_seed": derived_policy_seed},
                lambda path: run_random_rollout(
                    output=path,
                    character=character,
                    ascension=ascension,
                    seed=seed,
                    policy_seed=derived_policy_seed,
                    max_steps=max_steps,
                    config=config,
                ),
            ),
            (
                "heuristic",
                "greedy_rules_v1",
                build_id,
                None,
                lambda path: run_heuristic_rollout(
                    model_path=model_path,
                    output=path,
                    character=character,
                    ascension=ascension,
                    seed=seed,
                    max_steps=max_steps,
                    build_id=build_id,
                    config=config,
                ),
            ),
            (
                "hybrid",
                "hybrid_macro_bc_greedy_v1",
                build_id,
                None,
                lambda path: run_hybrid_rollout(
                    model_path=model_path,
                    output=path,
                    character=character,
                    ascension=ascension,
                    seed=seed,
                    max_steps=max_steps,
                    build_id=build_id,
                    config=config,
                ),
            ),
        )
        arm_results: dict[str, dict[str, Any]] = {}
        for arm_name, policy_name, arm_build_id, extra_metadata, run in arms:
            path = trajectory_dir / f"{stem}.{arm_name}.jsonl"
            result = _completed_rollout(
                path,
                character=character,
                seed=seed,
                ascension=ascension,
                policy=policy_name,
                build_id=arm_build_id,
                expected_metadata=extra_metadata,
            )
            if result is None:
                discarded_incomplete += int(
                    _discard_matching_incomplete(
                        path, character=character, seed=seed, policy=policy_name
                    )
                )
                result = _run_benchmark_rollout(
                    path, lambda run=run, path=path: run(path)
                )
            else:
                reused += 1
            arm_results[arm_name] = result
        random_result = arm_results["random"]
        heuristic_result = arm_results["heuristic"]
        hybrid_result = arm_results["hybrid"]
        rows.append(
            {
                "case_index": case_index,
                "character": character,
                "canonical_character": canonical_character(character),
                "seed": seed,
                "ascension": ascension,
                "random_policy_seed": derived_policy_seed,
                "random": random_result,
                "heuristic": heuristic_result,
                "hybrid": hybrid_result,
                "floor_delta": int(hybrid_result.get("floor") or 0)
                - int(random_result.get("floor") or 0),
            }
        )

    result = {
        "schema_version": 1,
        "evaluation": "paired_headless_random_heuristic_hybrid",
        "manifest": str(manifest_path.resolve()),
        "manifest_name": manifest.get("name"),
        "build_id": build_id,
        "ascension": ascension,
        "count_per_character": count_per_character,
        "random_policy_seed": random_policy_seed,
        "max_steps": max_steps,
        "model": str(Path(model_path).resolve()),
        "selection": "first precommitted manifest cases per character",
        "reused_completed_runs": reused,
        "discarded_incomplete": discarded_incomplete,
        "aggregate": aggregate_paired_results(rows),
        "cases": rows,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return result


def evaluate_combat_distillation(
    *,
    manifest_path: str | Path,
    macro_model_path: str | Path,
    combat_model_path: str | Path,
    output: str | Path,
    trajectory_dir: str | Path,
    count_per_character: int = 5,
    parallel_workers: int = 1,
    max_steps: int = 1_000,
    build_id: str = "v0.107.1",
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Evaluate the heuristic teacher and its combat distillation exactly once."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    target = manifest.get("target") or {}
    if target.get("build_id") and str(target["build_id"]) != build_id:
        raise ValueError(f"Manifest targets {target['build_id']}; evaluator targets {build_id}")
    ascension = int(target.get("ascension", 10))
    cases = select_benchmark_cases(manifest, count_per_character)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Frozen evaluation already exists: {destination}")
    trajectories = Path(trajectory_dir)
    trajectories.mkdir(parents=True, exist_ok=True)
    expected_names = {
        name
        for case_index, case in enumerate(cases)
        for name in (
            f"{case_index:03d}.{case['character'].lower()}.{case['seed']}.heuristic.jsonl",
            f"{case_index:03d}.{case['character'].lower()}.{case['seed']}.heuristic.jsonl.tmp",
            f"{case_index:03d}.{case['character'].lower()}.{case['seed']}.combat-bc.jsonl",
            f"{case_index:03d}.{case['character'].lower()}.{case['seed']}.combat-bc.jsonl.tmp",
        )
    }
    unexpected = [path.name for path in trajectories.iterdir() if path.name not in expected_names]
    if unexpected:
        raise ValueError(
            "Frozen evaluation trajectory directory has unexpected files: "
            + ", ".join(sorted(unexpected))
        )
    reused = {"heuristic": 0, "combat_bc": 0}
    discarded_incomplete = 0
    combat_sha256 = _sha256(combat_model_path)
    arm_results: dict[str, dict[str, Any]] = {}
    jobs: list[HeadlessBatchJob[dict[str, Any]]] = []
    for case_index, case in enumerate(cases):
        character = case["character"]
        seed = case["seed"]
        stem = f"{case_index:03d}.{character.lower()}.{seed}"
        for arm, policy_name, combat_sha in (
            ("heuristic", "greedy_rules_v1", None),
            ("combat_bc", "combat_bc_greedy_noncombat_v1", combat_sha256),
        ):
            suffix = "heuristic" if arm == "heuristic" else "combat-bc"
            path = trajectories / f"{stem}.{suffix}.jsonl"
            key = f"{case_index}:{arm}"
            completed = _completed_rollout(
                path,
                character=character,
                seed=seed,
                ascension=ascension,
                policy=policy_name,
                build_id=build_id,
                combat_model_sha256=combat_sha,
            )
            if completed is not None:
                reused[arm] += 1
                arm_results[key] = completed
                continue
            discarded_incomplete += int(
                _discard_matching_incomplete(
                    path,
                    character=character,
                    seed=seed,
                    policy=policy_name,
                )
            )
            if arm == "heuristic":
                run = partial(
                    _run_benchmark_rollout,
                    path,
                    partial(
                        run_heuristic_rollout,
                        model_path=macro_model_path,
                        output=path,
                        character=character,
                        ascension=ascension,
                        seed=seed,
                        max_steps=max_steps,
                        build_id=build_id,
                        config=config,
                    ),
                )
            else:
                run = partial(
                    _run_benchmark_rollout,
                    path,
                    partial(
                        run_combat_distilled_rollout,
                        macro_model_path=macro_model_path,
                        combat_model_path=combat_model_path,
                        output=path,
                        character=character,
                        ascension=ascension,
                        seed=seed,
                        max_steps=max_steps,
                        build_id=build_id,
                        config=config,
                    ),
                )
            jobs.append(HeadlessBatchJob(key, run))

    batch_report = HeadlessBatchManager(parallel_workers).run(jobs)
    for job, value in zip(jobs, batch_report.values(), strict=True):
        arm_results[job.key] = value

    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        character = case["character"]
        seed = case["seed"]
        heuristic = arm_results[f"{case_index}:heuristic"]
        combat_bc = arm_results[f"{case_index}:combat_bc"]
        rows.append(
            {
                "case_index": case_index,
                "character": character,
                "canonical_character": canonical_character(character),
                "seed": seed,
                "ascension": ascension,
                "heuristic": heuristic,
                "combat_bc": combat_bc,
                "paired_floor_delta": int(combat_bc.get("floor") or 0)
                - int(heuristic.get("floor") or 0),
            }
        )

    result = {
        "schema_version": 1,
        "evaluation": "frozen_paired_headless_heuristic_vs_combat_distillation",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_name": manifest.get("name"),
        "build_id": build_id,
        "ascension": ascension,
        "count_per_character": count_per_character,
        "max_steps": max_steps,
        "execution": {
            "mode": "bounded_native_process_batch",
            "parallel_workers": parallel_workers,
            "submitted_jobs": len(jobs),
        },
        "selection": "all precommitted held cases in manifest order; outcomes not inspected",
        "macro_model": str(Path(macro_model_path).resolve()),
        "macro_model_sha256": _sha256(macro_model_path),
        "combat_model": str(Path(combat_model_path).resolve()),
        "combat_model_sha256": _sha256(combat_model_path),
        "label_source": "deterministic_heuristic_distillation",
        "resume": {
            "reused_completed_trajectories": reused,
            "discarded_matching_incomplete_trajectories": discarded_incomplete,
        },
        "aggregate": aggregate_combat_distillation(rows),
        "cases": rows,
        "limitations": [
            (
                "This compares imitation against its heuristic teacher; "
                "it does not establish expert play."
            ),
            (
                f"{count_per_character} held seeds per character require interpreting the "
                "reported paired interval, not only the point estimate."
            ),
            "The descriptive interval is a normal approximation over paired final-floor deltas.",
            (
                "Native runtime errors are retained as truncations at the last recorded floor "
                "and counted in the aggregate."
            ),
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return result


def evaluate_chip_aware_heuristic_pair(
    *,
    manifest_path: str | Path,
    macro_model_path: str | Path,
    output: str | Path,
    trajectory_dir: str | Path,
    max_steps: int = 1_000,
    build_id: str = "v0.107.1",
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Run the frozen original-versus-chip-aware heuristic comparison once."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("name") != "stsai-frozen-chip-aware-heuristic-pair":
        raise ValueError("Manifest is not a frozen chip-aware heuristic pair")
    target = manifest.get("target") or {}
    if str(target.get("build_id")) != build_id:
        raise ValueError("Tactical manifest and evaluator build ids differ")
    ascension = int(target.get("ascension", 10))
    selection = manifest.get("selection") or {}
    count_per_character = int(selection.get("count_per_character") or 0)
    cases = select_benchmark_cases(manifest, count_per_character)
    if len(cases) != len(manifest.get("cases") or []):
        raise ValueError("Frozen tactical manifest contains unexpected extra cases")
    frozen_model = manifest.get("model")
    frozen_model = frozen_model if isinstance(frozen_model, Mapping) else {}
    macro_sha256 = _sha256(macro_model_path)
    if str(frozen_model.get("sha256")) != macro_sha256:
        raise ValueError("Macro interface checkpoint does not match tactical manifest")
    constraints = manifest.get("policy_constraints")
    constraints = constraints if isinstance(constraints, Mapping) else {}
    if constraints.get("safe_potion_ids") != ["POTION.ATTACK_POTION"]:
        raise ValueError("Tactical manifest does not bind the potion whitelist")

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Frozen tactical evaluation already exists: {destination}")
    trajectories = Path(trajectory_dir)
    trajectories.mkdir(parents=True, exist_ok=True)
    expected_names = {
        name
        for case_index, case in enumerate(cases)
        for arm in ("baseline", "candidate")
        for name in (
            f"{case_index:03d}.{case['character'].lower()}.{case['seed']}.{arm}.jsonl",
            f"{case_index:03d}.{case['character'].lower()}.{case['seed']}.{arm}.jsonl.tmp",
        )
    }
    unexpected = [path.name for path in trajectories.iterdir() if path.name not in expected_names]
    if unexpected:
        raise ValueError(
            "Frozen tactical trajectory directory has unexpected files: "
            + ", ".join(sorted(unexpected))
        )

    rows: list[dict[str, Any]] = []
    reused = {"baseline": 0, "candidate": 0}
    discarded_incomplete = 0
    for case_index, case in enumerate(cases):
        character = case["character"]
        seed = case["seed"]
        stem = f"{case_index:03d}.{character.lower()}.{seed}"
        arm_results: dict[str, dict[str, Any]] = {}
        for arm, heuristic_version in (
            ("baseline", "v1"),
            ("candidate", "v2_chip_aware"),
        ):
            policy_name = f"greedy_rules_{heuristic_version}"
            path = trajectories / f"{stem}.{arm}.jsonl"
            arm_result = _completed_rollout(
                path,
                character=character,
                seed=seed,
                ascension=ascension,
                policy=policy_name,
                build_id=build_id,
            )
            if arm_result is None:
                discarded_incomplete += int(
                    _discard_matching_incomplete(
                        path,
                        character=character,
                        seed=seed,
                        policy=policy_name,
                    )
                )
                arm_result = _run_benchmark_rollout(
                    path,
                    lambda path=path, version=heuristic_version: run_heuristic_rollout(
                        model_path=macro_model_path,
                        output=path,
                        character=character,
                        ascension=ascension,
                        seed=seed,
                        max_steps=max_steps,
                        build_id=build_id,
                        heuristic_version=version,
                        safe_potion_ids=("POTION.ATTACK_POTION",),
                        config=config,
                    ),
                )
            else:
                reused[arm] += 1
            arm_result["combat_action_counts"] = _combat_action_counts(path)
            arm_results[arm] = arm_result
        rows.append(
            {
                "case_index": case_index,
                "character": character,
                "canonical_character": canonical_character(character),
                "seed": seed,
                "ascension": ascension,
                **arm_results,
                "paired_floor_delta": int(arm_results["candidate"].get("floor") or 0)
                - int(arm_results["baseline"].get("floor") or 0),
            }
        )

    aggregate = aggregate_combat_model_pair(rows)
    frozen_gate = manifest.get("promotion_gate") or {}
    overall = aggregate["overall"]
    by_character = aggregate["by_character"]
    criteria = {
        "overall_mean_paired_floor_delta_at_least_0_75": (
            float(overall["paired_floor_delta"])
            >= float(frozen_gate["overall_mean_paired_floor_delta_at_least"])
        ),
        "every_character_mean_paired_floor_delta_at_least_minus_0_5": all(
            float(by_character[character]["paired_floor_delta"])
            >= float(frozen_gate["every_character_mean_paired_floor_delta_at_least"])
            for character in HEADLESS_CHARACTERS
        ),
        "candidate_victories_at_least_baseline": (
            int(overall["candidate"]["victories"]) >= int(overall["baseline"]["victories"])
        ),
        "candidate_truncations_at_most_baseline": (
            int(overall["candidate"]["truncations"]) <= int(overall["baseline"]["truncations"])
        ),
    }
    result = {
        "schema_version": 1,
        "evaluation": "frozen_paired_chip_aware_heuristic_comparison",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "build_id": build_id,
        "ascension": ascension,
        "count_per_character": count_per_character,
        "max_steps": max_steps,
        "macro_model": str(Path(macro_model_path).resolve()),
        "macro_model_sha256": macro_sha256,
        "policy_constraints": constraints,
        "resume": {
            "reused_completed_trajectories": reused,
            "discarded_matching_incomplete_trajectories": discarded_incomplete,
        },
        "aggregate": aggregate,
        "combat_action_counts": _pair_combat_action_counts(rows),
        "promotion_gate": {
            "passed": all(criteria.values()),
            "criteria": criteria,
            "decision": (
                "eligible_as_next_dagger_teacher"
                if all(criteria.values())
                else "reject_without_tuning_on_these_cases"
            ),
        },
        "cases": rows,
        "limitations": [
            "This tests a tactical rules teacher, not the learned Deep Sets policy.",
            "Ten fresh seeds per character are a bounded promotion screen.",
            "Failed cases cannot be reused to tune another tactical rule candidate.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return result


def evaluate_combat_policy_baseline(
    *,
    manifest_path: str | Path,
    macro_model_path: str | Path,
    combat_model_path: str | Path,
    output: str | Path,
    trajectory_dir: str | Path,
    max_steps: int = 1_000,
    build_id: str = "v0.107.1",
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Run every case in a frozen single-policy baseline exactly once."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("name") != "stsai-frozen-online-combat-policy-baseline":
        raise ValueError("Manifest is not a frozen online combat policy baseline")
    target = manifest.get("target") or {}
    if target.get("build_id") and str(target["build_id"]) != build_id:
        raise ValueError(f"Manifest targets {target['build_id']}; evaluator targets {build_id}")
    ascension = int(target.get("ascension", 10))
    selection = manifest.get("selection") or {}
    count_per_character = int(selection.get("count_per_character") or 0)
    cases = select_benchmark_cases(manifest, count_per_character)
    if len(cases) != len(manifest.get("cases") or []):
        raise ValueError("Frozen baseline manifest contains unexpected extra cases")

    frozen_models = manifest.get("models")
    frozen_models = frozen_models if isinstance(frozen_models, Mapping) else {}
    frozen_macro = frozen_models.get("macro")
    frozen_macro = frozen_macro if isinstance(frozen_macro, Mapping) else {}
    frozen_combat = frozen_models.get("combat")
    frozen_combat = frozen_combat if isinstance(frozen_combat, Mapping) else {}
    macro_sha256 = _sha256(macro_model_path)
    combat_sha256 = _sha256(combat_model_path)
    combat_kind = str(frozen_combat.get("kind"))
    if str(frozen_macro.get("sha256")) != macro_sha256:
        raise ValueError("Macro checkpoint does not match frozen baseline manifest")
    if combat_kind not in {"mlp", "set"} or str(frozen_combat.get("sha256")) != combat_sha256:
        raise ValueError("Combat checkpoint does not match frozen baseline manifest")
    constraints = manifest.get("policy_constraints")
    constraints = constraints if isinstance(constraints, Mapping) else {}
    if constraints.get("safe_potion_ids") != ["POTION.ATTACK_POTION"]:
        raise ValueError("Frozen baseline does not specify the audited potion whitelist")

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Frozen baseline evaluation already exists: {destination}")
    trajectories = Path(trajectory_dir)
    trajectories.mkdir(parents=True, exist_ok=True)
    expected_names = {
        name
        for case_index, case in enumerate(cases)
        for name in (
            f"{case_index:03d}.{case['character'].lower()}.{case['seed']}.policy.jsonl",
            f"{case_index:03d}.{case['character'].lower()}.{case['seed']}.policy.jsonl.tmp",
        )
    }
    unexpected = [path.name for path in trajectories.iterdir() if path.name not in expected_names]
    if unexpected:
        raise ValueError(
            "Frozen baseline trajectory directory has unexpected files: "
            + ", ".join(sorted(unexpected))
        )

    rows: list[dict[str, Any]] = []
    reused = 0
    discarded_incomplete = 0
    policy_name = _combat_policy_name(combat_kind)
    for case_index, case in enumerate(cases):
        character = case["character"]
        seed = case["seed"]
        stem = f"{case_index:03d}.{character.lower()}.{seed}"
        path = trajectories / f"{stem}.policy.jsonl"
        policy_result = _completed_rollout(
            path,
            character=character,
            seed=seed,
            ascension=ascension,
            policy=policy_name,
            build_id=build_id,
            combat_model_sha256=combat_sha256,
        )
        if policy_result is None:
            discarded_incomplete += int(
                _discard_matching_incomplete(
                    path,
                    character=character,
                    seed=seed,
                    policy=policy_name,
                )
            )
            policy_result = _run_benchmark_rollout(
                path,
                lambda path=path, character=character, seed=seed: _run_combat_checkpoint(
                    combat_kind,
                    macro_model_path=macro_model_path,
                    combat_model_path=combat_model_path,
                    output=path,
                    character=character,
                    ascension=ascension,
                    seed=seed,
                    max_steps=max_steps,
                    build_id=build_id,
                    config=config,
                ),
            )
        else:
            reused += 1
        policy_result["combat_action_counts"] = _combat_action_counts(path)
        rows.append(
            {
                "case_index": case_index,
                "character": character,
                "canonical_character": canonical_character(character),
                "seed": seed,
                "ascension": ascension,
                "policy": policy_result,
            }
        )

    def action_counts(selected: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for row in selected:
            counts.update(row["policy"].get("combat_action_counts") or {})
        return dict(counts)

    result = {
        "schema_version": 1,
        "evaluation": "frozen_online_combat_policy_baseline",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_name": manifest.get("name"),
        "build_id": build_id,
        "ascension": ascension,
        "count_per_character": count_per_character,
        "max_steps": max_steps,
        "selection": "all precommitted cases in manifest order; outcomes opened after completion",
        "macro_model": str(Path(macro_model_path).resolve()),
        "macro_model_sha256": macro_sha256,
        "combat_model": str(Path(combat_model_path).resolve()),
        "combat_model_sha256": combat_sha256,
        "combat_kind": combat_kind,
        "policy_constraints": constraints,
        "resume": {
            "reused_completed_trajectories": reused,
            "discarded_matching_incomplete_trajectories": discarded_incomplete,
        },
        "aggregate": aggregate_combat_policy_baseline(rows),
        "combat_action_counts": {
            "overall": action_counts(rows),
            "by_character": {
                character: action_counts([row for row in rows if row.get("character") == character])
                for character in HEADLESS_CHARACTERS
            },
        },
        "cases": rows,
        "limitations": [
            (
                f"{count_per_character} frozen seeds per character are a cheap online "
                "baseline, not a final statistical claim."
            ),
            "Runtime errors are retained as truncations and counted against the policy.",
            "This measurement does not promote or tune a policy.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return result


def evaluate_combat_model_pair(
    *,
    manifest_path: str | Path,
    macro_model_path: str | Path,
    baseline_model_path: str | Path,
    candidate_model_path: str | Path,
    baseline_kind: str = "mlp",
    candidate_kind: str = "mlp",
    output: str | Path,
    trajectory_dir: str | Path,
    pool: str = "validation",
    count_per_character: int = 5,
    parallel_workers: int = 1,
    max_steps: int = 1_000,
    build_id: str = "v0.107.1",
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Pair two combat checkpoints on one untouched precommitted manifest pool."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    target = manifest.get("target") or {}
    if target.get("build_id") and str(target["build_id"]) != build_id:
        raise ValueError(f"Manifest targets {target['build_id']}; evaluator targets {build_id}")
    ascension = int(target.get("ascension", 10))
    has_named_pool = any(
        isinstance(case, Mapping) and case.get("pool") == pool
        for case in manifest.get("cases") or []
    )
    cases = (
        select_benchmark_pool_cases(manifest, pool, count_per_character)
        if has_named_pool
        else select_benchmark_cases(manifest, count_per_character)
    )
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Frozen evaluation already exists: {destination}")
    trajectories = Path(trajectory_dir)
    trajectories.mkdir(parents=True, exist_ok=True)
    expected_names = {
        name
        for case_index, case in enumerate(cases)
        for name in (
            f"{case_index:03d}.{case['character'].lower()}.{case['seed']}.baseline.jsonl",
            f"{case_index:03d}.{case['character'].lower()}.{case['seed']}.baseline.jsonl.tmp",
            f"{case_index:03d}.{case['character'].lower()}.{case['seed']}.candidate.jsonl",
            f"{case_index:03d}.{case['character'].lower()}.{case['seed']}.candidate.jsonl.tmp",
        )
    }
    unexpected = [path.name for path in trajectories.iterdir() if path.name not in expected_names]
    if unexpected:
        raise ValueError(
            "Frozen evaluation trajectory directory has unexpected files: "
            + ", ".join(sorted(unexpected))
        )
    baseline_sha256 = _sha256(baseline_model_path)
    candidate_sha256 = _sha256(candidate_model_path)
    frozen_models = manifest.get("models")
    frozen_models = frozen_models if isinstance(frozen_models, Mapping) else {}
    for arm, kind, sha256 in (
        ("baseline", baseline_kind, baseline_sha256),
        ("candidate", candidate_kind, candidate_sha256),
    ):
        frozen = frozen_models.get(arm)
        frozen = frozen if isinstance(frozen, Mapping) else {}
        if frozen and (str(frozen.get("kind")) != kind or str(frozen.get("sha256")) != sha256):
            raise ValueError(f"{arm.title()} checkpoint does not match frozen manifest")
    reused = {"baseline": 0, "candidate": 0}
    discarded_incomplete = 0
    arm_results: dict[str, dict[str, Any]] = {}
    arm_paths: dict[str, Path] = {}
    jobs: list[HeadlessBatchJob[dict[str, Any]]] = []
    for case_index, case in enumerate(cases):
        character = case["character"]
        seed = case["seed"]
        stem = f"{case_index:03d}.{character.lower()}.{seed}"
        for arm, kind, model_path, model_sha256 in (
            ("baseline", baseline_kind, baseline_model_path, baseline_sha256),
            ("candidate", candidate_kind, candidate_model_path, candidate_sha256),
        ):
            path = trajectories / f"{stem}.{arm}.jsonl"
            key = f"{case_index}:{arm}"
            arm_paths[key] = path
            policy_name = _combat_policy_name(kind)
            completed = _completed_rollout(
                path,
                character=character,
                seed=seed,
                ascension=ascension,
                policy=policy_name,
                build_id=build_id,
                combat_model_sha256=model_sha256,
            )
            if completed is not None:
                reused[arm] += 1
                arm_results[key] = completed
                continue
            discarded_incomplete += int(
                _discard_matching_incomplete(
                    path,
                    character=character,
                    seed=seed,
                    policy=policy_name,
                )
            )
            run_checkpoint = partial(
                _run_combat_checkpoint,
                kind,
                macro_model_path=macro_model_path,
                combat_model_path=model_path,
                output=path,
                character=character,
                ascension=ascension,
                seed=seed,
                max_steps=max_steps,
                build_id=build_id,
                config=config,
            )
            jobs.append(
                HeadlessBatchJob(
                    key,
                    partial(_run_benchmark_rollout, path, run_checkpoint),
                )
            )

    batch_report = HeadlessBatchManager(parallel_workers).run(jobs)
    for job, value in zip(jobs, batch_report.values(), strict=True):
        arm_results[job.key] = value

    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        character = case["character"]
        seed = case["seed"]
        paired: dict[str, dict[str, Any]] = {}
        for arm in ("baseline", "candidate"):
            key = f"{case_index}:{arm}"
            arm_result = arm_results[key]
            arm_result["combat_action_counts"] = _combat_action_counts(arm_paths[key])
            paired[arm] = arm_result
        rows.append(
            {
                "case_index": case_index,
                "character": character,
                "canonical_character": canonical_character(character),
                "seed": seed,
                "ascension": ascension,
                **paired,
                "paired_floor_delta": int(paired["candidate"].get("floor") or 0)
                - int(paired["baseline"].get("floor") or 0),
            }
        )

    result = {
        "schema_version": 1,
        "evaluation": "frozen_paired_headless_combat_checkpoint_comparison",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_name": manifest.get("name"),
        "manifest_pool": pool,
        "build_id": build_id,
        "ascension": ascension,
        "count_per_character": count_per_character,
        "max_steps": max_steps,
        "execution": {
            "mode": "bounded_native_process_batch",
            "parallel_workers": parallel_workers,
            "submitted_jobs": len(jobs),
        },
        "selection": (
            "named precommitted manifest pool in original order"
            if has_named_pool
            else "all precommitted manifest cases in original order"
        ),
        "macro_model": str(Path(macro_model_path).resolve()),
        "macro_model_sha256": _sha256(macro_model_path),
        "baseline_model": str(Path(baseline_model_path).resolve()),
        "baseline_model_sha256": baseline_sha256,
        "baseline_kind": baseline_kind,
        "candidate_model": str(Path(candidate_model_path).resolve()),
        "candidate_model_sha256": candidate_sha256,
        "candidate_kind": candidate_kind,
        "resume": {
            "reused_completed_trajectories": reused,
            "discarded_matching_incomplete_trajectories": discarded_incomplete,
        },
        "aggregate": aggregate_combat_model_pair(rows),
        "combat_action_counts": _pair_combat_action_counts(rows),
        "cases": rows,
        "limitations": [
            (
                f"{count_per_character} validation seeds per character require interpreting "
                "the reported paired interval, not only the point estimate."
            ),
            "Runtime errors are retained as truncations and counted against their arm.",
            "No policy is promoted by this validation comparison alone.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return result
