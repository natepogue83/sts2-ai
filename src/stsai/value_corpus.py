"""Outcome-blind planning and collection for independent value calibration."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from stsai.headless import HeadlessConfig
from stsai.headless_env import HEADLESS_CHARACTERS
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
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_seeds(
    paths: Sequence[str | os.PathLike[str]],
) -> tuple[set[str], list[dict[str, str]]]:
    seeds: set[str] = set()
    sources: list[dict[str, str]] = []
    for raw in paths:
        path = Path(raw)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        for case in payload.get("cases") or []:
            if isinstance(case, Mapping) and case.get("seed"):
                seeds.add(str(case["seed"]))
        sources.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    return seeds, sources


def create_value_calibration_manifest(
    output: str | Path,
    *,
    build_id: str = "v0.107.1",
    ascension: int = 10,
    development_per_character: int = 15,
    audit_per_character: int = 15,
    seed_prefix: str = "VCAL3",
    excluded_manifests: Sequence[str | os.PathLike[str]] = (),
) -> dict[str, Any]:
    """Precommit disjoint character-balanced development and blind-audit seeds."""

    if not build_id or not 0 <= ascension <= 10:
        raise ValueError("A build id and ascension from 0 through 10 are required")
    if development_per_character < 0 or audit_per_character <= 0:
        raise ValueError("Development count must be nonnegative and audit positive")
    if not seed_prefix:
        raise ValueError("A non-empty value calibration seed prefix is required")
    excluded_seeds, exclusion_sources = _manifest_seeds(excluded_manifests)
    cases: list[dict[str, str]] = []
    for character in HEADLESS_CHARACTERS:
        canonical = canonical_character(character)
        code = _CHARACTER_CODES[character]
        for pool, marker, count in (
            ("development", "D", development_per_character),
            ("audit", "A", audit_per_character),
        ):
            accepted = 0
            candidate = 0
            while accepted < count:
                seed = f"{seed_prefix}{marker}{code}{candidate:06d}"
                candidate += 1
                if seed in excluded_seeds:
                    continue
                cases.append({"pool": pool, "character": canonical, "seed": seed})
                accepted += 1
    selected = [case["seed"] for case in cases]
    if len(selected) != len(set(selected)):
        raise AssertionError("Value calibration seeds must be unique")
    if set(selected) & excluded_seeds:
        raise AssertionError("Value calibration seeds overlap an excluded manifest")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "name": f"stsai-value-calibration-{seed_prefix.lower()}-independent-audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {
            "build_id": build_id,
            "ascension": ascension,
            "value": "survive_next_3_floors",
        },
        "selection": {
            "method": "synthetic-character-balanced-seeds-frozen-before-outcomes",
            "seed_prefix": seed_prefix,
            "development_per_character": development_per_character,
            "audit_per_character": audit_per_character,
            "outcomes_inspected": False,
            "pools_disjoint": True,
            "excluded_manifests": exclusion_sources,
            "excluded_seed_count": len(excluded_seeds),
        },
        "audit_protocol": {
            "collection_is_outcome_blind": True,
            "development_pool_may_be_used_for_training": (development_per_character > 0),
            "audit_pool_must_not_be_read_until_checkpoint_is_frozen": True,
            "audit_is_opened_once": True,
            "promotion_gate": {
                "overall_ece_improved": True,
                "brier_degradation_at_most": 0.002,
                "every_character_ece_at_most": 0.15,
            },
        },
        "cases": cases,
    }
    _atomic_json(Path(output), payload)
    return payload


def _is_completed_rollout(path: Path, *, character: str, seed: str) -> bool:
    if not path.is_file():
        return False
    first: Mapping[str, Any] | None = None
    last: Mapping[str, Any] | None = None
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, Mapping):
                    return False
                if first is None:
                    first = event
                last = event
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return bool(
        first
        and last
        and first.get("event_type") == "episode_start"
        and canonical_character(first.get("character")) == canonical_character(character)
        and str(first.get("seed")) == seed
        and last.get("event_type") == "episode_end"
        and last.get("terminated") is True
        and last.get("truncated") is not True
    )


def collect_value_calibration_pool(
    manifest_path: str | Path,
    pool: str,
    trajectory_dir: str | Path,
    *,
    macro_model_path: str | Path,
    combat_model_path: str | Path,
    receipt_output: str | Path | None = None,
    max_steps: int = 1_000,
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Collect one frozen pool without exposing any gameplay outcomes."""

    if pool not in {"development", "audit"}:
        raise ValueError(f"Unsupported value calibration pool: {pool}")
    from stsai.set_policy import run_set_combat_rollout

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    target = manifest.get("target") or {}
    build_id = str(target.get("build_id") or "")
    ascension = int(target.get("ascension", 10))
    if not build_id:
        raise ValueError("Value calibration manifest is missing target.build_id")
    cases = [
        case
        for case in manifest.get("cases") or []
        if isinstance(case, Mapping) and case.get("pool") == pool
    ]
    if not cases:
        raise ValueError(f"Value calibration manifest has no {pool} cases")
    destination = Path(trajectory_dir)
    destination.mkdir(parents=True, exist_ok=True)
    by_character: Counter[str] = Counter()
    reused = 0
    completed_files: list[Path] = []
    for index, case in enumerate(cases):
        canonical = canonical_character(case.get("character"))
        character = canonical.rsplit(".", 1)[-1].title()
        seed = str(case.get("seed") or "")
        if not seed:
            raise ValueError(f"Value calibration case {index} has no seed")
        path = destination / f"{index:03d}.{character.lower()}.{seed}.jsonl"
        if _is_completed_rollout(path, character=character, seed=seed):
            reused += 1
        else:
            result = run_set_combat_rollout(
                macro_model_path=macro_model_path,
                combat_model_path=combat_model_path,
                output=path,
                character=character,
                ascension=ascension,
                seed=seed,
                max_steps=max_steps,
                build_id=build_id,
                safe_potion_ids=("POTION.ATTACK_POTION",),
                config=config,
            )
            if result.get("terminated") is not True or result.get("truncated") is True:
                raise RuntimeError(f"Calibration rollout did not complete: {canonical}/{seed}")
        if not _is_completed_rollout(path, character=character, seed=seed):
            raise RuntimeError(f"Calibration rollout failed validation: {canonical}/{seed}")
        completed_files.append(path)
        by_character[canonical] += 1
    file_entries = [f"{path.name}:{_sha256(path)}" for path in completed_files]
    corpus_digest = hashlib.sha256("\n".join(file_entries).encode()).hexdigest()
    receipt = {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "pool": pool,
        "trajectory_dir": str(destination.resolve()),
        "runs": len(cases),
        "completed_runs": len(completed_files),
        "reused_completed_runs": reused,
        "by_character": dict(by_character),
        "corpus_sha256": corpus_digest,
        "macro_model_sha256": _sha256(Path(macro_model_path)),
        "combat_model_sha256": _sha256(Path(combat_model_path)),
        "outcomes_reported": False,
    }
    if receipt_output is not None:
        _atomic_json(Path(receipt_output), receipt)
    return receipt
