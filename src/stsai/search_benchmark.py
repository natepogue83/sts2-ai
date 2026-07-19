"""Precommitted paired evaluation for rank-only one-ply native search."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from stsai.headless import HeadlessConfig
from stsai.headless_batch import HeadlessBatchJob, HeadlessBatchManager
from stsai.headless_benchmark import (
    _combat_action_counts,
    _completed_rollout,
    _discard_matching_incomplete,
    _pair_combat_action_counts,
    _run_benchmark_rollout,
    aggregate_combat_model_pair,
    select_benchmark_cases,
)
from stsai.headless_env import HEADLESS_CHARACTERS, run_headless_rollout
from stsai.ids import canonical_character
from stsai.policy import MaskedMLPPolicy
from stsai.search import OnePlyForkSearchPolicy, OnePlySearchConfig
from stsai.set_policy import DeepSetCombatPolicy, HeadlessSetCombatPolicy, run_set_combat_rollout
from stsai.value import RankOnlyPublicValueEvaluator


BASE_POLICY_NAME = "combat_deepsets_greedy_noncombat_v1"
SEARCH_POLICY_NAME = "combat_deepsets_one_ply_rank_v1"
_CHARACTER_CODES = {
    "Defect": "D",
    "Ironclad": "I",
    "Necrobinder": "N",
    "Regent": "R",
    "Silent": "S",
}


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        with temporary.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and path.exists():
            raise FileExistsError(f"Refusing to replace frozen artifact: {path}")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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


def create_rank_search_manifest(
    output: str | os.PathLike[str],
    *,
    macro_model_path: str | os.PathLike[str],
    combat_model_path: str | os.PathLike[str],
    value_model_path: str | os.PathLike[str],
    seed_prefix: str,
    count_per_character: int,
    excluded_manifests: Sequence[str | os.PathLike[str]] = (),
    build_id: str = "v0.107.1",
    ascension: int = 10,
    max_nodes: int = 8,
    parallel_branches: int = 1,
) -> dict[str, Any]:
    """Freeze models, seeds, power target, and promotion gate before outcomes."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Rank-search manifest already exists: {destination}")
    if count_per_character <= 0 or max_nodes <= 0 or parallel_branches <= 0:
        raise ValueError("Counts, max_nodes, and parallel_branches must be positive")
    if not seed_prefix:
        raise ValueError("A non-empty seed prefix is required")
    macro_sha = _sha256(macro_model_path)
    combat_sha = _sha256(combat_model_path)
    value_sha = _sha256(value_model_path)
    ranker = RankOnlyPublicValueEvaluator(value_model_path, expected_sha256=value_sha)
    excluded, exclusion_sources = _manifest_seeds(excluded_manifests)
    cases: list[dict[str, str]] = []
    for character in HEADLESS_CHARACTERS:
        accepted = 0
        candidate = 0
        while accepted < count_per_character:
            seed = f"{seed_prefix}{_CHARACTER_CODES[character]}{candidate:06d}"
            candidate += 1
            if seed in excluded:
                continue
            cases.append({"character": canonical_character(character), "seed": seed})
            accepted += 1
    selected = {case["seed"] for case in cases}
    if len(selected) != len(cases) or selected & excluded:
        raise AssertionError("Rank-search seeds must be unique and fresh")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "name": "stsai-frozen-rank-only-one-ply-search-pair",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {"build_id": build_id, "ascension": ascension},
        "selection": {
            "method": "fresh-prefix-seeds-selected-before-outcomes",
            "seed_prefix": seed_prefix,
            "count_per_character": count_per_character,
            "total_cases": len(cases),
            "outcomes_inspected": False,
            "excluded_manifests": exclusion_sources,
            "excluded_seed_count": len(excluded),
        },
        "base_policy": {
            "name": BASE_POLICY_NAME,
            "macro_model_sha256": macro_sha,
            "combat_model_sha256": combat_sha,
            "safe_potion_ids": ["POTION.ATTACK_POTION"],
        },
        "search_policy": {
            "name": SEARCH_POLICY_NAME,
            "depth": 1,
            "max_nodes": max_nodes,
            "parallel_branches": parallel_branches,
            "visibility_mode": "public_deterministic_only",
            "value_checkpoint_sha256": value_sha,
            "value_feature_schema": ranker.model.feature_schema,
            "original_calibration_status": ranker.model.calibration_status,
            "rank_invariant_positive_slopes": list(ranker.rank_invariant_slopes),
            "absolute_probability_claimed": False,
            "training_labels_emitted": False,
        },
        "power": {
            "paired_cases": len(cases),
            "conservative_paired_floor_sd": 2.6,
            "two_sided_alpha": 0.05,
            "practical_effect_floor_delta": 0.75,
            "approximate_power_for_practical_effect": 0.983,
            "small_effects_are_out_of_scope": True,
        },
        "promotion_gate": {
            "paired_floor_delta_interval_lower_above": 0.0,
            "candidate_victories_at_least_baseline": True,
            "candidate_truncations_at_most_baseline": True,
            "every_character_mean_delta_at_least": -0.5,
            "search_used_in_at_least_one_case": True,
            "training_labels_remain_disabled_until_gate_passes": True,
        },
        "cases": cases,
    }
    _atomic_json(destination, payload, exclusive=True)
    return payload


def run_rank_search_rollout(
    *,
    macro_model_path: str | os.PathLike[str],
    combat_model_path: str | os.PathLike[str],
    value_model_path: str | os.PathLike[str],
    expected_value_sha256: str,
    output: str | os.PathLike[str],
    character: str,
    ascension: int,
    seed: str,
    max_nodes: int,
    parallel_branches: int,
    max_steps: int = 1_000,
    build_id: str = "v0.107.1",
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Run Deep Sets plus rank-only search without emitting training labels."""

    macro_model_path = Path(macro_model_path)
    combat_model_path = Path(combat_model_path)
    value_model_path = Path(value_model_path)
    base_policy = HeadlessSetCombatPolicy(
        MaskedMLPPolicy.load(macro_model_path),
        DeepSetCombatPolicy.load(combat_model_path),
        build_id=build_id,
        safe_potion_ids=frozenset(("POTION.ATTACK_POTION",)),
    )
    ranker = RankOnlyPublicValueEvaluator(
        value_model_path,
        expected_sha256=expected_value_sha256,
    )
    search_policy = OnePlyForkSearchPolicy(
        base_policy,
        ranker,
        config=OnePlySearchConfig(
            max_nodes=max_nodes,
            parallel_branches=parallel_branches,
        ),
    )
    return run_headless_rollout(
        output=output,
        character=character,
        ascension=ascension,
        seed=seed,
        policy_name=SEARCH_POLICY_NAME,
        select_action=search_policy.select,
        policy_metadata={
            "build_id": build_id,
            "macro_model_sha256": _sha256(macro_model_path),
            "combat_model_sha256": _sha256(combat_model_path),
            "value_model_sha256": expected_value_sha256,
            "value_original_calibration_status": ranker.model.calibration_status,
            "value_rank_invariant_positive_slopes": list(ranker.rank_invariant_slopes),
            "visibility_mode": "public_deterministic_only",
            "depth": 1,
            "max_nodes": max_nodes,
            "parallel_branches": parallel_branches,
            "safe_potion_ids": ["POTION.ATTACK_POTION"],
            "training_labels_emitted": False,
        },
        max_steps=max_steps,
        config=config,
    )


def _search_coverage(path: Path) -> dict[str, Any]:
    combat_decisions = 0
    metadata_decisions = 0
    used = 0
    changed = 0
    nodes = 0
    reasons: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, Mapping) or event.get("event_type") != "transition":
                continue
            observation = event.get("observation")
            observation = observation if isinstance(observation, Mapping) else {}
            combat_decisions += int(observation.get("decision_type") == "combat_play")
            info = event.get("info")
            info = info if isinstance(info, Mapping) else {}
            policy = info.get("policy")
            policy = policy if isinstance(policy, Mapping) else {}
            search = policy.get("search")
            if not isinstance(search, Mapping):
                continue
            metadata_decisions += 1
            if search.get("used") is True:
                used += 1
                nodes += int(search.get("nodes") or 0)
                changed += int(
                    search.get("selected_action_index") != search.get("base_action_index")
                )
            else:
                reasons[str(search.get("reason") or "unknown")] += 1
    return {
        "combat_decisions": combat_decisions,
        "decisions_with_search_metadata": metadata_decisions,
        "search_used_decisions": used,
        "search_changed_action_decisions": changed,
        "branch_nodes": nodes,
        "search_used_fraction_of_combat_decisions": (
            round(used / combat_decisions, 6) if combat_decisions else 0.0
        ),
        "fallback_reasons": dict(reasons),
    }


def _validate_completed_search_metadata(path: Path, expected_value_sha256: str) -> None:
    with path.open("r", encoding="utf-8-sig") as handle:
        first = json.loads(handle.readline())
    metadata = first.get("policy_metadata") if isinstance(first, Mapping) else None
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if str(metadata.get("value_model_sha256")) != expected_value_sha256:
        raise ValueError(f"Completed search trajectory has a different value checkpoint: {path}")


def evaluate_rank_search_pair(
    *,
    manifest_path: str | os.PathLike[str],
    macro_model_path: str | os.PathLike[str],
    combat_model_path: str | os.PathLike[str],
    value_model_path: str | os.PathLike[str],
    output: str | os.PathLike[str],
    trajectory_dir: str | os.PathLike[str],
    parallel_workers: int = 4,
    max_steps: int = 1_000,
    build_id: str = "v0.107.1",
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Pair base Deep Sets and rank-only one-ply search on every frozen case."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("name") != "stsai-frozen-rank-only-one-ply-search-pair":
        raise ValueError("Manifest is not a frozen rank-only search pair")
    target = manifest.get("target") or {}
    if str(target.get("build_id")) != build_id:
        raise ValueError("Rank-search manifest and evaluator build ids differ")
    ascension = int(target.get("ascension", 10))
    selection = manifest.get("selection") or {}
    count_per_character = int(selection.get("count_per_character") or 0)
    cases = select_benchmark_cases(manifest, count_per_character)
    frozen_base = manifest.get("base_policy") or {}
    frozen_search = manifest.get("search_policy") or {}
    macro_sha = _sha256(macro_model_path)
    combat_sha = _sha256(combat_model_path)
    value_sha = _sha256(value_model_path)
    if str(frozen_base.get("macro_model_sha256")) != macro_sha:
        raise ValueError("Macro checkpoint does not match rank-search manifest")
    if str(frozen_base.get("combat_model_sha256")) != combat_sha:
        raise ValueError("Combat checkpoint does not match rank-search manifest")
    if str(frozen_search.get("value_checkpoint_sha256")) != value_sha:
        raise ValueError("Value checkpoint does not match rank-search manifest")
    max_nodes = int(frozen_search.get("max_nodes") or 0)
    parallel_branches = int(frozen_search.get("parallel_branches") or 1)
    RankOnlyPublicValueEvaluator(value_model_path, expected_sha256=value_sha)

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Frozen rank-search evaluation already exists: {destination}")
    trajectories = Path(trajectory_dir)
    trajectories.mkdir(parents=True, exist_ok=True)
    expected_names = {
        name
        for case_index, case in enumerate(cases)
        for arm in ("baseline", "search")
        for name in (
            f"{case_index:03d}.{case['character'].lower()}.{case['seed']}.{arm}.jsonl",
            f"{case_index:03d}.{case['character'].lower()}.{case['seed']}.{arm}.jsonl.tmp",
        )
    }
    unexpected = [path.name for path in trajectories.iterdir() if path.name not in expected_names]
    if unexpected:
        raise ValueError(
            "Rank-search trajectory directory has unexpected files: "
            + ", ".join(sorted(unexpected))
        )

    results: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    reused = {"baseline": 0, "search": 0}
    discarded_incomplete = 0
    jobs: list[HeadlessBatchJob[dict[str, Any]]] = []
    for case_index, case in enumerate(cases):
        character = case["character"]
        seed = case["seed"]
        stem = f"{case_index:03d}.{character.lower()}.{seed}"
        for arm, policy_name in (
            ("baseline", BASE_POLICY_NAME),
            ("search", SEARCH_POLICY_NAME),
        ):
            key = f"{case_index}:{arm}"
            path = trajectories / f"{stem}.{arm}.jsonl"
            paths[key] = path
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
                if arm == "search":
                    _validate_completed_search_metadata(path, value_sha)
                results[key] = completed
                reused[arm] += 1
                continue
            if path.exists():
                raise ValueError(f"Refusing to replace mismatched completed trajectory: {path}")
            discarded_incomplete += int(
                _discard_matching_incomplete(
                    path,
                    character=character,
                    seed=seed,
                    policy=policy_name,
                )
            )
            if arm == "baseline":
                run = partial(
                    run_set_combat_rollout,
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
            else:
                run = partial(
                    run_rank_search_rollout,
                    macro_model_path=macro_model_path,
                    combat_model_path=combat_model_path,
                    value_model_path=value_model_path,
                    expected_value_sha256=value_sha,
                    output=path,
                    character=character,
                    ascension=ascension,
                    seed=seed,
                    max_nodes=max_nodes,
                    parallel_branches=parallel_branches,
                    max_steps=max_steps,
                    build_id=build_id,
                    config=config,
                )
            jobs.append(HeadlessBatchJob(key, partial(_run_benchmark_rollout, path, run)))

    batch_report = HeadlessBatchManager(parallel_workers).run(jobs)
    for job, value in zip(jobs, batch_report.values(), strict=True):
        results[job.key] = value

    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        paired: dict[str, dict[str, Any]] = {}
        for source_arm, result_arm in (("baseline", "baseline"), ("search", "candidate")):
            key = f"{case_index}:{source_arm}"
            arm_result = results[key]
            arm_result["combat_action_counts"] = _combat_action_counts(paths[key])
            if source_arm == "search":
                arm_result["search_coverage"] = _search_coverage(paths[key])
            paired[result_arm] = arm_result
        rows.append(
            {
                "case_index": case_index,
                "character": case["character"],
                "canonical_character": canonical_character(case["character"]),
                "seed": case["seed"],
                "ascension": ascension,
                **paired,
                "paired_floor_delta": int(paired["candidate"].get("floor") or 0)
                - int(paired["baseline"].get("floor") or 0),
            }
        )
    aggregate = aggregate_combat_model_pair(rows)
    coverage = {
        key: sum(int(row["candidate"]["search_coverage"].get(key) or 0) for row in rows)
        for key in (
            "combat_decisions",
            "decisions_with_search_metadata",
            "search_used_decisions",
            "search_changed_action_decisions",
            "branch_nodes",
        )
    }
    coverage["cases_with_search_used"] = sum(
        int(row["candidate"]["search_coverage"]["search_used_decisions"] > 0) for row in rows
    )
    overall = aggregate["overall"]
    by_character = aggregate["by_character"]
    interval = overall["descriptive_paired_95_interval"]
    gate = manifest.get("promotion_gate") or {}
    criteria = {
        "paired_interval_lower_above_zero": (
            float(interval[0]) > float(gate["paired_floor_delta_interval_lower_above"])
        ),
        "candidate_victories_at_least_baseline": (
            int(overall["candidate"]["victories"]) >= int(overall["baseline"]["victories"])
        ),
        "candidate_truncations_at_most_baseline": (
            int(overall["candidate"]["truncations"]) <= int(overall["baseline"]["truncations"])
        ),
        "every_character_delta_at_least_minus_0_5": all(
            float(by_character[character]["paired_floor_delta"])
            >= float(gate["every_character_mean_delta_at_least"])
            for character in HEADLESS_CHARACTERS
        ),
        "search_used_in_at_least_one_case": coverage["cases_with_search_used"] > 0,
    }
    passed = all(criteria.values())
    payload = {
        "schema_version": 1,
        "evaluation": "frozen_paired_rank_only_one_ply_search",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "build_id": build_id,
        "ascension": ascension,
        "count_per_character": count_per_character,
        "max_steps": max_steps,
        "models": {
            "macro_sha256": macro_sha,
            "combat_sha256": combat_sha,
            "value_sha256": value_sha,
        },
        "execution": {
            "mode": "bounded_native_process_batch",
            "parallel_workers": parallel_workers,
            "submitted_jobs": len(jobs),
            "max_search_nodes": max_nodes,
            "parallel_search_branches": parallel_branches,
        },
        "resume": {
            "reused_completed_trajectories": reused,
            "discarded_matching_incomplete_trajectories": discarded_incomplete,
        },
        "aggregate": aggregate,
        "combat_action_counts": _pair_combat_action_counts(rows),
        "search_coverage": coverage,
        "promotion_gate": {
            "passed": passed,
            "criteria": criteria,
            "decision": (
                "eligible_for_bounded_expert_iteration_design"
                if passed
                else "do_not_emit_search_training_labels"
            ),
        },
        "training_labels_emitted": False,
        "training_labels_authorized": False,
        "cases": rows,
        "limitations": [
            "The value checkpoint is used only as a ranker; no calibrated probability claim is made.",
            "Search is depth one and limited to public-deterministic allowlisted actions.",
            "No search branch value or selected action is emitted as a training label in this tranche.",
        ],
    }
    _atomic_json(destination, payload, exclusive=True)
    return payload
