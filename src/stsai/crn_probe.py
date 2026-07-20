"""Measurement-only probe for paired native counterfactual rollout coupling."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean, variance
from typing import Any, Mapping, Sequence

from stsai.headless import HeadlessConfig
from stsai.headless_batch import HeadlessBatchJob, HeadlessBatchManager
from stsai.headless_benchmark import select_benchmark_cases
from stsai.headless_env import HEADLESS_CHARACTERS, HeadlessAction, Sts2Env
from stsai.ids import canonical_character
from stsai.policy import MaskedMLPPolicy
from stsai.set_policy import DeepSetCombatPolicy, HeadlessSetCombatPolicy


BASE_POLICY_NAME = "combat_deepsets_greedy_noncombat_v1"
PROBE_NAME = "stsai-frozen-crn-coupling-feasibility-v1"
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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


def create_crn_probe_manifest(
    output: str | os.PathLike[str],
    *,
    macro_model_path: str | os.PathLike[str],
    combat_model_path: str | os.PathLike[str],
    seed_prefix: str = "CRN1",
    count_per_character: int = 5,
    roots_per_case: int = 2,
    horizons: Sequence[int] = (1, 2, 4, 8, 16),
    build_id: str = "v0.107.1",
    ascension: int = 10,
) -> dict[str, Any]:
    """Freeze the coupling measurement and mechanical horizon rule before outcomes."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"CRN probe manifest already exists: {destination}")
    normalized_horizons = sorted({int(value) for value in horizons})
    if (
        not seed_prefix
        or count_per_character <= 0
        or roots_per_case <= 0
        or not normalized_horizons
        or normalized_horizons[0] <= 0
    ):
        raise ValueError("Seed prefix, counts, roots, and horizons must be positive")
    if not 0 <= ascension <= 10:
        raise ValueError("Ascension must be between zero and ten")
    runtime = HeadlessConfig().runtime
    cases = [
        {
            "character": canonical_character(character),
            "seed": f"{seed_prefix}{_CHARACTER_CODES[character]}{index:06d}",
        }
        for character in HEADLESS_CHARACTERS
        for index in range(count_per_character)
    ]
    expected_pairs = len(cases) * roots_per_case
    expected_per_character = count_per_character * roots_per_case
    minimum_valid_pairs = min(expected_pairs, max(10, (expected_pairs * 3 + 4) // 5))
    minimum_valid_per_character = min(
        expected_per_character,
        max(2, (expected_per_character * 2 + 4) // 5),
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "name": PROBE_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {
            "build_id": build_id,
            "ascension": ascension,
            "runtime_dll_sha256": _sha256(runtime) if runtime.is_file() else None,
        },
        "base_policy": {
            "name": BASE_POLICY_NAME,
            "macro_model_sha256": _sha256(macro_model_path),
            "combat_model_sha256": _sha256(combat_model_path),
            "safe_potion_ids": ["POTION.ATTACK_POTION"],
            "frozen_for_entire_tranche": True,
            "updates_forbidden_until": "counterfactual_line_passes_or_stops",
        },
        "selection": {
            "method": "fresh-prefix-seeds-selected-before-measurement",
            "seed_prefix": seed_prefix,
            "count_per_character": count_per_character,
            "roots_per_case": roots_per_case,
            "expected_root_pairs": expected_pairs,
            "outcomes_inspected": False,
        },
        "measurement": {
            "horizons_in_policy_decisions": normalized_horizons,
            "root": "first_distinct_top_two_combat_actions_under_frozen_base",
            "branches": ["base_action", "exact_base_repeat", "highest_ranked_alternative"],
            "continuation_policy": BASE_POLICY_NAME,
            "rng_observation": "native_semantic_checkpoint_named_stream_counters",
            "return_metric": "floor_plus_player_hp_fraction_measurement_proxy",
            "variance_comparator": (
                "same_root_paired_difference_vs_within_character_circularly_unpaired_difference"
            ),
            "training_data_emitted": False,
        },
        "horizon_selection_gate": {
            "method": "longest_contiguous_passing_horizon",
            "minimum_valid_pair_fraction": 0.9,
            "minimum_valid_pairs": minimum_valid_pairs,
            "minimum_valid_pairs_per_character": minimum_valid_per_character,
            "minimum_exact_base_repeat_fraction": 1.0,
            "minimum_mean_root_stream_alignment_fraction": 0.75,
            "maximum_paired_to_unpaired_variance_ratio": 0.5,
            "zero_unpaired_variance_is_not_estimable": True,
            "terminal_decision_if_none_pass": "stop_before_counterfactual_dataset_design",
        },
        "authorization": {
            "counterfactual_dataset": False,
            "training_labels": False,
            "policy_overrides": False,
        },
        "cases": cases,
    }
    _atomic_json(destination, payload, exclusive=True)
    return payload


def extract_rng_streams(checkpoint: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Flatten named run, player, and monster RNGs from one native checkpoint."""

    semantic = checkpoint.get("semantic")
    semantic = semantic if isinstance(semantic, Mapping) else {}
    run_seed = str(semantic.get("run_seed_sha256") or "")
    streams: dict[str, dict[str, Any]] = {}
    run_counters = semantic.get("run_rng_counters")
    if isinstance(run_counters, Mapping):
        for name, counter in sorted(run_counters.items()):
            streams[f"run:{name}"] = {
                "scope": "run",
                "seed": run_seed,
                "counter": int(counter),
            }
    for raw_player in semantic.get("player_rngs") or []:
        if not isinstance(raw_player, Mapping):
            continue
        player_index = int(raw_player.get("player_index") or 0)
        seed = raw_player.get("seed")
        counters = raw_player.get("counters")
        if not isinstance(counters, Mapping):
            continue
        for name, counter in sorted(counters.items()):
            streams[f"player:{player_index}:{name}"] = {
                "scope": "player",
                "seed": seed,
                "counter": int(counter),
            }
    combat = semantic.get("combat")
    combat = combat if isinstance(combat, Mapping) else {}
    for creature in combat.get("creatures") or []:
        if not isinstance(creature, Mapping) or creature.get("monster_rng_counter") is None:
            continue
        combat_id = int(creature.get("combat_id") or 0)
        monster_id = str(creature.get("monster_id") or "UNKNOWN")
        streams[f"monster:{combat_id}:{monster_id}"] = {
            "scope": "monster",
            "seed": creature.get("monster_rng_seed"),
            "counter": int(creature["monster_rng_counter"]),
        }
    return streams


def _read_checkpoint(env: Sts2Env) -> dict[str, Any]:
    checkpoint = env.client.command({"cmd": "snapshot_checkpoint"})
    if checkpoint.get("type") != "snapshot_checkpoint" or checkpoint.get("schema_version") != 1:
        raise RuntimeError("Native runtime returned an invalid semantic checkpoint")
    return checkpoint


def _state_metrics(state: Mapping[str, Any]) -> dict[str, Any]:
    player = state.get("player")
    player = player if isinstance(player, Mapping) else {}
    hp = float(player.get("hp") or 0.0)
    max_hp = max(float(player.get("max_hp") or 1.0), 1.0)
    context = state.get("context")
    context = context if isinstance(context, Mapping) else {}
    floor = int(state.get("floor") or context.get("floor") or 0)
    enemies = [
        enemy
        for enemy in state.get("enemies") or []
        if isinstance(enemy, Mapping) and float(enemy.get("hp") or 0.0) > 0.0
    ]
    enemy_hp = sum(float(enemy.get("hp") or 0.0) for enemy in enemies)
    enemy_max_hp = sum(
        float(enemy.get("max_hp") or enemy.get("hp") or 0.0) for enemy in enemies
    )
    return {
        "decision": str(state.get("decision") or "unknown"),
        "floor": floor,
        "player_hp": hp,
        "player_max_hp": max_hp,
        "player_hp_fraction": hp / max_hp,
        "living_enemies": len(enemies),
        "enemy_hp": enemy_hp,
        "enemy_max_hp": enemy_max_hp,
        "floor_plus_player_hp_fraction": floor + hp / max_hp,
        "terminated": state.get("decision") == "game_over",
        "victory": state.get("victory") is True,
    }


def _capture(env: Sts2Env, *, absorbing_terminal: bool = False) -> dict[str, Any]:
    checkpoint = _read_checkpoint(env)
    return {
        "checkpoint_sha256": _json_sha(checkpoint),
        "checkpoint_snapshot_safe": checkpoint.get("snapshot_safe") is True,
        "rng_streams": extract_rng_streams(checkpoint),
        "state": _state_metrics(env.raw_state),
        "absorbing_terminal": absorbing_terminal,
    }


def _policy_factory(
    macro_model: MaskedMLPPolicy,
    combat_model: DeepSetCombatPolicy,
    *,
    build_id: str,
) -> HeadlessSetCombatPolicy:
    return HeadlessSetCombatPolicy(
        macro_model,
        combat_model,
        build_id=build_id,
        safe_potion_ids=frozenset(("POTION.ATTACK_POTION",)),
    )


def _run_branch(
    parent: Sts2Env,
    snapshot: Any,
    root_action: HeadlessAction,
    *,
    macro_model: MaskedMLPPolicy,
    combat_model: DeepSetCombatPolicy,
    build_id: str,
    horizons: Sequence[int],
) -> dict[str, Any]:
    captures: dict[str, dict[str, Any]] = {}
    try:
        with parent.fork(snapshot) as branch:
            policy = _policy_factory(macro_model, combat_model, build_id=build_id)
            terminal_capture: dict[str, Any] | None = None
            for decision_step in range(1, max(horizons) + 1):
                if terminal_capture is not None:
                    if decision_step in horizons:
                        captures[str(decision_step)] = {
                            **terminal_capture,
                            "absorbing_terminal": True,
                        }
                    continue
                if decision_step == 1:
                    try:
                        action_index = branch.legal_actions.index(root_action)
                    except ValueError as exc:
                        raise RuntimeError("Fork did not preserve the root action") from exc
                else:
                    action_index, _ = policy.select(branch, decision_step - 1)
                result = branch.step(action_index)
                if decision_step in horizons or result.terminated:
                    current = _capture(branch)
                    if decision_step in horizons:
                        captures[str(decision_step)] = current
                    if result.terminated:
                        terminal_capture = current
            return {"completed": True, "horizons": captures}
    except Exception as exc:
        return {
            "completed": False,
            "horizons": captures,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def _alternative_index(
    legal_actions: Sequence[HeadlessAction],
    base_index: int,
    policy_info: Mapping[str, Any],
) -> int | None:
    for ranked in policy_info.get("ranked_actions") or []:
        if isinstance(ranked, Mapping):
            index = int(ranked.get("action_index", -1))
            if 0 <= index < len(legal_actions) and index != base_index:
                return index
    return next((index for index in range(len(legal_actions)) if index != base_index), None)


def _run_probe_case(
    *,
    character: str,
    seed: str,
    ascension: int,
    roots_per_case: int,
    horizons: Sequence[int],
    macro_model_path: str | os.PathLike[str],
    combat_model_path: str | os.PathLike[str],
    build_id: str,
    max_parent_steps: int,
    config: HeadlessConfig,
) -> dict[str, Any]:
    macro_model = MaskedMLPPolicy.load(macro_model_path)
    combat_model = DeepSetCombatPolicy.load(combat_model_path)
    parent_policy = _policy_factory(macro_model, combat_model, build_id=build_id)
    roots: list[dict[str, Any]] = []
    try:
        with Sts2Env(config) as env:
            env.reset(character=character, ascension=ascension, seed=seed)
            for parent_step in range(max_parent_steps):
                if env.terminated or len(roots) >= roots_per_case:
                    break
                base_index, policy_info = parent_policy.select(env, parent_step)
                alternative_index = None
                if env.raw_state.get("decision") == "combat_play":
                    alternative_index = _alternative_index(
                        env.legal_actions, base_index, policy_info
                    )
                if alternative_index is not None:
                    snapshot = env.snapshot()
                    root_checkpoint = json.loads(snapshot.native.checkpoint_json)
                    base_action = env.legal_actions[base_index]
                    alternative_action = env.legal_actions[alternative_index]
                    branch_args = {
                        "macro_model": macro_model,
                        "combat_model": combat_model,
                        "build_id": build_id,
                        "horizons": horizons,
                    }
                    roots.append(
                        {
                            "root_index": len(roots),
                            "parent_step": parent_step,
                            "root_checkpoint_sha256": snapshot.native.checkpoint_sha256,
                            "root_rng_streams": extract_rng_streams(root_checkpoint),
                            "root_state": _state_metrics(env.raw_state),
                            "base_action_index": base_index,
                            "base_action": base_action.as_dict(),
                            "alternative_action_index": alternative_index,
                            "alternative_action": alternative_action.as_dict(),
                            "base": _run_branch(
                                env, snapshot, base_action, **branch_args
                            ),
                            "base_repeat": _run_branch(
                                env, snapshot, base_action, **branch_args
                            ),
                            "alternative": _run_branch(
                                env, snapshot, alternative_action, **branch_args
                            ),
                        }
                    )
                env.step(base_index)
        return {
            "character": character,
            "canonical_character": canonical_character(character),
            "seed": seed,
            "completed": True,
            "roots_requested": roots_per_case,
            "roots_collected": len(roots),
            "roots": roots,
        }
    except Exception as exc:
        return {
            "character": character,
            "canonical_character": canonical_character(character),
            "seed": seed,
            "completed": False,
            "roots_requested": roots_per_case,
            "roots_collected": len(roots),
            "roots": roots,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def _stream_alignment(root: Mapping[str, Any], left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[float, list[str]]:
    root_names = sorted((root.get("root_rng_streams") or {}).keys())
    if not root_names:
        return 0.0, []
    left_streams = left.get("rng_streams") or {}
    right_streams = right.get("rng_streams") or {}
    different = [
        name for name in root_names if left_streams.get(name) != right_streams.get(name)
    ]
    return (len(root_names) - len(different)) / len(root_names), different


def _sample_variance(values: Sequence[float]) -> float | None:
    return variance(values) if len(values) >= 2 else None


def summarize_horizon(
    cases: Sequence[Mapping[str, Any]],
    horizon: int,
    *,
    expected_per_character: int,
) -> dict[str, Any]:
    """Summarize one frozen horizon, including deterministic unpaired comparison."""

    valid: list[dict[str, Any]] = []
    repeat_attempts = 0
    repeat_exact = 0
    desynchronized: Counter[str] = Counter()
    for case in cases:
        character = str(case.get("character") or "unknown")
        for root in case.get("roots") or []:
            if not isinstance(root, Mapping):
                continue
            base = (root.get("base") or {}).get("horizons", {}).get(str(horizon))
            repeated = (root.get("base_repeat") or {}).get("horizons", {}).get(str(horizon))
            alternative = (root.get("alternative") or {}).get("horizons", {}).get(str(horizon))
            if isinstance(base, Mapping) and isinstance(repeated, Mapping):
                repeat_attempts += 1
                repeat_exact += int(
                    base.get("checkpoint_sha256") == repeated.get("checkpoint_sha256")
                )
            if not all(isinstance(value, Mapping) for value in (base, repeated, alternative)):
                continue
            alignment, different = _stream_alignment(root, base, alternative)
            desynchronized.update(different)
            valid.append(
                {
                    "character": character,
                    "base_return": float(base["state"]["floor_plus_player_hp_fraction"]),
                    "alternative_return": float(
                        alternative["state"]["floor_plus_player_hp_fraction"]
                    ),
                    "paired_difference": float(
                        alternative["state"]["floor_plus_player_hp_fraction"]
                        - base["state"]["floor_plus_player_hp_fraction"]
                    ),
                    "root_stream_alignment_fraction": alignment,
                }
            )
    expected_total = expected_per_character * len(HEADLESS_CHARACTERS)
    by_character: dict[str, dict[str, Any]] = {}
    unpaired_differences: list[float] = []
    for character in HEADLESS_CHARACTERS:
        rows = [row for row in valid if row["character"] == character]
        for index, row in enumerate(rows):
            next_base = rows[(index + 1) % len(rows)]["base_return"] if rows else 0.0
            if len(rows) >= 2:
                unpaired_differences.append(row["alternative_return"] - next_base)
        by_character[character] = {
            "valid_pairs": len(rows),
            "expected_pairs": expected_per_character,
            "valid_fraction": round(len(rows) / expected_per_character, 6),
        }
    paired_differences = [row["paired_difference"] for row in valid]
    paired_variance = _sample_variance(paired_differences)
    unpaired_variance = _sample_variance(unpaired_differences)
    variance_ratio = (
        paired_variance / unpaired_variance
        if paired_variance is not None and unpaired_variance not in (None, 0.0)
        else None
    )
    return {
        "horizon": horizon,
        "expected_pairs": expected_total,
        "valid_pairs": len(valid),
        "valid_fraction": round(len(valid) / expected_total, 6),
        "valid_by_character": by_character,
        "base_repeat_attempts": repeat_attempts,
        "exact_base_repeats": repeat_exact,
        "exact_base_repeat_fraction": (
            round(repeat_exact / repeat_attempts, 6) if repeat_attempts else 0.0
        ),
        "mean_root_stream_alignment_fraction": (
            round(fmean(row["root_stream_alignment_fraction"] for row in valid), 6)
            if valid
            else 0.0
        ),
        "desynchronized_root_stream_counts": dict(sorted(desynchronized.items())),
        "return_variance": {
            "paired_difference": paired_variance,
            "within_character_circularly_unpaired_difference": unpaired_variance,
            "paired_to_unpaired_ratio": variance_ratio,
            "comparison_includes_shared_root_state_and_shared_rng": True,
        },
    }


def apply_horizon_gate(
    summaries: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the precommitted longest-contiguous passing-horizon rule."""

    selected: int | None = None
    evaluated: list[dict[str, Any]] = []
    prefix_open = True
    for summary in sorted(summaries, key=lambda item: int(item["horizon"])):
        ratio = (summary.get("return_variance") or {}).get("paired_to_unpaired_ratio")
        criteria = {
            "valid_pair_fraction": float(summary["valid_fraction"])
            >= float(gate["minimum_valid_pair_fraction"]),
            "minimum_valid_pairs": int(summary["valid_pairs"])
            >= int(gate["minimum_valid_pairs"]),
            "per_character_support": all(
                int(row["valid_pairs"]) >= int(gate["minimum_valid_pairs_per_character"])
                for row in (summary.get("valid_by_character") or {}).values()
            ),
            "exact_base_repeat": float(summary["exact_base_repeat_fraction"])
            >= float(gate["minimum_exact_base_repeat_fraction"]),
            "root_stream_alignment": float(summary["mean_root_stream_alignment_fraction"])
            >= float(gate["minimum_mean_root_stream_alignment_fraction"]),
            "paired_variance_reduction": ratio is not None
            and float(ratio) <= float(gate["maximum_paired_to_unpaired_variance_ratio"]),
        }
        passed = all(criteria.values())
        prefix_open = prefix_open and passed
        if prefix_open:
            selected = int(summary["horizon"])
        evaluated.append(
            {
                "horizon": int(summary["horizon"]),
                "passed": passed,
                "contiguous_prefix_passed": prefix_open,
                "criteria": criteria,
            }
        )
    return {
        "passed": selected is not None,
        "selected_horizon": selected,
        "evaluated": evaluated,
        "decision": (
            "eligible_to_precommit_single_use_counterfactual_dataset"
            if selected is not None
            else str(gate["terminal_decision_if_none_pass"])
        ),
    }


def evaluate_crn_probe(
    *,
    manifest_path: str | os.PathLike[str],
    macro_model_path: str | os.PathLike[str],
    combat_model_path: str | os.PathLike[str],
    output: str | os.PathLike[str],
    parallel_workers: int = 4,
    max_parent_steps: int = 300,
    build_id: str = "v0.107.1",
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Run the frozen probe without emitting counterfactual training records."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("name") != PROBE_NAME:
        raise ValueError("Manifest is not a frozen CRN coupling probe")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Frozen CRN probe output already exists: {destination}")
    target = manifest.get("target") or {}
    if str(target.get("build_id")) != build_id:
        raise ValueError("CRN manifest and evaluator build ids differ")
    effective_config = config or HeadlessConfig()
    runtime_sha = _sha256(effective_config.runtime)
    if str(target.get("runtime_dll_sha256")) != runtime_sha:
        raise ValueError("Headless runtime does not match CRN manifest")
    frozen_base = manifest.get("base_policy") or {}
    macro_sha = _sha256(macro_model_path)
    combat_sha = _sha256(combat_model_path)
    if str(frozen_base.get("macro_model_sha256")) != macro_sha:
        raise ValueError("Macro checkpoint does not match CRN manifest")
    if str(frozen_base.get("combat_model_sha256")) != combat_sha:
        raise ValueError("Combat checkpoint does not match CRN manifest")
    selection = manifest.get("selection") or {}
    count_per_character = int(selection.get("count_per_character") or 0)
    roots_per_case = int(selection.get("roots_per_case") or 0)
    cases = select_benchmark_cases(manifest, count_per_character)
    horizons = [int(value) for value in manifest["measurement"]["horizons_in_policy_decisions"]]
    jobs = [
        HeadlessBatchJob(
            f"{index}:{case['character']}:{case['seed']}",
            partial(
                _run_probe_case,
                character=case["character"],
                seed=case["seed"],
                ascension=int(target.get("ascension") or 10),
                roots_per_case=roots_per_case,
                horizons=horizons,
                macro_model_path=macro_model_path,
                combat_model_path=combat_model_path,
                build_id=build_id,
                max_parent_steps=max_parent_steps,
                config=effective_config,
            ),
        )
        for index, case in enumerate(cases)
    ]
    batch = HeadlessBatchManager(parallel_workers).run(jobs)
    results = list(batch.values())
    expected_per_character = count_per_character * roots_per_case
    summaries = [
        summarize_horizon(results, horizon, expected_per_character=expected_per_character)
        for horizon in horizons
    ]
    gate_result = apply_horizon_gate(summaries, manifest["horizon_selection_gate"])
    all_root_streams: dict[str, dict[str, Any]] = {}
    for case in results:
        for root in case.get("roots") or []:
            all_root_streams.update(root.get("root_rng_streams") or {})
    scope_counts = Counter(
        str(stream.get("scope") or "unknown") for stream in all_root_streams.values()
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "evaluation": "frozen_measurement_only_crn_coupling_feasibility",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "build_id": build_id,
        "models": {"macro_sha256": macro_sha, "combat_sha256": combat_sha},
        "runtime_dll_sha256": runtime_sha,
        "execution": {
            "parallel_workers": parallel_workers,
            "submitted_cases": len(jobs),
            "elapsed_seconds": round(batch.elapsed_seconds, 3),
            "max_parent_steps": max_parent_steps,
        },
        "rng_topology": {
            "kind": "split_named_streams" if scope_counts.get("run", 0) > 1 else "single_stream",
            "stream_count_at_observed_roots": len(all_root_streams),
            "scope_counts": dict(sorted(scope_counts.items())),
            "stream_names": sorted(all_root_streams),
            "chaotic_presentation_rng_excluded_by_native_checkpoint": True,
        },
        "collection": {
            "cases_completed": sum(int(case.get("completed") is True) for case in results),
            "cases_failed": sum(int(case.get("completed") is not True) for case in results),
            "roots_expected": len(cases) * roots_per_case,
            "roots_collected": sum(int(case.get("roots_collected") or 0) for case in results),
        },
        "horizons": summaries,
        "horizon_gate": gate_result,
        "authorization": {
            "counterfactual_dataset": False,
            "training_labels": False,
            "policy_overrides": False,
        },
        "cases": results,
        "limitations": [
            "The floor-plus-HP return is a coupling measurement proxy, not a training target.",
            "The variance comparison combines matched-root control with shared native RNG; the pinned runtime does not expose independent future-RNG resampling at a fixed root.",
            "Passing makes a separately precommitted single-use dataset eligible for design; it does not authorize training data or policy overrides.",
        ],
    }
    _atomic_json(destination, payload, exclusive=True)
    return payload
