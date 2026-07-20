"""Command-line entry points for STSAI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from stsai.dataset import SUPPORTED_DECISION_TYPES, build_card_dataset, build_decision_dataset
from stsai.runs import RunFilter, RunReadError, discover_sources, summarize_runs


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Run file, NDJSON(.gz) snapshot, or directory. Repeatable; defaults to local saves.",
    )
    parser.add_argument("--min-ascension", type=int, default=0)
    parser.add_argument("--include-abandoned", action="store_true")
    parser.add_argument("--include-coop", action="store_true")
    parser.add_argument("--include-cheated", action="store_true")
    parser.add_argument(
        "--build-id",
        action="append",
        default=[],
        help="Only accept this game build id. Repeatable.",
    )


def _filter_from_args(args: argparse.Namespace) -> RunFilter:
    return RunFilter(
        min_ascension=args.min_ascension,
        include_abandoned=args.include_abandoned,
        include_coop=args.include_coop,
        include_cheated=args.include_cheated,
        build_ids=frozenset(args.build_id),
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stsai", description="STS2 AI research tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-runs", help="Summarize run sources")
    _add_source_arguments(inspect_parser)

    build_parser = subparsers.add_parser(
        "build-card-dataset", help="Create anonymized card-reward behavior-cloning NDJSON"
    )
    _add_source_arguments(build_parser)
    build_parser.add_argument(
        "--output", type=Path, required=True, help="Destination .jsonl/.ndjson file"
    )

    decisions_parser = subparsers.add_parser(
        "build-decision-dataset",
        help="Create card, relic, potion, and Ancient choice behavior-cloning NDJSON",
    )
    _add_source_arguments(decisions_parser)
    decisions_parser.add_argument(
        "--decision-type",
        action="append",
        choices=sorted(SUPPORTED_DECISION_TYPES),
        default=[],
        help="Only include this decision type. Repeatable; defaults to all supported types.",
    )
    decisions_parser.add_argument(
        "--output", type=Path, required=True, help="Destination .jsonl/.ndjson file"
    )

    combat_parser = subparsers.add_parser(
        "build-combat-dataset",
        help="Convert recorded Workshop transitions to masked combat decisions",
    )
    combat_parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="Bridge trajectory .jsonl file. Repeatable.",
    )
    combat_parser.add_argument("--output", type=Path, required=True)
    combat_parser.add_argument("--build-id", required=True)

    human_manifest_parser = subparsers.add_parser(
        "create-human-combat-corpus-manifest",
        help="Freeze a balanced first-party Workshop demonstration corpus",
    )
    human_manifest_parser.add_argument("--output", type=Path, required=True)
    human_manifest_parser.add_argument("--game-assembly", type=Path, required=True)
    human_manifest_parser.add_argument("--bridge-dll", type=Path, required=True)
    human_manifest_parser.add_argument("--telemetry-patch", type=Path, required=True)
    human_manifest_parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    human_manifest_parser.add_argument("--build-id", default="v0.107.1")
    human_manifest_parser.add_argument("--ascension", type=int, default=10)
    human_manifest_parser.add_argument("--decisions-per-character", type=int, default=100)

    human_dataset_parser = subparsers.add_parser(
        "build-human-combat-dataset",
        help="Fail closed on provenance and emit balanced train-only UI demonstrations",
    )
    human_dataset_parser.add_argument("--input", action="append", type=Path, required=True)
    human_dataset_parser.add_argument("--manifest", type=Path, required=True)
    human_dataset_parser.add_argument("--output", type=Path, required=True)
    human_dataset_parser.add_argument("--receipt", type=Path, required=True)

    replay_audit_parser = subparsers.add_parser(
        "audit-runreplays",
        help="Audit RunReplays snapshots without creating training data",
    )
    replay_audit_parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="actions.sts2replay file or directory. Repeatable.",
    )
    replay_audit_parser.add_argument("--output", type=Path, required=True)
    replay_audit_parser.add_argument("--expected-build", required=True)
    replay_audit_parser.add_argument("--source-uri")
    replay_audit_parser.add_argument("--source-revision")
    replay_audit_parser.add_argument("--license-spdx")
    replay_audit_parser.add_argument("--license-file", type=Path)

    replay_probe_parser = subparsers.add_parser(
        "probe-runreplays-headless",
        help="Fail-closed headless reconstruction probe for one replay suffix",
    )
    replay_probe_parser.add_argument(
        "--base", type=Path, required=True, help="Earlier actions.sts2replay snapshot"
    )
    replay_probe_parser.add_argument(
        "--target", type=Path, required=True, help="Later cumulative snapshot"
    )
    replay_probe_parser.add_argument("--output", type=Path, required=True)
    replay_probe_parser.add_argument("--expected-build", required=True)
    replay_probe_parser.add_argument("--max-commands", type=int, default=100)
    replay_probe_parser.add_argument(
        "--max-overlap",
        type=int,
        default=0,
        help="Search this many commands backward from the base boundary",
    )
    replay_probe_parser.add_argument(
        "--runtime",
        type=Path,
        help="Override the runtime built by scripts/setup-sts2-cli.ps1",
    )
    replay_probe_parser.add_argument("--startup-timeout", type=float, default=60.0)
    replay_probe_parser.add_argument("--command-timeout", type=float, default=120.0)

    headless_combat_parser = subparsers.add_parser(
        "build-headless-combat-dataset",
        help="Convert native headless teacher trajectories to masked combat decisions",
    )
    headless_combat_parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="Headless trajectory file or directory. Repeatable.",
    )
    headless_combat_parser.add_argument("--output", type=Path, required=True)
    headless_combat_parser.add_argument("--build-id", required=True)
    headless_combat_parser.add_argument("--teacher-source", default="heuristic")

    corpus_manifest_parser = subparsers.add_parser(
        "create-headless-combat-corpus",
        help="Precommit fresh character-balanced heuristic-distillation seeds",
    )
    corpus_manifest_parser.add_argument("--output", type=Path, required=True)
    corpus_manifest_parser.add_argument("--build-id", default="v0.107.1")
    corpus_manifest_parser.add_argument("--ascension", type=int, default=10)
    corpus_manifest_parser.add_argument("--train-per-character", type=int, default=12)
    corpus_manifest_parser.add_argument("--validation-per-character", type=int, default=4)
    corpus_manifest_parser.add_argument("--test-per-character", type=int, default=4)

    corpus_collect_parser = subparsers.add_parser(
        "collect-headless-combat-corpus",
        help="Run the deterministic heuristic teacher on a frozen corpus manifest",
    )
    corpus_collect_parser.add_argument("--manifest", type=Path, required=True)
    corpus_collect_parser.add_argument("--model", type=Path, required=True)
    corpus_collect_parser.add_argument("--trajectory-dir", type=Path, required=True)
    corpus_collect_parser.add_argument("--max-steps", type=int, default=1000)
    corpus_collect_parser.add_argument(
        "--runtime",
        type=Path,
        help="Override the runtime built by scripts/setup-sts2-cli.ps1",
    )
    corpus_collect_parser.add_argument("--startup-timeout", type=float, default=60.0)

    dagger_manifest_parser = subparsers.add_parser(
        "create-headless-dagger-round",
        help="Precommit character-balanced DAgger aggregation and validation seeds",
    )
    dagger_manifest_parser.add_argument("--output", type=Path, required=True)
    dagger_manifest_parser.add_argument("--build-id", default="v0.107.1")
    dagger_manifest_parser.add_argument("--ascension", type=int, default=10)
    dagger_manifest_parser.add_argument("--round", type=int, default=1)
    dagger_manifest_parser.add_argument("--beta", type=float, default=0.5)
    dagger_manifest_parser.add_argument("--aggregate-per-character", type=int, default=20)
    dagger_manifest_parser.add_argument("--validation-per-character", type=int, default=5)
    dagger_manifest_parser.add_argument(
        "--learner-name", default="combat_policy.heuristic.v0.107.1"
    )
    dagger_manifest_parser.add_argument("--learner-model", type=Path)

    dagger_collect_parser = subparsers.add_parser(
        "collect-headless-dagger-round",
        help="Collect beta-mixed learner states with heuristic teacher labels",
    )
    dagger_collect_parser.add_argument("--manifest", type=Path, required=True)
    dagger_collect_parser.add_argument("--macro-model", type=Path, required=True)
    dagger_collect_parser.add_argument("--combat-model", type=Path, required=True)
    dagger_collect_parser.add_argument("--trajectory-dir", type=Path, required=True)
    dagger_collect_parser.add_argument("--output", type=Path, required=True)
    dagger_collect_parser.add_argument("--max-steps", type=int, default=1000)
    dagger_collect_parser.add_argument(
        "--runtime",
        type=Path,
        help="Override the runtime built by scripts/setup-sts2-cli.ps1",
    )
    dagger_collect_parser.add_argument("--startup-timeout", type=float, default=60.0)
    dagger_collect_parser.add_argument("--command-timeout", type=float, default=30.0)

    dagger_dataset_parser = subparsers.add_parser(
        "build-headless-dagger-dataset",
        help="Build teacher-labeled DAgger records while excluding validation seeds",
    )
    dagger_dataset_parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="DAgger trajectory file or directory. Repeatable.",
    )
    dagger_dataset_parser.add_argument("--manifest", type=Path, required=True)
    dagger_dataset_parser.add_argument("--output", type=Path, required=True)
    dagger_dataset_parser.add_argument("--build-id", required=True)

    dagger_merge_parser = subparsers.add_parser(
        "merge-headless-dagger-dataset",
        help="Atomically append DAgger training labels while preserving base splits",
    )
    dagger_merge_parser.add_argument("--base", type=Path, required=True)
    dagger_merge_parser.add_argument("--dagger", type=Path, required=True)
    dagger_merge_parser.add_argument("--output", type=Path, required=True)

    train_parser = subparsers.add_parser(
        "train-card-policy", help="Train the masked NumPy MLP card-choice baseline"
    )
    train_parser.add_argument("--dataset", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--dimension", type=int, default=512)
    train_parser.add_argument("--hidden", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--seed", type=int, default=7)

    evaluate_parser = subparsers.add_parser(
        "evaluate-card-policy", help="Evaluate a trained masked card-choice policy"
    )
    evaluate_parser.add_argument("--dataset", type=Path, required=True)
    evaluate_parser.add_argument("--model", type=Path, required=True)
    evaluate_parser.add_argument("--split", choices=("train", "validation", "test"), default="test")

    combat_train_parser = subparsers.add_parser(
        "train-combat-policy", help="Train a masked character-conditioned combat policy"
    )
    combat_train_parser.add_argument("--dataset", type=Path, required=True)
    combat_train_parser.add_argument("--output", type=Path, required=True)
    combat_train_parser.add_argument("--dimension", type=int, default=1024)
    combat_train_parser.add_argument("--hidden", type=int, default=128)
    combat_train_parser.add_argument("--epochs", type=int, default=20)
    combat_train_parser.add_argument("--batch-size", type=int, default=64)
    combat_train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    combat_train_parser.add_argument("--seed", type=int, default=7)
    combat_train_parser.add_argument("--metrics-output", type=Path)

    combat_evaluate_parser = subparsers.add_parser(
        "evaluate-combat-policy", help="Evaluate a masked combat policy"
    )
    combat_evaluate_parser.add_argument("--dataset", type=Path, required=True)
    combat_evaluate_parser.add_argument("--model", type=Path, required=True)
    combat_evaluate_parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    combat_evaluate_parser.add_argument("--output", type=Path)

    set_train_parser = subparsers.add_parser(
        "train-set-combat-policy",
        help="Train the compact typed Deep Sets combat candidate scorer",
    )
    set_train_parser.add_argument("--dataset", type=Path, required=True)
    set_train_parser.add_argument("--output", type=Path, required=True)
    set_train_parser.add_argument("--metrics-output", type=Path)
    set_train_parser.add_argument("--epochs", type=int, default=12)
    set_train_parser.add_argument("--batch-size", type=int, default=128)
    set_train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    set_train_parser.add_argument("--weight-decay", type=float, default=1e-5)
    set_train_parser.add_argument("--seed", type=int, default=7)

    set_evaluate_parser = subparsers.add_parser(
        "evaluate-set-combat-policy",
        help="Evaluate a typed Deep Sets combat checkpoint",
    )
    set_evaluate_parser.add_argument("--dataset", type=Path, required=True)
    set_evaluate_parser.add_argument("--model", type=Path, required=True)
    set_evaluate_parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    set_evaluate_parser.add_argument("--batch-size", type=int, default=128)
    set_evaluate_parser.add_argument("--output", type=Path)

    value_train_parser = subparsers.add_parser(
        "train-value-model",
        help="Train and calibrate an explainable public-state survival value",
    )
    value_train_parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="First-party headless trajectory file or directory. Repeatable.",
    )
    value_train_parser.add_argument("--output", type=Path, required=True)
    value_train_parser.add_argument("--metrics-output", type=Path)
    value_train_parser.add_argument("--horizon-floors", type=int, default=3)
    value_train_parser.add_argument("--epochs", type=int, default=800)
    value_train_parser.add_argument("--learning-rate", type=float, default=0.03)
    value_train_parser.add_argument("--l2", type=float, default=1e-3)
    value_train_parser.add_argument("--build-id", default="v0.107.1")
    value_train_parser.add_argument(
        "--feature-schema",
        choices=("linear_v1", "character_interactions_v2"),
        default="linear_v1",
    )
    value_train_parser.add_argument(
        "--require-independent-audit",
        action="store_true",
        help="Keep the frozen checkpoint ineligible until a separate audit pool passes",
    )

    value_v4_train_parser = subparsers.add_parser(
        "train-value-model-v4",
        help="Train the one bounded nonlinear ensemble and grouped calibrator selection",
    )
    value_v4_train_parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="First-party development trajectory file or directory. Repeatable.",
    )
    value_v4_train_parser.add_argument("--output", type=Path, required=True)
    value_v4_train_parser.add_argument("--metrics-output", type=Path, required=True)
    value_v4_train_parser.add_argument("--horizon-floors", type=int, default=3)
    value_v4_train_parser.add_argument("--epochs", type=int, default=400)
    value_v4_train_parser.add_argument("--learning-rate", type=float, default=0.02)
    value_v4_train_parser.add_argument("--l2", type=float, default=3e-3)
    value_v4_train_parser.add_argument("--build-id", default="v0.107.1")

    rank_value_train_parser = subparsers.add_parser(
        "train-rank-value-model",
        help="Train a policy-filtered all-action-state value for rank-only search tests",
    )
    rank_value_train_parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="Existing first-party trajectory file or directory. Repeatable.",
    )
    rank_value_train_parser.add_argument("--output", type=Path, required=True)
    rank_value_train_parser.add_argument("--metrics-output", type=Path, required=True)
    rank_value_train_parser.add_argument(
        "--allowed-policy",
        action="append",
        default=[],
        help=(
            "Episode-start policy admitted to the corpus. Repeatable; defaults to the "
            "Deep Sets policy."
        ),
    )
    rank_value_train_parser.add_argument("--horizon-floors", type=int, default=3)
    rank_value_train_parser.add_argument("--epochs", type=int, default=150)
    rank_value_train_parser.add_argument("--learning-rate", type=float, default=0.02)
    rank_value_train_parser.add_argument("--l2", type=float, default=3e-3)
    rank_value_train_parser.add_argument("--build-id", default="v0.107.1")

    value_evaluate_parser = subparsers.add_parser(
        "evaluate-value-model",
        help="Measure held-out calibration of a public-state survival value",
    )
    value_evaluate_parser.add_argument("--input", action="append", type=Path, required=True)
    value_evaluate_parser.add_argument("--model", type=Path, required=True)
    value_evaluate_parser.add_argument(
        "--split", choices=("train", "validation", "test", "all"), default="test"
    )

    value_manifest_parser = subparsers.add_parser(
        "create-value-calibration-manifest",
        help="Precommit disjoint development and blind-audit value seeds",
    )
    value_manifest_parser.add_argument("--output", type=Path, required=True)
    value_manifest_parser.add_argument("--build-id", default="v0.107.1")
    value_manifest_parser.add_argument("--ascension", type=int, default=10)
    value_manifest_parser.add_argument("--development-per-character", type=int, default=15)
    value_manifest_parser.add_argument("--audit-per-character", type=int, default=15)
    value_manifest_parser.add_argument("--seed-prefix", default="VCAL3")
    value_manifest_parser.add_argument("--exclude-manifest", action="append", type=Path, default=[])

    value_collect_parser = subparsers.add_parser(
        "collect-value-calibration-pool",
        help="Collect a frozen value pool without reporting gameplay outcomes",
    )
    value_collect_parser.add_argument("--manifest", type=Path, required=True)
    value_collect_parser.add_argument("--pool", choices=("development", "audit"), required=True)
    value_collect_parser.add_argument("--macro-model", type=Path, required=True)
    value_collect_parser.add_argument("--combat-model", type=Path, required=True)
    value_collect_parser.add_argument("--trajectory-dir", type=Path, required=True)
    value_collect_parser.add_argument("--receipt-output", type=Path)
    value_collect_parser.add_argument("--max-steps", type=int, default=1000)
    value_collect_parser.add_argument(
        "--runtime",
        type=Path,
        help="Override the runtime built by scripts/setup-sts2-cli.ps1",
    )
    value_collect_parser.add_argument("--startup-timeout", type=float, default=60.0)
    value_collect_parser.add_argument("--command-timeout", type=float, default=30.0)

    value_audit_parser = subparsers.add_parser(
        "audit-value-model",
        help="Open an independent audit pool once and apply the frozen promotion gate",
    )
    value_audit_parser.add_argument("--input", action="append", type=Path, required=True)
    value_audit_parser.add_argument("--model", type=Path, required=True)
    value_audit_parser.add_argument("--manifest", type=Path, required=True)
    value_audit_parser.add_argument("--metrics-output", type=Path, required=True)

    replay_audit_parser = subparsers.add_parser(
        "sim-audit-replay",
        help="Measure native replay card-play coverage of a simulator catalog",
    )
    replay_audit_parser.add_argument("--replay", type=Path, required=True)
    replay_audit_parser.add_argument("--catalog", type=Path, required=True)
    replay_audit_parser.add_argument(
        "--hands", action="store_true", help="Also reconstruct every turn's hand"
    )

    headless_parser = subparsers.add_parser(
        "headless-probe",
        help="Start a fixed run in the native headless simulator and print its first decision",
    )
    headless_parser.add_argument(
        "--character",
        choices=("Ironclad", "Silent", "Defect", "Necrobinder", "Regent"),
        default="Silent",
    )
    headless_parser.add_argument("--ascension", type=int, default=10)
    headless_parser.add_argument("--seed", default="STSAI_PROBE")
    headless_parser.add_argument(
        "--runtime",
        type=Path,
        help="Override the runtime built by scripts/setup-sts2-cli.ps1",
    )
    headless_parser.add_argument("--startup-timeout", type=float, default=60.0)

    rollout_parser = subparsers.add_parser(
        "headless-rollout",
        help="Run and record one legal-random episode in the native simulator",
    )
    rollout_parser.add_argument(
        "--character",
        choices=("Ironclad", "Silent", "Defect", "Necrobinder", "Regent"),
        default="Silent",
    )
    rollout_parser.add_argument("--ascension", type=int, default=10)
    rollout_parser.add_argument("--seed", default="STSAI_ROLLOUT")
    rollout_parser.add_argument("--policy-seed", type=int, default=7)
    rollout_parser.add_argument("--max-steps", type=int, default=1000)
    rollout_parser.add_argument("--output", type=Path, required=True)
    rollout_parser.add_argument(
        "--runtime",
        type=Path,
        help="Override the runtime built by scripts/setup-sts2-cli.ps1",
    )
    rollout_parser.add_argument("--startup-timeout", type=float, default=60.0)

    hybrid_parser = subparsers.add_parser(
        "headless-hybrid-rollout",
        help="Run the macro imitation model with labeled deterministic fallbacks",
    )
    hybrid_parser.add_argument("--model", type=Path, required=True)
    hybrid_parser.add_argument(
        "--character",
        choices=("Ironclad", "Silent", "Defect", "Necrobinder", "Regent"),
        default="Silent",
    )
    hybrid_parser.add_argument("--ascension", type=int, default=10)
    hybrid_parser.add_argument("--seed", default="STSAI_ROLLOUT")
    hybrid_parser.add_argument("--max-steps", type=int, default=1000)
    hybrid_parser.add_argument("--build-id", default="v0.107.1")
    hybrid_parser.add_argument("--output", type=Path, required=True)
    hybrid_parser.add_argument(
        "--runtime",
        type=Path,
        help="Override the runtime built by scripts/setup-sts2-cli.ps1",
    )
    hybrid_parser.add_argument("--startup-timeout", type=float, default=60.0)

    headless_eval_parser = subparsers.add_parser(
        "evaluate-headless-policies",
        help="Pair legal-random and hybrid policies on a fixed-seed manifest",
    )
    headless_eval_parser.add_argument("--manifest", type=Path, required=True)
    headless_eval_parser.add_argument("--model", type=Path, required=True)
    headless_eval_parser.add_argument("--output", type=Path, required=True)
    headless_eval_parser.add_argument("--trajectory-dir", type=Path, required=True)
    headless_eval_parser.add_argument("--count-per-character", type=int, default=5)
    headless_eval_parser.add_argument("--random-policy-seed", type=int, default=7)
    headless_eval_parser.add_argument("--max-steps", type=int, default=1000)
    headless_eval_parser.add_argument("--build-id", default="v0.107.1")
    headless_eval_parser.add_argument(
        "--runtime",
        type=Path,
        help="Override the runtime built by scripts/setup-sts2-cli.ps1",
    )
    headless_eval_parser.add_argument("--startup-timeout", type=float, default=60.0)

    combat_headless_eval_parser = subparsers.add_parser(
        "evaluate-headless-combat-distillation",
        help="Pair the heuristic teacher and learned combat policy on frozen seeds",
    )
    combat_headless_eval_parser.add_argument("--manifest", type=Path, required=True)
    combat_headless_eval_parser.add_argument("--macro-model", type=Path, required=True)
    combat_headless_eval_parser.add_argument("--combat-model", type=Path, required=True)
    combat_headless_eval_parser.add_argument("--output", type=Path, required=True)
    combat_headless_eval_parser.add_argument("--trajectory-dir", type=Path, required=True)
    combat_headless_eval_parser.add_argument("--count-per-character", type=int, default=5)
    combat_headless_eval_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum independent native simulator processes to run concurrently",
    )
    combat_headless_eval_parser.add_argument("--max-steps", type=int, default=1000)
    combat_headless_eval_parser.add_argument("--build-id", default="v0.107.1")
    combat_headless_eval_parser.add_argument(
        "--runtime",
        type=Path,
        help="Override the runtime built by scripts/setup-sts2-cli.ps1",
    )
    combat_headless_eval_parser.add_argument("--startup-timeout", type=float, default=60.0)
    combat_headless_eval_parser.add_argument("--command-timeout", type=float, default=30.0)

    combat_pair_parser = subparsers.add_parser(
        "evaluate-headless-combat-models",
        help="Pair two combat checkpoints on a reserved manifest pool",
    )
    combat_pair_parser.add_argument("--manifest", type=Path, required=True)
    combat_pair_parser.add_argument("--macro-model", type=Path, required=True)
    combat_pair_parser.add_argument("--baseline-model", type=Path, required=True)
    combat_pair_parser.add_argument("--candidate-model", type=Path, required=True)
    combat_pair_parser.add_argument("--baseline-kind", choices=("mlp", "set"), default="mlp")
    combat_pair_parser.add_argument("--candidate-kind", choices=("mlp", "set"), default="mlp")
    combat_pair_parser.add_argument("--output", type=Path, required=True)
    combat_pair_parser.add_argument("--trajectory-dir", type=Path, required=True)
    combat_pair_parser.add_argument("--pool", default="validation")
    combat_pair_parser.add_argument("--count-per-character", type=int, default=5)
    combat_pair_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum independent native simulator processes to run concurrently",
    )
    combat_pair_parser.add_argument("--max-steps", type=int, default=1000)
    combat_pair_parser.add_argument("--build-id", default="v0.107.1")
    combat_pair_parser.add_argument("--runtime", type=Path)
    combat_pair_parser.add_argument("--startup-timeout", type=float, default=60.0)
    combat_pair_parser.add_argument("--command-timeout", type=float, default=30.0)

    combat_pair_manifest_parser = subparsers.add_parser(
        "create-headless-combat-model-manifest",
        help="Bind two combat checkpoints to fresh outcome-blind paired seeds",
    )
    combat_pair_manifest_parser.add_argument("--output", type=Path, required=True)
    combat_pair_manifest_parser.add_argument("--baseline-model", type=Path, required=True)
    combat_pair_manifest_parser.add_argument("--candidate-model", type=Path, required=True)
    combat_pair_manifest_parser.add_argument(
        "--baseline-kind", choices=("mlp", "set"), default="mlp"
    )
    combat_pair_manifest_parser.add_argument(
        "--candidate-kind", choices=("mlp", "set"), default="set"
    )
    combat_pair_manifest_parser.add_argument("--seed-prefix", default="SETV1")
    combat_pair_manifest_parser.add_argument("--count-per-character", type=int, default=5)
    combat_pair_manifest_parser.add_argument("--build-id", default="v0.107.1")
    combat_pair_manifest_parser.add_argument("--ascension", type=int, default=10)

    combat_baseline_manifest_parser = subparsers.add_parser(
        "create-headless-combat-baseline-manifest",
        help="Freeze one character-balanced online combat policy baseline",
    )
    combat_baseline_manifest_parser.add_argument("--output", type=Path, required=True)
    combat_baseline_manifest_parser.add_argument("--macro-model", type=Path, required=True)
    combat_baseline_manifest_parser.add_argument("--combat-model", type=Path, required=True)
    combat_baseline_manifest_parser.add_argument(
        "--combat-kind", choices=("mlp", "set"), default="set"
    )
    combat_baseline_manifest_parser.add_argument("--seed-prefix", default="BASEV1")
    combat_baseline_manifest_parser.add_argument("--count-per-character", type=int, default=10)
    combat_baseline_manifest_parser.add_argument(
        "--exclude-manifest", type=Path, action="append", default=[]
    )
    combat_baseline_manifest_parser.add_argument("--build-id", default="v0.107.1")
    combat_baseline_manifest_parser.add_argument("--ascension", type=int, default=10)

    combat_baseline_parser = subparsers.add_parser(
        "evaluate-headless-combat-baseline",
        help="Run every case in a frozen online combat policy baseline",
    )
    combat_baseline_parser.add_argument("--manifest", type=Path, required=True)
    combat_baseline_parser.add_argument("--macro-model", type=Path, required=True)
    combat_baseline_parser.add_argument("--combat-model", type=Path, required=True)
    combat_baseline_parser.add_argument("--output", type=Path, required=True)
    combat_baseline_parser.add_argument("--trajectory-dir", type=Path, required=True)
    combat_baseline_parser.add_argument("--max-steps", type=int, default=1000)
    combat_baseline_parser.add_argument("--build-id", default="v0.107.1")
    combat_baseline_parser.add_argument("--runtime", type=Path)
    combat_baseline_parser.add_argument("--startup-timeout", type=float, default=60.0)
    combat_baseline_parser.add_argument("--command-timeout", type=float, default=30.0)

    chip_manifest_parser = subparsers.add_parser(
        "create-chip-aware-heuristic-manifest",
        help="Freeze a paired current-versus-chip-aware tactical teacher screen",
    )
    chip_manifest_parser.add_argument("--output", type=Path, required=True)
    chip_manifest_parser.add_argument("--macro-model", type=Path, required=True)
    chip_manifest_parser.add_argument("--seed-prefix", default="TACTV1")
    chip_manifest_parser.add_argument("--count-per-character", type=int, default=10)
    chip_manifest_parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    chip_manifest_parser.add_argument("--build-id", default="v0.107.1")
    chip_manifest_parser.add_argument("--ascension", type=int, default=10)

    chip_evaluate_parser = subparsers.add_parser(
        "evaluate-chip-aware-heuristic",
        help="Run every frozen seed with both tactical teacher versions",
    )
    chip_evaluate_parser.add_argument("--manifest", type=Path, required=True)
    chip_evaluate_parser.add_argument("--macro-model", type=Path, required=True)
    chip_evaluate_parser.add_argument("--output", type=Path, required=True)
    chip_evaluate_parser.add_argument("--trajectory-dir", type=Path, required=True)
    chip_evaluate_parser.add_argument("--max-steps", type=int, default=1000)
    chip_evaluate_parser.add_argument("--build-id", default="v0.107.1")
    chip_evaluate_parser.add_argument("--runtime", type=Path)
    chip_evaluate_parser.add_argument("--startup-timeout", type=float, default=60.0)
    chip_evaluate_parser.add_argument("--command-timeout", type=float, default=30.0)

    rank_search_manifest_parser = subparsers.add_parser(
        "create-headless-rank-search-manifest",
        help="Freeze a no-label Deep Sets versus rank-only one-ply search comparison",
    )
    rank_search_manifest_parser.add_argument("--output", type=Path, required=True)
    rank_search_manifest_parser.add_argument("--macro-model", type=Path, required=True)
    rank_search_manifest_parser.add_argument("--combat-model", type=Path, required=True)
    rank_search_manifest_parser.add_argument("--value-model", type=Path, required=True)
    rank_search_manifest_parser.add_argument("--seed-prefix", required=True)
    rank_search_manifest_parser.add_argument("--count-per-character", type=int, required=True)
    rank_search_manifest_parser.add_argument("--max-nodes", type=int, default=8)
    rank_search_manifest_parser.add_argument("--parallel-branches", type=int, default=1)
    rank_search_manifest_parser.add_argument(
        "--exclude-manifest", type=Path, action="append", default=[]
    )
    rank_search_manifest_parser.add_argument("--build-id", default="v0.107.1")
    rank_search_manifest_parser.add_argument("--ascension", type=int, default=10)

    rank_search_parser = subparsers.add_parser(
        "evaluate-headless-rank-search",
        help="Pair base Deep Sets and rank-only one-ply search on frozen seeds",
    )
    rank_search_parser.add_argument("--manifest", type=Path, required=True)
    rank_search_parser.add_argument("--macro-model", type=Path, required=True)
    rank_search_parser.add_argument("--combat-model", type=Path, required=True)
    rank_search_parser.add_argument("--value-model", type=Path, required=True)
    rank_search_parser.add_argument("--output", type=Path, required=True)
    rank_search_parser.add_argument("--trajectory-dir", type=Path, required=True)
    rank_search_parser.add_argument("--workers", type=int, default=4)
    rank_search_parser.add_argument("--max-steps", type=int, default=1000)
    rank_search_parser.add_argument("--build-id", default="v0.107.1")
    rank_search_parser.add_argument("--runtime", type=Path)
    rank_search_parser.add_argument("--startup-timeout", type=float, default=60.0)
    rank_search_parser.add_argument("--command-timeout", type=float, default=30.0)

    crn_manifest_parser = subparsers.add_parser(
        "create-crn-coupling-manifest",
        help="Freeze a measurement-only native counterfactual RNG-coupling probe",
    )
    crn_manifest_parser.add_argument("--output", type=Path, required=True)
    crn_manifest_parser.add_argument("--macro-model", type=Path, required=True)
    crn_manifest_parser.add_argument("--combat-model", type=Path, required=True)
    crn_manifest_parser.add_argument("--seed-prefix", default="CRN1")
    crn_manifest_parser.add_argument("--count-per-character", type=int, default=5)
    crn_manifest_parser.add_argument("--roots-per-case", type=int, default=2)
    crn_manifest_parser.add_argument(
        "--horizon", type=int, action="append", default=None
    )
    crn_manifest_parser.add_argument("--build-id", default="v0.107.1")
    crn_manifest_parser.add_argument("--ascension", type=int, default=10)

    crn_evaluate_parser = subparsers.add_parser(
        "evaluate-crn-coupling",
        help="Measure paired RNG-stream coupling at every frozen rollout horizon",
    )
    crn_evaluate_parser.add_argument("--manifest", type=Path, required=True)
    crn_evaluate_parser.add_argument("--macro-model", type=Path, required=True)
    crn_evaluate_parser.add_argument("--combat-model", type=Path, required=True)
    crn_evaluate_parser.add_argument("--output", type=Path, required=True)
    crn_evaluate_parser.add_argument("--workers", type=int, default=4)
    crn_evaluate_parser.add_argument("--max-parent-steps", type=int, default=300)
    crn_evaluate_parser.add_argument("--build-id", default="v0.107.1")
    crn_evaluate_parser.add_argument("--runtime", type=Path)
    crn_evaluate_parser.add_argument("--startup-timeout", type=float, default=60.0)
    crn_evaluate_parser.add_argument("--command-timeout", type=float, default=120.0)

    bridge_parser = subparsers.add_parser(
        "bridge-state", help="Read the current state from a local STS2MCP mod"
    )
    bridge_parser.add_argument("--url", default="http://127.0.0.1:15526")
    bridge_parser.add_argument(
        "--mode", choices=("singleplayer", "multiplayer"), default="singleplayer"
    )

    replay_parser = subparsers.add_parser(
        "bridge-replay",
        help="Export the latest native combat replay through the running mod",
    )
    replay_parser.add_argument("--url", default="http://127.0.0.1:15526")
    replay_parser.add_argument("--output", type=Path, help="Optional destination JSON file")

    act_parser = subparsers.add_parser(
        "bridge-act", help="Perform one explicit bridge action and record its transition"
    )
    act_parser.add_argument("--url", default="http://127.0.0.1:15526")
    act_parser.add_argument(
        "--mode", choices=("singleplayer", "multiplayer"), default="singleplayer"
    )
    act_parser.add_argument(
        "--action", required=True, help='JSON object, e.g. {"action":"end_turn"}'
    )
    act_parser.add_argument("--trajectory", type=Path, required=True)
    act_parser.add_argument("--settle-timeout", type=float, default=3.0)

    recommend_parser = subparsers.add_parser(
        "agent-recommend", help="Recommend one supported macro action without changing the game"
    )
    recommend_parser.add_argument("--model", type=Path, required=True)
    recommend_parser.add_argument("--url", default="http://127.0.0.1:15526")
    recommend_parser.add_argument("--build-id")
    recommend_parser.add_argument("--save-root", type=Path)
    recommend_parser.add_argument(
        "--mode", choices=("singleplayer", "multiplayer"), default="singleplayer"
    )

    step_parser = subparsers.add_parser(
        "agent-step", help="Execute and record one learned macro action in the modded game"
    )
    step_parser.add_argument("--model", type=Path, required=True)
    step_parser.add_argument("--trajectory", type=Path, required=True)
    step_parser.add_argument("--url", default="http://127.0.0.1:15526")
    step_parser.add_argument("--build-id")
    step_parser.add_argument("--save-root", type=Path)
    step_parser.add_argument("--settle-timeout", type=float, default=3.0)
    step_parser.add_argument(
        "--mode", choices=("singleplayer", "multiplayer"), default="singleplayer"
    )

    combat_recommend_parser = subparsers.add_parser(
        "combat-recommend", help="Recommend one legal combat action without changing the game"
    )
    combat_recommend_parser.add_argument("--model", type=Path, required=True)
    combat_recommend_parser.add_argument("--build-id", required=True)
    combat_recommend_parser.add_argument("--url", default="http://127.0.0.1:15526")
    combat_recommend_parser.add_argument(
        "--mode", choices=("singleplayer", "multiplayer"), default="singleplayer"
    )

    combat_step_parser = subparsers.add_parser(
        "combat-step", help="Execute and record one learned, legally masked combat action"
    )
    combat_step_parser.add_argument("--model", type=Path, required=True)
    combat_step_parser.add_argument("--build-id", required=True)
    combat_step_parser.add_argument("--trajectory", type=Path, required=True)
    combat_step_parser.add_argument("--url", default="http://127.0.0.1:15526")
    combat_step_parser.add_argument("--settle-timeout", type=float, default=3.0)
    combat_step_parser.add_argument(
        "--mode", choices=("singleplayer", "multiplayer"), default="singleplayer"
    )

    benchmark_parser = subparsers.add_parser(
        "create-benchmark", help="Create a character-balanced fixed-seed manifest"
    )
    benchmark_parser.add_argument("--input", action="append", default=[])
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--build-id", required=True)
    benchmark_parser.add_argument("--source-build-id")
    benchmark_parser.add_argument("--ascension", type=int, default=10)
    benchmark_parser.add_argument("--count-per-character", type=int, default=20)

    score_parser = subparsers.add_parser(
        "evaluate-benchmark", help="Score first attempts against a fixed-seed manifest"
    )
    score_parser.add_argument("--input", action="append", default=[])
    score_parser.add_argument("--manifest", type=Path, required=True)
    return parser


def _resolve_sources(values: Sequence[str]) -> list[Path]:
    sources = discover_sources(values or None)
    if not sources:
        shown = ", ".join(values) if values else "the default STS2 save directory"
        raise FileNotFoundError(f"No supported run files found under {shown}")
    return sources


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect-runs":
            sources = _resolve_sources(args.input)
            run_filter = _filter_from_args(args)
            result = summarize_runs(sources, run_filter)
        elif args.command in {"build-card-dataset", "build-decision-dataset"}:
            sources = _resolve_sources(args.input)
            run_filter = _filter_from_args(args)
            if args.command == "build-card-dataset":
                result = build_card_dataset(sources, args.output, run_filter)
            else:
                types = frozenset(args.decision_type) or SUPPORTED_DECISION_TYPES
                result = build_decision_dataset(
                    sources, args.output, run_filter, decision_types=types
                )
        elif args.command == "build-combat-dataset":
            from stsai.combat import build_combat_dataset

            result = build_combat_dataset(args.input, args.output, build_id=args.build_id)
        elif args.command == "create-human-combat-corpus-manifest":
            from stsai.human_telemetry import create_human_combat_corpus_manifest

            payload = create_human_combat_corpus_manifest(
                args.output,
                game_assembly_path=args.game_assembly,
                bridge_dll_path=args.bridge_dll,
                telemetry_patch_path=args.telemetry_patch,
                excluded_manifests=args.exclude_manifest,
                build_id=args.build_id,
                ascension=args.ascension,
                decisions_per_character=args.decisions_per_character,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "build-human-combat-dataset":
            from stsai.human_telemetry import build_human_combat_dataset

            payload = build_human_combat_dataset(
                args.input,
                args.output,
                manifest_path=args.manifest,
                receipt_output=args.receipt,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "audit-runreplays":
            from stsai.runreplays import audit_runreplays

            payload = audit_runreplays(
                args.input,
                args.output,
                expected_build=args.expected_build,
                source_uri=args.source_uri,
                source_revision=args.source_revision,
                license_spdx=args.license_spdx,
                license_file=args.license_file,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "probe-runreplays-headless":
            from stsai.headless import HeadlessConfig
            from stsai.runreplays import probe_runreplays_reconstruction

            defaults = HeadlessConfig()
            config = HeadlessConfig(
                runtime=args.runtime or defaults.runtime,
                startup_timeout=args.startup_timeout,
                command_timeout=args.command_timeout,
            )
            payload = probe_runreplays_reconstruction(
                args.base,
                args.target,
                args.output,
                expected_build=args.expected_build,
                max_commands=args.max_commands,
                max_overlap=args.max_overlap,
                config=config,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "build-headless-combat-dataset":
            from stsai.combat import build_headless_combat_dataset

            result = build_headless_combat_dataset(
                args.input,
                args.output,
                build_id=args.build_id,
                feature_schema=args.feature_schema,
                teacher_source=args.teacher_source,
            )
        elif args.command == "create-headless-combat-corpus":
            from stsai.headless_corpus import create_combat_bootstrap_manifest

            payload = create_combat_bootstrap_manifest(
                args.output,
                build_id=args.build_id,
                ascension=args.ascension,
                train_per_character=args.train_per_character,
                validation_per_character=args.validation_per_character,
                test_per_character=args.test_per_character,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "collect-headless-combat-corpus":
            from stsai.headless import HeadlessConfig
            from stsai.headless_corpus import collect_combat_bootstrap_corpus

            config = HeadlessConfig(
                runtime=args.runtime or HeadlessConfig().runtime,
                startup_timeout=args.startup_timeout,
            )
            payload = collect_combat_bootstrap_corpus(
                args.manifest,
                args.trajectory_dir,
                macro_model_path=args.model,
                max_steps=args.max_steps,
                config=config,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "create-headless-dagger-round":
            from stsai.dagger import create_dagger_manifest

            payload = create_dagger_manifest(
                args.output,
                build_id=args.build_id,
                ascension=args.ascension,
                round_number=args.round,
                beta=args.beta,
                aggregate_per_character=args.aggregate_per_character,
                validation_per_character=args.validation_per_character,
                learner_name=args.learner_name,
                learner_model_path=args.learner_model,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "collect-headless-dagger-round":
            from stsai.dagger import collect_dagger_round
            from stsai.headless import HeadlessConfig

            config = HeadlessConfig(
                runtime=args.runtime or HeadlessConfig().runtime,
                startup_timeout=args.startup_timeout,
                command_timeout=args.command_timeout,
            )
            payload = collect_dagger_round(
                args.manifest,
                args.trajectory_dir,
                macro_model_path=args.macro_model,
                combat_model_path=args.combat_model,
                output=args.output,
                max_steps=args.max_steps,
                config=config,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "build-headless-dagger-dataset":
            from stsai.dagger import build_dagger_dataset

            result = build_dagger_dataset(
                args.input,
                args.output,
                manifest_path=args.manifest,
                build_id=args.build_id,
            )
        elif args.command == "merge-headless-dagger-dataset":
            from stsai.dagger import merge_dagger_training_dataset

            payload = merge_dagger_training_dataset(args.base, args.dagger, args.output)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "train-card-policy":
            from stsai.policy import train_policy

            payload = train_policy(
                args.dataset,
                args.output,
                dimension=args.dimension,
                hidden=args.hidden,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "evaluate-card-policy":
            from stsai.policy import evaluate_policy

            payload = evaluate_policy(args.dataset, args.model, args.split)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "train-combat-policy":
            from stsai.combat_policy import train_combat_policy

            payload = train_combat_policy(
                args.dataset,
                args.output,
                dimension=args.dimension,
                hidden=args.hidden,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                metrics_output=args.metrics_output,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "evaluate-combat-policy":
            from stsai.combat_policy import evaluate_combat_policy

            payload = evaluate_combat_policy(
                args.dataset, args.model, args.split, output=args.output
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "train-set-combat-policy":
            from stsai.set_policy import train_set_policy

            payload = train_set_policy(
                args.dataset,
                args.output,
                metrics_output=args.metrics_output,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=args.seed,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "evaluate-set-combat-policy":
            from stsai.set_policy import evaluate_set_policy

            payload = evaluate_set_policy(
                args.dataset,
                args.model,
                split=args.split,
                batch_size=args.batch_size,
                output=args.output,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "train-value-model":
            from stsai.value import train_value_model

            payload = train_value_model(
                args.input,
                args.output,
                metrics_output=args.metrics_output,
                horizon_floors=args.horizon_floors,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                l2=args.l2,
                build_id=args.build_id,
                feature_schema=args.feature_schema,
                require_independent_audit=args.require_independent_audit,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "train-value-model-v4":
            from stsai.value import train_value_model_v4

            payload = train_value_model_v4(
                args.input,
                args.output,
                metrics_output=args.metrics_output,
                horizon_floors=args.horizon_floors,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                l2=args.l2,
                build_id=args.build_id,
                require_independent_audit=True,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "train-rank-value-model":
            from stsai.value import train_rank_value_model

            payload = train_rank_value_model(
                args.input,
                args.output,
                metrics_output=args.metrics_output,
                allowed_policies=(
                    args.allowed_policy or ["combat_deepsets_greedy_noncombat_v1"]
                ),
                horizon_floors=args.horizon_floors,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                l2=args.l2,
                build_id=args.build_id,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "evaluate-value-model":
            from stsai.value import evaluate_value_model

            payload = evaluate_value_model(args.input, args.model, split=args.split)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "create-value-calibration-manifest":
            from stsai.value_corpus import create_value_calibration_manifest

            payload = create_value_calibration_manifest(
                args.output,
                build_id=args.build_id,
                ascension=args.ascension,
                development_per_character=args.development_per_character,
                audit_per_character=args.audit_per_character,
                seed_prefix=args.seed_prefix,
                excluded_manifests=args.exclude_manifest,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "collect-value-calibration-pool":
            from stsai.headless import HeadlessConfig
            from stsai.value_corpus import collect_value_calibration_pool

            config = HeadlessConfig(
                runtime=args.runtime or HeadlessConfig().runtime,
                startup_timeout=args.startup_timeout,
                command_timeout=args.command_timeout,
            )
            payload = collect_value_calibration_pool(
                args.manifest,
                args.pool,
                args.trajectory_dir,
                macro_model_path=args.macro_model,
                combat_model_path=args.combat_model,
                receipt_output=args.receipt_output,
                max_steps=args.max_steps,
                config=config,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "audit-value-model":
            from stsai.value import audit_value_model

            payload = audit_value_model(
                args.input,
                args.model,
                args.manifest,
                metrics_output=args.metrics_output,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "sim-audit-replay":
            from stsai.simulator import (
                CardCatalog,
                audit_replay_coverage,
                audit_replay_hands,
            )

            catalog = CardCatalog.from_path(args.catalog)
            payload = (
                audit_replay_hands(args.replay, catalog)
                if args.hands
                else audit_replay_coverage(args.replay, catalog)
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "headless-probe":
            from stsai.headless import HeadlessConfig, Sts2HeadlessClient

            if not 0 <= args.ascension <= 10:
                raise ValueError("--ascension must be between 0 and 10")
            config = HeadlessConfig(
                runtime=args.runtime or HeadlessConfig().runtime,
                startup_timeout=args.startup_timeout,
            )
            with Sts2HeadlessClient(config) as client:
                opening = client.start_run(
                    character=args.character,
                    ascension=args.ascension,
                    seed=args.seed,
                )
                payload = {
                    "protocol_version": client.version,
                    "character": args.character,
                    "ascension": args.ascension,
                    "seed": args.seed,
                    "opening": opening,
                }
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
            return 0
        elif args.command == "headless-rollout":
            from stsai.headless import HeadlessConfig
            from stsai.headless_env import run_random_rollout

            config = HeadlessConfig(
                runtime=args.runtime or HeadlessConfig().runtime,
                startup_timeout=args.startup_timeout,
            )
            payload = run_random_rollout(
                output=args.output,
                character=args.character,
                ascension=args.ascension,
                seed=args.seed,
                policy_seed=args.policy_seed,
                max_steps=args.max_steps,
                config=config,
            )
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
            return 0
        elif args.command == "headless-hybrid-rollout":
            from stsai.headless import HeadlessConfig
            from stsai.headless_policy import run_hybrid_rollout

            config = HeadlessConfig(
                runtime=args.runtime or HeadlessConfig().runtime,
                startup_timeout=args.startup_timeout,
            )
            payload = run_hybrid_rollout(
                model_path=args.model,
                output=args.output,
                character=args.character,
                ascension=args.ascension,
                seed=args.seed,
                max_steps=args.max_steps,
                build_id=args.build_id,
                config=config,
            )
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
            return 0
        elif args.command == "evaluate-headless-policies":
            from stsai.headless import HeadlessConfig
            from stsai.headless_benchmark import evaluate_hybrid_vs_random

            config = HeadlessConfig(
                runtime=args.runtime or HeadlessConfig().runtime,
                startup_timeout=args.startup_timeout,
            )
            payload = evaluate_hybrid_vs_random(
                manifest_path=args.manifest,
                model_path=args.model,
                output=args.output,
                trajectory_dir=args.trajectory_dir,
                count_per_character=args.count_per_character,
                random_policy_seed=args.random_policy_seed,
                max_steps=args.max_steps,
                build_id=args.build_id,
                config=config,
            )
            print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
            return 0
        elif args.command == "evaluate-headless-combat-distillation":
            from stsai.headless import HeadlessConfig
            from stsai.headless_benchmark import evaluate_combat_distillation

            config = HeadlessConfig(
                runtime=args.runtime or HeadlessConfig().runtime,
                startup_timeout=args.startup_timeout,
                command_timeout=args.command_timeout,
            )
            payload = evaluate_combat_distillation(
                manifest_path=args.manifest,
                macro_model_path=args.macro_model,
                combat_model_path=args.combat_model,
                output=args.output,
                trajectory_dir=args.trajectory_dir,
                count_per_character=args.count_per_character,
                parallel_workers=args.workers,
                max_steps=args.max_steps,
                build_id=args.build_id,
                config=config,
            )
            print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
            return 0
        elif args.command == "evaluate-headless-combat-models":
            from stsai.headless import HeadlessConfig
            from stsai.headless_benchmark import evaluate_combat_model_pair

            config = HeadlessConfig(
                runtime=args.runtime or HeadlessConfig().runtime,
                startup_timeout=args.startup_timeout,
                command_timeout=args.command_timeout,
            )
            payload = evaluate_combat_model_pair(
                manifest_path=args.manifest,
                macro_model_path=args.macro_model,
                baseline_model_path=args.baseline_model,
                candidate_model_path=args.candidate_model,
                baseline_kind=args.baseline_kind,
                candidate_kind=args.candidate_kind,
                output=args.output,
                trajectory_dir=args.trajectory_dir,
                pool=args.pool,
                count_per_character=args.count_per_character,
                parallel_workers=args.workers,
                max_steps=args.max_steps,
                build_id=args.build_id,
                config=config,
            )
            print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
            return 0
        elif args.command == "create-headless-combat-model-manifest":
            from stsai.headless_benchmark import create_combat_model_pair_manifest

            payload = create_combat_model_pair_manifest(
                args.output,
                baseline_model_path=args.baseline_model,
                candidate_model_path=args.candidate_model,
                baseline_kind=args.baseline_kind,
                candidate_kind=args.candidate_kind,
                seed_prefix=args.seed_prefix,
                count_per_character=args.count_per_character,
                build_id=args.build_id,
                ascension=args.ascension,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "create-headless-combat-baseline-manifest":
            from stsai.headless_benchmark import create_combat_policy_baseline_manifest

            payload = create_combat_policy_baseline_manifest(
                args.output,
                macro_model_path=args.macro_model,
                combat_model_path=args.combat_model,
                combat_kind=args.combat_kind,
                seed_prefix=args.seed_prefix,
                count_per_character=args.count_per_character,
                excluded_manifests=args.exclude_manifest,
                build_id=args.build_id,
                ascension=args.ascension,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "evaluate-headless-combat-baseline":
            from stsai.headless import HeadlessConfig
            from stsai.headless_benchmark import evaluate_combat_policy_baseline

            config = HeadlessConfig(
                runtime=args.runtime or HeadlessConfig().runtime,
                startup_timeout=args.startup_timeout,
                command_timeout=args.command_timeout,
            )
            payload = evaluate_combat_policy_baseline(
                manifest_path=args.manifest,
                macro_model_path=args.macro_model,
                combat_model_path=args.combat_model,
                output=args.output,
                trajectory_dir=args.trajectory_dir,
                max_steps=args.max_steps,
                build_id=args.build_id,
                config=config,
            )
            print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
            return 0
        elif args.command == "create-chip-aware-heuristic-manifest":
            from stsai.headless_benchmark import create_chip_aware_heuristic_manifest

            payload = create_chip_aware_heuristic_manifest(
                args.output,
                macro_model_path=args.macro_model,
                seed_prefix=args.seed_prefix,
                count_per_character=args.count_per_character,
                excluded_manifests=args.exclude_manifest,
                build_id=args.build_id,
                ascension=args.ascension,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "evaluate-chip-aware-heuristic":
            from stsai.headless import HeadlessConfig
            from stsai.headless_benchmark import evaluate_chip_aware_heuristic_pair

            config = HeadlessConfig(
                runtime=args.runtime or HeadlessConfig().runtime,
                startup_timeout=args.startup_timeout,
                command_timeout=args.command_timeout,
            )
            payload = evaluate_chip_aware_heuristic_pair(
                manifest_path=args.manifest,
                macro_model_path=args.macro_model,
                output=args.output,
                trajectory_dir=args.trajectory_dir,
                max_steps=args.max_steps,
                build_id=args.build_id,
                config=config,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "create-headless-rank-search-manifest":
            from stsai.search_benchmark import create_rank_search_manifest

            payload = create_rank_search_manifest(
                args.output,
                macro_model_path=args.macro_model,
                combat_model_path=args.combat_model,
                value_model_path=args.value_model,
                seed_prefix=args.seed_prefix,
                count_per_character=args.count_per_character,
                max_nodes=args.max_nodes,
                parallel_branches=args.parallel_branches,
                excluded_manifests=args.exclude_manifest,
                build_id=args.build_id,
                ascension=args.ascension,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "evaluate-headless-rank-search":
            from stsai.headless import HeadlessConfig
            from stsai.search_benchmark import evaluate_rank_search_pair

            config = HeadlessConfig(
                runtime=args.runtime or HeadlessConfig().runtime,
                startup_timeout=args.startup_timeout,
                command_timeout=args.command_timeout,
            )
            payload = evaluate_rank_search_pair(
                manifest_path=args.manifest,
                macro_model_path=args.macro_model,
                combat_model_path=args.combat_model,
                value_model_path=args.value_model,
                output=args.output,
                trajectory_dir=args.trajectory_dir,
                parallel_workers=args.workers,
                max_steps=args.max_steps,
                build_id=args.build_id,
                config=config,
            )
            print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
            print(json.dumps(payload["search_coverage"], indent=2, sort_keys=True))
            return 0
        elif args.command == "create-crn-coupling-manifest":
            from stsai.crn_probe import create_crn_probe_manifest

            payload = create_crn_probe_manifest(
                args.output,
                macro_model_path=args.macro_model,
                combat_model_path=args.combat_model,
                seed_prefix=args.seed_prefix,
                count_per_character=args.count_per_character,
                roots_per_case=args.roots_per_case,
                horizons=args.horizon or (1, 2, 4, 8, 16),
                build_id=args.build_id,
                ascension=args.ascension,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        elif args.command == "evaluate-crn-coupling":
            from stsai.crn_probe import evaluate_crn_probe
            from stsai.headless import HeadlessConfig

            config = HeadlessConfig(
                runtime=args.runtime or HeadlessConfig().runtime,
                startup_timeout=args.startup_timeout,
                command_timeout=args.command_timeout,
            )
            payload = evaluate_crn_probe(
                manifest_path=args.manifest,
                macro_model_path=args.macro_model,
                combat_model_path=args.combat_model,
                output=args.output,
                parallel_workers=args.workers,
                max_parent_steps=args.max_parent_steps,
                build_id=args.build_id,
                config=config,
            )
            print(json.dumps(payload["rng_topology"], indent=2, sort_keys=True))
            print(json.dumps(payload["horizons"], indent=2, sort_keys=True))
            print(json.dumps(payload["horizon_gate"], indent=2, sort_keys=True))
            return 0
        elif args.command == "bridge-state":
            from stsai.bridge import BridgeConfig, STS2MCPClient

            client = STS2MCPClient(BridgeConfig(args.url))
            print(json.dumps(client.get_state(args.mode), indent=2, sort_keys=True))
            return 0
        elif args.command == "bridge-replay":
            from stsai.bridge import BridgeConfig, STS2MCPClient

            client = STS2MCPClient(BridgeConfig(args.url))
            payload = client.get_latest_replay()
            encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                temporary.write_text(encoded + "\n", encoding="utf-8", newline="\n")
                temporary.replace(args.output)
            print(encoded)
            return 0
        elif args.command == "bridge-act":
            from stsai.bridge import BridgeConfig, STS2MCPClient, TransitionRecorder

            action = json.loads(args.action)
            if not isinstance(action, dict):
                raise ValueError("--action must be a JSON object")
            client = STS2MCPClient(BridgeConfig(args.url))
            recorder = TransitionRecorder(client, args.trajectory, args.mode, args.settle_timeout)
            event = recorder.act(action)
            print(json.dumps(event, indent=2, sort_keys=True))
            return 0
        elif args.command == "agent-recommend":
            from stsai.bridge import BridgeConfig, STS2MCPClient
            from stsai.live import MacroPolicyAgent

            client = STS2MCPClient(BridgeConfig(args.url))
            agent = MacroPolicyAgent.from_path(
                args.model, build_id=args.build_id, save_root=args.save_root
            )
            recommendation = agent.recommend(client.get_state(args.mode))
            print(json.dumps(recommendation, indent=2, sort_keys=True))
            return 0
        elif args.command == "agent-step":
            from stsai.bridge import BridgeConfig, STS2MCPClient, TransitionRecorder
            from stsai.live import MacroPolicyAgent

            client = STS2MCPClient(BridgeConfig(args.url))
            agent = MacroPolicyAgent.from_path(
                args.model, build_id=args.build_id, save_root=args.save_root
            )
            recommendation = agent.recommend(client.get_state(args.mode))
            recorder = TransitionRecorder(client, args.trajectory, args.mode, args.settle_timeout)
            event = recorder.act(recommendation["action"])
            print(
                json.dumps(
                    {
                        "recommendation": recommendation,
                        "state_changed": event["state_changed"],
                        "terminal": event["terminal"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        elif args.command in {"combat-recommend", "combat-step"}:
            from stsai.bridge import BridgeConfig, STS2MCPClient, TransitionRecorder
            from stsai.combat_policy import CombatPolicyAgent

            client = STS2MCPClient(BridgeConfig(args.url))
            agent = CombatPolicyAgent.from_path(args.model, build_id=args.build_id)
            recommendation = agent.recommend(client.get_state(args.mode))
            if args.command == "combat-recommend":
                print(json.dumps(recommendation, indent=2, sort_keys=True))
                return 0
            recorder = TransitionRecorder(client, args.trajectory, args.mode, args.settle_timeout)
            event = recorder.act(recommendation["action"])
            print(
                json.dumps(
                    {
                        "recommendation": recommendation,
                        "state_changed": event["state_changed"],
                        "terminal": event["terminal"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        elif args.command == "create-benchmark":
            from stsai.benchmark import create_benchmark

            sources = _resolve_sources(args.input)
            payload = create_benchmark(
                sources,
                args.output,
                build_id=args.build_id,
                ascension=args.ascension,
                count_per_character=args.count_per_character,
                source_build_id=args.source_build_id,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        else:
            from stsai.benchmark import evaluate_benchmark

            sources = _resolve_sources(args.input)
            payload = evaluate_benchmark(args.manifest, sources)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, RunReadError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2
