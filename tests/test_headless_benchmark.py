from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from stsai.headless_benchmark import (
    _discard_matching_incomplete,
    _publish_runtime_failure,
    aggregate_combat_policy_baseline,
    aggregate_combat_model_pair,
    aggregate_combat_distillation,
    aggregate_paired_results,
    create_chip_aware_heuristic_manifest,
    create_combat_model_pair_manifest,
    create_combat_policy_baseline_manifest,
    select_benchmark_cases,
    select_benchmark_pool_cases,
)
from stsai.headless_env import HEADLESS_CHARACTERS


class HeadlessBenchmarkTests(unittest.TestCase):
    def test_case_selection_uses_manifest_order_without_outcomes(self) -> None:
        cases = []
        for character in reversed(HEADLESS_CHARACTERS):
            for index in range(3):
                cases.append(
                    {
                        "character": f"CHARACTER.{character.upper()}",
                        "seed": f"{character}-{index}",
                        "outcome_that_must_not_matter": index == 2,
                    }
                )
        selected = select_benchmark_cases({"cases": cases}, 2)
        self.assertEqual(len(selected), 10)
        self.assertEqual(
            [case["seed"] for case in selected[:2]],
            ["Regent-0", "Regent-1"],
        )
        self.assertNotIn("outcome_that_must_not_matter", selected[0])

    def test_pool_selection_excludes_aggregation_cases(self) -> None:
        cases = []
        for character in HEADLESS_CHARACTERS:
            cases.append(
                {"character": character, "seed": f"{character}-train", "pool": "aggregate"}
            )
            cases.append(
                {"character": character, "seed": f"{character}-held", "pool": "validation"}
            )
        selected = select_benchmark_pool_cases({"cases": cases}, "validation", 1)
        self.assertEqual(len(selected), 5)
        self.assertTrue(all(case["seed"].endswith("-held") for case in selected))

    def test_paired_aggregate_reports_floor_deltas(self) -> None:
        rows = []
        for index, character in enumerate(HEADLESS_CHARACTERS):
            rows.append(
                {
                    "character": character,
                    "random": {
                        "floor": index + 1,
                        "steps": 10,
                        "victory": False,
                        "truncated": False,
                    },
                    "hybrid": {
                        "floor": index + 2,
                        "steps": 12,
                        "victory": character == "Regent",
                        "truncated": character == "Defect",
                    },
                    "heuristic": {
                        "floor": index + 1,
                        "steps": 11,
                        "victory": False,
                        "truncated": False,
                    },
                }
            )
        aggregate = aggregate_paired_results(rows)
        self.assertEqual(aggregate["overall"]["paired_floor_delta"], 1.0)
        self.assertEqual(aggregate["overall"]["improved"], 5)
        self.assertEqual(aggregate["overall"]["hybrid"]["victories"], 1)
        self.assertEqual(aggregate["overall"]["hybrid"]["truncations"], 1)
        self.assertEqual(
            aggregate["overall"]["hybrid_vs_heuristic"]["paired_floor_delta"],
            1.0,
        )
        self.assertEqual(aggregate["by_character"]["Silent"]["cases"], 1)

    def test_combat_distillation_aggregate_is_paired_by_case(self) -> None:
        rows = []
        for index, character in enumerate(HEADLESS_CHARACTERS):
            rows.append(
                {
                    "character": character,
                    "heuristic": {
                        "floor": index + 3,
                        "steps": 20,
                        "victory": False,
                        "truncated": False,
                    },
                    "combat_bc": {
                        "floor": index + 4,
                        "steps": 22,
                        "victory": character == "Silent",
                        "truncated": character == "Defect",
                    },
                }
            )
        aggregate = aggregate_combat_distillation(rows)
        self.assertEqual(aggregate["overall"]["paired_floor_delta"], 1.0)
        self.assertEqual(aggregate["overall"]["improved"], 5)
        self.assertEqual(aggregate["overall"]["combat_bc"]["victories"], 1)
        self.assertEqual(aggregate["overall"]["combat_bc"]["truncations"], 1)
        self.assertEqual(aggregate["by_character"]["Regent"]["cases"], 1)

    def test_combat_model_pair_aggregate_uses_candidate_minus_baseline(self) -> None:
        rows = [
            {
                "character": character,
                "baseline": {"floor": 5, "steps": 10, "truncated": False},
                "candidate": {"floor": 6, "steps": 11, "truncated": False},
            }
            for character in HEADLESS_CHARACTERS
        ]
        aggregate = aggregate_combat_model_pair(rows)
        self.assertEqual(aggregate["overall"]["paired_floor_delta"], 1.0)
        self.assertEqual(aggregate["overall"]["improved"], 5)
        self.assertEqual(aggregate["by_character"]["Ironclad"]["cases"], 1)

    def test_combat_pair_manifest_binds_models_before_fresh_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.npz"
            candidate = root / "candidate.pt"
            output = root / "manifest.json"
            baseline.write_bytes(b"baseline")
            candidate.write_bytes(b"candidate")

            manifest = create_combat_model_pair_manifest(
                output,
                baseline_model_path=baseline,
                candidate_model_path=candidate,
                baseline_kind="mlp",
                candidate_kind="set",
                seed_prefix="FRESH",
                count_per_character=1,
            )

            self.assertTrue(output.is_file())
            self.assertEqual(len(manifest["cases"]), len(HEADLESS_CHARACTERS))
            self.assertEqual(manifest["models"]["baseline"]["kind"], "mlp")
            self.assertEqual(manifest["models"]["candidate"]["kind"], "set")
            self.assertEqual(len(manifest["models"]["baseline"]["sha256"]), 64)
            self.assertFalse(manifest["selection"]["outcomes_inspected"])
            self.assertTrue(all(case["seed"].startswith("FRESH") for case in manifest["cases"]))

    def test_combat_policy_baseline_aggregate_is_character_balanced(self) -> None:
        rows = [
            {
                "character": character,
                "policy": {
                    "floor": index + 1,
                    "steps": 10 + index,
                    "victory": character == "Silent",
                    "truncated": character == "Defect",
                    "runtime_error": None,
                },
            }
            for index, character in enumerate(HEADLESS_CHARACTERS)
        ]

        aggregate = aggregate_combat_policy_baseline(rows)

        self.assertEqual(aggregate["overall"]["cases"], 5)
        self.assertEqual(aggregate["overall"]["victories"], 1)
        self.assertEqual(aggregate["overall"]["win_rate"], 0.2)
        self.assertEqual(aggregate["by_character"]["Silent"]["win_rate"], 1.0)
        self.assertEqual(aggregate["by_character"]["Defect"]["truncations"], 1)

    def test_combat_policy_baseline_manifest_excludes_reserved_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            macro = root / "macro.npz"
            combat = root / "combat.pt"
            excluded = root / "excluded.json"
            output = root / "baseline.json"
            macro.write_bytes(b"macro")
            combat.write_bytes(b"combat")
            excluded.write_text(
                json.dumps(
                    {"cases": [{"character": "CHARACTER.SILENT", "seed": "SEARCHV1S000000"}]}
                ),
                encoding="utf-8",
            )

            manifest = create_combat_policy_baseline_manifest(
                output,
                macro_model_path=macro,
                combat_model_path=combat,
                seed_prefix="BASEV1",
                count_per_character=2,
                excluded_manifests=[excluded],
            )

            self.assertEqual(len(manifest["cases"]), 10)
            self.assertEqual(manifest["models"]["combat"]["kind"], "set")
            self.assertEqual(manifest["selection"]["excluded_seed_count"], 1)
            self.assertEqual(manifest["selection"]["overlap_count"], 0)
            self.assertEqual(
                manifest["policy_constraints"]["safe_potion_ids"],
                ["POTION.ATTACK_POTION"],
            )
            self.assertTrue(all(case["seed"].startswith("BASEV1") for case in manifest["cases"]))

    def test_chip_aware_manifest_freezes_pair_gates_and_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            macro = root / "macro.npz"
            excluded = root / "excluded.json"
            output = root / "tactical.json"
            macro.write_bytes(b"macro")
            excluded.write_text(
                json.dumps({"cases": [{"character": "Silent", "seed": "OLD000001"}]}),
                encoding="utf-8",
            )

            manifest = create_chip_aware_heuristic_manifest(
                output,
                macro_model_path=macro,
                seed_prefix="TACTV1",
                count_per_character=2,
                excluded_manifests=[excluded],
            )

            self.assertEqual(len(manifest["cases"]), 10)
            self.assertEqual(manifest["arms"]["baseline"], "greedy_rules_v1")
            self.assertEqual(manifest["arms"]["candidate"], "greedy_rules_v2_chip_aware")
            self.assertEqual(manifest["selection"]["excluded_seed_count"], 1)
            self.assertFalse(manifest["selection"]["outcomes_inspected"])
            self.assertEqual(
                manifest["promotion_gate"]["overall_mean_paired_floor_delta_at_least"],
                0.75,
            )
            self.assertTrue(all(case["seed"].startswith("TACTV1") for case in manifest["cases"]))

    def test_runtime_failure_publishes_partial_trajectory_as_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "held.jsonl"
            temporary = output.with_suffix(".jsonl.tmp")
            events = [
                {
                    "schema_version": 1,
                    "event_type": "episode_start",
                    "trajectory_id": "abc",
                    "policy": "greedy_rules_v1",
                },
                {
                    "schema_version": 1,
                    "event_type": "transition",
                    "trajectory_id": "abc",
                    "step": 4,
                    "reward": 0.0,
                    "next_observation": {"context": {"act": 1, "floor": 17}},
                    "info": {"policy": {"source": "heuristic"}},
                },
            ]
            temporary.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

            result = _publish_runtime_failure(output, TimeoutError("native command timed out"))

            self.assertTrue(output.is_file())
            self.assertFalse(temporary.exists())
            self.assertTrue(result["truncated"])
            self.assertEqual(result["floor"], 17)
            self.assertEqual(result["runtime_error"]["type"], "TimeoutError")
            last = json.loads(output.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(last["event_type"], "episode_end")
            self.assertEqual(last["steps"], 4)

    def test_discard_removes_unreadable_incomplete_header(self) -> None:
        # A crash before episode_start finished writing must not block resume.
        for corrupt in ("", '{"event_type": "episode_st'):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "case.jsonl"
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_text(corrupt, encoding="utf-8")

                removed = _discard_matching_incomplete(
                    path, character="Silent", seed="ABC", policy="legal_random"
                )

                self.assertTrue(removed)
                self.assertFalse(temporary.exists())

    def test_discard_still_refuses_mismatched_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.jsonl"
            temporary = path.with_suffix(path.suffix + ".tmp")
            header = {
                "event_type": "episode_start",
                "character": "Ironclad",
                "seed": "OTHER",
                "policy": "legal_random",
            }
            temporary.write_text(json.dumps(header) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                _discard_matching_incomplete(
                    path, character="Silent", seed="ABC", policy="legal_random"
                )
            self.assertTrue(temporary.exists())


if __name__ == "__main__":
    unittest.main()
