from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from stsai.value import (
    VALUE_CHARACTERS,
    VALUE_FEATURE_NAMES,
    VALUE_INTERACTION_FEATURE_NAMES,
    VALUE_NONLINEAR_FEATURE_NAMES,
    CalibratedLinearValueEvaluator,
    RankOnlyPublicValueEvaluator,
    audit_value_model,
    build_value_corpus,
    calibration_metrics,
    public_value_features,
)
from stsai.value_corpus import create_value_calibration_manifest


def observation(character: str = "CHARACTER.IRONCLAD", *, seed: str = "HIDDEN"):
    return {
        "character": character,
        "ascension": 10,
        "seed": seed,
        "decision_type": "combat_play",
        "context": {"act": 1, "floor": 4},
        "player": {
            "hp": 60,
            "max_hp": 80,
            "block": 4,
            "gold": 120,
            "deck_size": 12,
            "relics": [{"id": "RELIC.BURNING_BLOOD"}],
            "potions": [],
        },
        "decision_state": {
            "round": 2,
            "energy": 2,
            "hand": [
                {"id": "CARD.STRIKE_IRONCLAD", "can_play": True},
                {"id": "CARD.ASCENDERS_BANE", "can_play": False},
            ],
            "enemies": [
                {
                    "id": "MONSTER.TEST",
                    "hp": 20,
                    "max_hp": 40,
                    "block": 2,
                    "intents": [{"damage": 7}],
                }
            ],
            "player_powers": [],
            "orbs": [],
        },
    }


class FakeEnv:
    def __init__(self, value):
        self.observation = value


class ValueTests(unittest.TestCase):
    def test_public_features_are_character_aware_but_ignore_seed(self) -> None:
        first = public_value_features(observation(seed="ONE"))
        second = public_value_features(observation(seed="TWO"))
        silent = public_value_features(observation("CHARACTER.SILENT"))

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, silent))
        self.assertEqual(first[: len(VALUE_CHARACTERS)].sum(), 1.0)
        self.assertNotIn("seed", VALUE_FEATURE_NAMES)

        interactions = public_value_features(
            observation(), feature_schema="character_interactions_v2"
        )
        self.assertEqual(interactions.shape, (len(VALUE_INTERACTION_FEATURE_NAMES),))
        self.assertGreater(np.count_nonzero(interactions), np.count_nonzero(first))

        nonlinear = public_value_features(observation(), feature_schema="character_nonlinear_v3")
        self.assertEqual(nonlinear.shape, (len(VALUE_NONLINEAR_FEATURE_NAMES),))
        self.assertGreater(np.count_nonzero(nonlinear), np.count_nonzero(interactions))

    def test_evaluator_is_explainable_and_round_trips(self) -> None:
        weights = np.linspace(-0.3, 0.4, len(VALUE_FEATURE_NAMES))
        model = CalibratedLinearValueEvaluator(
            weights,
            -0.1,
            0.8,
            0.2,
            horizon_floors=3,
            calibration_status="passed_held_out_gate",
            search_eligible=True,
        )
        estimate = model.evaluate(FakeEnv(observation()))

        self.assertGreaterEqual(estimate.value, 0.0)
        self.assertLessEqual(estimate.value, 1.0)
        self.assertEqual(estimate.metadata["character"], "CHARACTER.IRONCLAD")
        self.assertFalse(estimate.metadata["uses_hidden_state"])
        self.assertIn("Ironclad", estimate.metadata["explanation"]["summary"])
        self.assertEqual(len(estimate.metadata["explanation"]["top_factors"]), 5)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.npz"
            model.save(path)
            restored = CalibratedLinearValueEvaluator.load(path)
            self.assertAlmostEqual(restored.evaluate(FakeEnv(observation())).value, estimate.value)

    def test_nonlinear_ensemble_round_trips_per_character_calibration(self) -> None:
        weights = np.stack(
            [
                np.linspace(-0.2, 0.3, len(VALUE_NONLINEAR_FEATURE_NAMES)),
                np.linspace(-0.1, 0.2, len(VALUE_NONLINEAR_FEATURE_NAMES)),
                np.linspace(-0.3, 0.4, len(VALUE_NONLINEAR_FEATURE_NAMES)),
            ]
        )
        model = CalibratedLinearValueEvaluator(
            weights,
            np.asarray([-0.1, 0.0, 0.1]),
            np.asarray([0.8, 0.9, 1.0, 1.1, 1.2]),
            np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2]),
            horizon_floors=3,
            feature_schema="character_nonlinear_v3",
            calibration_kind="per_character_platt",
        )
        estimate = model.evaluate(FakeEnv(observation("CHARACTER.DEFECT")))

        self.assertGreaterEqual(estimate.value, 0.0)
        self.assertLessEqual(estimate.value, 1.0)
        self.assertEqual(estimate.metadata["ensemble_members"], 3)
        self.assertEqual(estimate.metadata["calibration_kind"], "per_character_platt")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value-v4.npz"
            model.save(path)
            restored = CalibratedLinearValueEvaluator.load(path)
            restored_estimate = restored.evaluate(FakeEnv(observation("CHARACTER.DEFECT")))

        self.assertAlmostEqual(restored_estimate.value, estimate.value)
        np.testing.assert_array_equal(restored.weights, weights)

    def test_rank_only_adapter_binds_checkpoint_and_requires_positive_slopes(self) -> None:
        weights = np.zeros((3, len(VALUE_NONLINEAR_FEATURE_NAMES)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value-v4.npz"
            model = CalibratedLinearValueEvaluator(
                weights,
                np.zeros(3),
                np.asarray([0.8, 0.9, 1.0, 1.1, 1.2]),
                np.zeros(5),
                horizon_floors=3,
                feature_schema="character_nonlinear_v3",
                calibration_kind="per_character_platt",
                calibration_status="rejected_held_out_gate",
                search_eligible=False,
            )
            model.save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            ranker = RankOnlyPublicValueEvaluator(path, expected_sha256=digest)
            estimate = ranker.evaluate(FakeEnv(observation("CHARACTER.REGENT")))

            self.assertTrue(ranker.search_eligible)
            self.assertFalse(estimate.metadata["uses_hidden_state"])
            self.assertFalse(estimate.metadata["training_labels_emitted"])
            self.assertEqual(
                estimate.metadata["original_calibration_status"],
                "rejected_held_out_gate",
            )
            with self.assertRaisesRegex(ValueError, "frozen hash"):
                RankOnlyPublicValueEvaluator(path, expected_sha256="0" * 64)

            rejected = CalibratedLinearValueEvaluator(
                weights,
                np.zeros(3),
                np.asarray([0.8, 0.9, 0.0, 1.1, 1.2]),
                np.zeros(5),
                horizon_floors=3,
                feature_schema="character_nonlinear_v3",
                calibration_kind="per_character_platt",
            )
            rejected.save(path)
            rejected_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "preserve within-character ranking"):
                RankOnlyPublicValueEvaluator(path, expected_sha256=rejected_digest)

    def test_corpus_labels_future_survival_and_groups_by_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(12):
                final_floor = 8 if index % 2 == 0 else 5
                start = {
                    "event_type": "episode_start",
                    "character": "CHARACTER.IRONCLAD",
                    "seed": f"GROUP-{index}",
                }
                transition = {
                    "event_type": "transition",
                    "observation": observation(seed=start["seed"]),
                }
                duplicate_round = json.loads(json.dumps(transition))
                end = {
                    "event_type": "episode_end",
                    "terminated": True,
                    "truncated": False,
                    "victory": False,
                    "floor": final_floor,
                }
                path = root / f"case-{index}.jsonl"
                path.write_text(
                    "\n".join(
                        json.dumps(item) for item in (start, transition, duplicate_round, end)
                    )
                    + "\n",
                    encoding="utf-8",
                )

            corpus = build_value_corpus([root], horizon_floors=3)

        self.assertEqual(corpus.trajectories, 12)
        self.assertEqual(len(corpus.examples), 12)
        self.assertEqual(sum(example.label for example in corpus.examples), 6.0)
        self.assertTrue(all(len(example.group_sha256) == 64 for example in corpus.examples))

    def test_calibration_metrics_report_brier_ece_and_bins(self) -> None:
        metrics = calibration_metrics(
            np.asarray([0.0, 0.0, 1.0, 1.0]),
            np.asarray([0.1, 0.2, 0.8, 0.9]),
        )
        self.assertEqual(metrics["examples"], 4)
        self.assertLess(metrics["brier"], 0.05)
        self.assertGreater(len(metrics["bins"]), 0)

    def test_value_manifest_freezes_disjoint_balanced_pools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            excluded = root / "reserved.json"
            excluded.write_text(
                json.dumps({"cases": [{"seed": "VCAL3DI000000"}]}) + "\n",
                encoding="utf-8",
            )
            payload = create_value_calibration_manifest(
                root / "manifest.json",
                development_per_character=2,
                audit_per_character=3,
                excluded_manifests=[excluded],
            )

        development = [case for case in payload["cases"] if case["pool"] == "development"]
        audit = [case for case in payload["cases"] if case["pool"] == "audit"]
        self.assertEqual(len(development), len(VALUE_CHARACTERS) * 2)
        self.assertEqual(len(audit), len(VALUE_CHARACTERS) * 3)
        self.assertFalse({case["seed"] for case in development} & {case["seed"] for case in audit})
        self.assertNotIn("VCAL3DI000000", {case["seed"] for case in payload["cases"]})
        self.assertFalse(payload["selection"]["outcomes_inspected"])

    def test_value_manifest_can_freeze_a_fresh_audit_only_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = create_value_calibration_manifest(
                Path(directory) / "manifest.json",
                development_per_character=0,
                audit_per_character=2,
                seed_prefix="VCAL4",
            )

        self.assertEqual(len(payload["cases"]), len(VALUE_CHARACTERS) * 2)
        self.assertTrue(all(case["pool"] == "audit" for case in payload["cases"]))
        self.assertTrue(all(case["seed"].startswith("VCAL4A") for case in payload["cases"]))
        self.assertFalse(payload["audit_protocol"]["development_pool_may_be_used_for_training"])

    def test_independent_audit_promotes_only_after_exact_balanced_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest = create_value_calibration_manifest(
                manifest_path,
                development_per_character=1,
                audit_per_character=4,
            )
            audit_dir = root / "audit"
            audit_dir.mkdir()
            audit_cases = [case for case in manifest["cases"] if case["pool"] == "audit"]
            character_counts = {character: 0 for character in VALUE_CHARACTERS}
            for index, case in enumerate(audit_cases):
                within_character = character_counts[case["character"]]
                character_counts[case["character"]] += 1
                transition_observation = observation(case["character"])
                path = audit_dir / f"{index:03d}.jsonl"
                events = (
                    {
                        "event_type": "episode_start",
                        "character": case["character"],
                        "seed": case["seed"],
                    },
                    {
                        "event_type": "transition",
                        "observation": transition_observation,
                    },
                    {
                        "event_type": "episode_end",
                        "terminated": True,
                        "truncated": False,
                        "victory": False,
                        "floor": 7 if within_character % 2 == 0 else 4,
                    },
                )
                path.write_text(
                    "\n".join(json.dumps(event) for event in events) + "\n",
                    encoding="utf-8",
                )
            model_path = root / "value.npz"
            model = CalibratedLinearValueEvaluator(
                np.zeros(len(VALUE_FEATURE_NAMES)),
                0.0,
                1.0,
                0.0,
                horizon_floors=3,
                calibration_status="awaiting_independent_audit",
                search_eligible=False,
            )
            model.save(model_path)
            report = audit_value_model(
                [audit_dir],
                model_path,
                manifest_path,
                metrics_output=root / "audit.json",
            )
            restored = CalibratedLinearValueEvaluator.load(model_path)

        self.assertTrue(report["calibration_promotion_gate"]["passed"])
        self.assertTrue(report["parameters_frozen_during_audit"])
        self.assertEqual(report["audit_groups"], len(VALUE_CHARACTERS) * 4)
        self.assertTrue(restored.search_eligible)
        self.assertEqual(restored.calibration_status, "passed_independent_audit_gate")
        np.testing.assert_array_equal(restored.weights, model.weights)


if __name__ == "__main__":
    unittest.main()
