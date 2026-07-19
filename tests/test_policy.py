from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np  # noqa: F401
except ImportError:  # pragma: no cover - the importer intentionally has no NumPy dependency
    np = None


@unittest.skipIf(np is None, "NumPy training extra is not installed")
class PolicyTests(unittest.TestCase):
    def test_outcomes_are_not_policy_features(self) -> None:
        from stsai.policy import FeatureHasher

        hasher = FeatureHasher(64)
        record = self._record("train", "CARD.GOOD", "CHARACTER.A")
        other = json.loads(json.dumps(record))
        other["label_metadata"] = {"run_won": False, "floors_reached": 2}
        np.testing.assert_array_equal(
            hasher.encode(record, "CARD.GOOD"), hasher.encode(other, "CARD.GOOD")
        )

    def test_policy_learns_a_masked_character_preference_and_round_trips(self) -> None:
        from stsai.policy import (
            FeatureHasher,
            MaskedMLPPolicy,
            evaluate_policy,
            prepare_records,
        )

        train_records = []
        test_records = []
        for index in range(100):
            character = "CHARACTER.A" if index % 2 == 0 else "CHARACTER.B"
            chosen = "CARD.GOOD" if character.endswith("A") else "SKIP"
            target = train_records if index < 80 else test_records
            target.append(self._record("train", chosen, character))
        hasher = FeatureHasher(96)
        train = prepare_records(train_records, hasher)
        test = prepare_records(test_records, hasher)
        model = MaskedMLPPolicy(96, 20, seed=3)
        model.fit(train, epochs=25, batch_size=16, learning_rate=0.01)
        self.assertGreaterEqual(model.evaluate(test).top1_accuracy, 0.95)

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.npz"
            data_path = Path(directory) / "data.jsonl"
            model.save(model_path)
            with data_path.open("w", encoding="utf-8") as handle:
                for record in test_records:
                    record["split"] = "test"
                    handle.write(json.dumps(record) + "\n")
            result = evaluate_policy(data_path, model_path)
        self.assertGreaterEqual(result["overall"]["top1_accuracy"], 0.95)
        self.assertEqual(
            set(result["by_character"]), {"CHARACTER.A", "CHARACTER.B"}
        )

    @staticmethod
    def _record(split: str, chosen: str, character: str) -> dict:
        return {
            "split": split,
            "patch": {"build_id": "test"},
            "observation": {
                "character": character,
                "ascension": 1,
                "act": 1,
                "floor": 2,
                "current_hp": 70,
                "max_hp": 75,
                "current_gold": 100,
                "deck": ["CARD.START"],
                "relics": ["RELIC.START"],
                "monsters": [],
            },
            "legal_options": ["CARD.GOOD", "CARD.BAD", "SKIP"],
            "chosen": chosen,
            "label_metadata": {"run_won": True, "floors_reached": 50},
        }
