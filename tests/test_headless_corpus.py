from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from stsai.combat import headless_combat_split
from stsai.headless_env import HEADLESS_CHARACTERS
from stsai.headless_corpus import _discard_incomplete, create_combat_bootstrap_manifest


class HeadlessCorpusTests(unittest.TestCase):
    def test_manifest_has_exact_character_split_counts_without_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            payload = create_combat_bootstrap_manifest(
                output,
                train_per_character=2,
                validation_per_character=1,
                test_per_character=1,
            )
            published = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(payload["cases"]), len(HEADLESS_CHARACTERS) * 4)
        self.assertEqual(payload["cases"], published["cases"])
        self.assertFalse(payload["selection"]["outcomes_inspected"])
        self.assertEqual(len({case["seed"] for case in payload["cases"]}), 20)
        for character in HEADLESS_CHARACTERS:
            canonical = f"CHARACTER.{character.upper()}"
            cases = [
                case for case in payload["cases"] if case["character"] == canonical
            ]
            self.assertEqual(
                {split: sum(case["split"] == split for case in cases) for split in (
                    "train",
                    "validation",
                    "test",
                )},
                {"train": 2, "validation": 1, "test": 1},
            )
            for case in cases:
                self.assertEqual(
                    case["split"],
                    headless_combat_split(
                        build_id="v0.107.1",
                        character=character,
                        seed=case["seed"],
                    ),
                )


class DiscardIncompleteTests(unittest.TestCase):
    def test_removes_matching_and_unreadable_incomplete_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000.train.silent.CBTS000000.jsonl"
            temporary = path.with_suffix(path.suffix + ".tmp")

            self.assertFalse(
                _discard_incomplete(path, character="Silent", seed="CBTS000000")
            )

            header = {
                "event_type": "episode_start",
                "character": "Silent",
                "seed": "CBTS000000",
            }
            temporary.write_text(json.dumps(header) + "\n", encoding="utf-8")
            self.assertTrue(
                _discard_incomplete(path, character="Silent", seed="CBTS000000")
            )
            self.assertFalse(temporary.exists())

            # A crash before the header finished writing must also unblock.
            temporary.write_text('{"event_type": "epis', encoding="utf-8")
            self.assertTrue(
                _discard_incomplete(path, character="Silent", seed="CBTS000000")
            )
            self.assertFalse(temporary.exists())

    def test_refuses_to_remove_a_different_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000.train.silent.CBTS000000.jsonl"
            temporary = path.with_suffix(path.suffix + ".tmp")
            header = {
                "event_type": "episode_start",
                "character": "Ironclad",
                "seed": "CBTI000042",
            }
            temporary.write_text(json.dumps(header) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                _discard_incomplete(path, character="Silent", seed="CBTS000000")
            self.assertTrue(temporary.exists())


if __name__ == "__main__":
    unittest.main()
