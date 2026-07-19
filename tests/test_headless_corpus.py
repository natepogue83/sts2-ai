from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from stsai.combat import headless_combat_split
from stsai.headless_env import HEADLESS_CHARACTERS
from stsai.headless_corpus import create_combat_bootstrap_manifest


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


if __name__ == "__main__":
    unittest.main()
