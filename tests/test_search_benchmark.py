from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from stsai.search_benchmark import _search_coverage, create_rank_search_manifest
from stsai.value import (
    VALUE_NONLINEAR_FEATURE_NAMES,
    CalibratedLinearValueEvaluator,
)


class SearchBenchmarkTests(unittest.TestCase):
    def test_manifest_binds_ranker_and_fresh_character_balanced_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            macro = root / "macro.npz"
            combat = root / "combat.pt"
            value = root / "value.npz"
            excluded = root / "excluded.json"
            output = root / "manifest.json"
            macro.write_bytes(b"macro")
            combat.write_bytes(b"combat")
            CalibratedLinearValueEvaluator(
                np.zeros((3, len(VALUE_NONLINEAR_FEATURE_NAMES))),
                np.zeros(3),
                np.ones(5),
                np.zeros(5),
                horizon_floors=3,
                feature_schema="character_nonlinear_v3",
                calibration_kind="per_character_platt",
                calibration_status="rejected_held_out_gate",
            ).save(value)
            excluded.write_text(
                json.dumps({"cases": [{"character": "Ironclad", "seed": "SEARCHR1I000000"}]}),
                encoding="utf-8",
            )

            manifest = create_rank_search_manifest(
                output,
                macro_model_path=macro,
                combat_model_path=combat,
                value_model_path=value,
                seed_prefix="SEARCHR1",
                count_per_character=1,
                excluded_manifests=[excluded],
            )

            self.assertEqual(len(manifest["cases"]), 5)
            self.assertEqual(len({case["character"] for case in manifest["cases"]}), 5)
            self.assertNotIn("SEARCHR1I000000", {case["seed"] for case in manifest["cases"]})
            self.assertFalse(manifest["search_policy"]["training_labels_emitted"])
            self.assertFalse(manifest["search_policy"]["absolute_probability_claimed"])
            self.assertEqual(manifest["power"]["paired_cases"], 5)

    def test_search_coverage_counts_used_changed_and_fallback_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "search.jsonl"
            events = [
                {
                    "event_type": "transition",
                    "observation": {"decision_type": "combat_play"},
                    "info": {
                        "policy": {
                            "search": {
                                "used": True,
                                "nodes": 3,
                                "selected_action_index": 1,
                                "base_action_index": 0,
                            }
                        }
                    },
                },
                {
                    "event_type": "transition",
                    "observation": {"decision_type": "card_reward"},
                    "info": {"policy": {"search": {"used": False, "reason": "noncombat_fallback"}}},
                },
            ]
            path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

            coverage = _search_coverage(path)

            self.assertEqual(coverage["combat_decisions"], 1)
            self.assertEqual(coverage["search_used_decisions"], 1)
            self.assertEqual(coverage["search_changed_action_decisions"], 1)
            self.assertEqual(coverage["branch_nodes"], 3)
            self.assertEqual(coverage["fallback_reasons"], {"noncombat_fallback": 1})


if __name__ == "__main__":
    unittest.main()
