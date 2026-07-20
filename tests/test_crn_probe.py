from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from stsai.crn_probe import (
    _alternative_index,
    apply_horizon_gate,
    create_crn_probe_manifest,
    extract_rng_streams,
    summarize_horizon,
)
from stsai.headless_env import HEADLESS_CHARACTERS, HeadlessAction


class CrnProbeTests(unittest.TestCase):
    def test_alternative_skips_duplicate_card_copies(self) -> None:
        actions = (
            HeadlessAction.create("play_card", "CARD.STRIKE|TARGET.0", card_index=0),
            HeadlessAction.create("play_card", "CARD.STRIKE|TARGET.0", card_index=2),
            HeadlessAction.create("play_card", "CARD.DEFEND", card_index=1),
        )
        policy_info = {
            "ranked_actions": [
                {"action_index": 0},
                {"action_index": 1},
                {"action_index": 2},
            ]
        }

        self.assertEqual(_alternative_index(actions, 0, policy_info), 2)

    def test_manifest_freezes_policy_and_mechanical_horizon_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            macro = root / "macro.npz"
            combat = root / "combat.pt"
            output = root / "manifest.json"
            macro.write_bytes(b"macro")
            combat.write_bytes(b"combat")

            manifest = create_crn_probe_manifest(
                output,
                macro_model_path=macro,
                combat_model_path=combat,
                count_per_character=2,
                roots_per_case=3,
                horizons=(8, 1, 4, 4),
            )

        self.assertEqual(manifest["measurement"]["horizons_in_policy_decisions"], [1, 4, 8])
        self.assertEqual(len(manifest["cases"]), 10)
        self.assertEqual(len({case["character"] for case in manifest["cases"]}), 5)
        self.assertEqual(manifest["selection"]["expected_root_pairs"], 30)
        self.assertTrue(manifest["base_policy"]["frozen_for_entire_tranche"])
        self.assertEqual(
            manifest["base_policy"]["macro_model_sha256"],
            hashlib.sha256(b"macro").hexdigest(),
        )
        self.assertEqual(
            manifest["horizon_selection_gate"]["method"],
            "longest_contiguous_passing_horizon",
        )
        self.assertFalse(manifest["authorization"]["training_labels"])

    def test_extract_rng_streams_preserves_split_scopes(self) -> None:
        checkpoint = {
            "semantic": {
                "run_seed_sha256": "seed-hash",
                "run_rng_counters": {"Shuffle": 11, "MonsterAi": 2},
                "player_rngs": [
                    {
                        "player_index": 0,
                        "seed": 123,
                        "counters": {"Rewards": 7, "Shops": 1},
                    }
                ],
                "combat": {
                    "creatures": [
                        {
                            "combat_id": 1,
                            "monster_id": "MONSTER.SLIME",
                            "monster_rng_seed": 456,
                            "monster_rng_counter": 3,
                        }
                    ]
                },
            }
        }

        streams = extract_rng_streams(checkpoint)

        self.assertEqual(streams["run:Shuffle"]["counter"], 11)
        self.assertEqual(streams["run:Shuffle"]["seed"], "seed-hash")
        self.assertEqual(streams["player:0:Rewards"]["counter"], 7)
        self.assertEqual(streams["monster:1:MONSTER.SLIME"]["counter"], 3)
        self.assertEqual({row["scope"] for row in streams.values()}, {"run", "player", "monster"})

    def test_summary_estimates_pairing_gain_and_keeps_character_support(self) -> None:
        cases = []
        counter = 0
        for character in HEADLESS_CHARACTERS:
            roots = []
            for _ in range(2):
                base_return = float(counter)
                counter += 1
                streams = {"run:Shuffle": {"scope": "run", "seed": "s", "counter": 4}}
                capture = {
                    "checkpoint_sha256": f"same-{counter}",
                    "rng_streams": streams,
                    "state": {"floor_plus_player_hp_fraction": base_return},
                }
                alternative = {
                    "checkpoint_sha256": f"alt-{counter}",
                    "rng_streams": streams,
                    "state": {"floor_plus_player_hp_fraction": base_return + 0.25},
                }
                roots.append(
                    {
                        "root_rng_streams": streams,
                        "base": {"horizons": {"1": capture}},
                        "base_repeat": {"horizons": {"1": dict(capture)}},
                        "alternative": {"horizons": {"1": alternative}},
                    }
                )
            cases.append({"character": character, "roots": roots})

        summary = summarize_horizon(cases, 1, expected_per_character=2)

        self.assertEqual(summary["valid_pairs"], 10)
        self.assertEqual(summary["exact_base_repeat_fraction"], 1.0)
        self.assertEqual(summary["mean_root_stream_alignment_fraction"], 1.0)
        self.assertEqual(summary["return_variance"]["paired_difference"], 0.0)
        self.assertEqual(
            {row["valid_pairs"] for row in summary["valid_by_character"].values()},
            {2},
        )

    def test_gate_selects_only_longest_contiguous_passing_horizon(self) -> None:
        def summary(horizon: int, ratio: float, repeat: float = 1.0) -> dict:
            return {
                "horizon": horizon,
                "valid_pairs": 10,
                "valid_fraction": 1.0,
                "valid_by_character": {
                    character: {"valid_pairs": 2} for character in HEADLESS_CHARACTERS
                },
                "exact_base_repeat_fraction": repeat,
                "mean_root_stream_alignment_fraction": 0.9,
                "return_variance": {"paired_to_unpaired_ratio": ratio},
            }

        gate = {
            "minimum_valid_pair_fraction": 0.9,
            "minimum_valid_pairs": 10,
            "minimum_valid_pairs_per_character": 2,
            "minimum_exact_base_repeat_fraction": 1.0,
            "minimum_mean_root_stream_alignment_fraction": 0.75,
            "maximum_paired_to_unpaired_variance_ratio": 0.5,
            "terminal_decision_if_none_pass": "stop",
        }

        result = apply_horizon_gate(
            [summary(1, 0.2), summary(2, 0.7), summary(4, 0.1)], gate
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["selected_horizon"], 1)
        self.assertFalse(result["evaluated"][2]["contiguous_prefix_passed"])


if __name__ == "__main__":
    unittest.main()
