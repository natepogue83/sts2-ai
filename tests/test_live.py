from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from stsai.live import MacroPolicyAgent, canonical_character, load_current_deck


class _HighestLastModel:
    dimension = 64

    def scores(self, features: np.ndarray) -> np.ndarray:
        return np.arange(len(features), dtype=np.float32)


class _HighestFirstModel:
    dimension = 64

    def scores(self, features: np.ndarray) -> np.ndarray:
        return -np.arange(len(features), dtype=np.float32)


class LivePolicyTests(unittest.TestCase):
    def test_character_names_match_training_ids(self) -> None:
        self.assertEqual(canonical_character("The Ironclad"), "CHARACTER.IRONCLAD")
        self.assertEqual(canonical_character("NECROBINDER"), "CHARACTER.NECROBINDER")

    def test_loads_only_matching_character_deck(self) -> None:
        save = {
            "players": [
                {"character_id": "SILENT", "deck": [{"id": "CARD.NEUTRALIZE"}]},
                {"character_id": "IRONCLAD", "deck": [{"id": "CARD.BASH"}]},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current_run.save"
            path.write_text(json.dumps(save), encoding="utf-8")
            deck = load_current_deck("The Ironclad", directory)
        self.assertEqual(deck, ["CARD.BASH"])

    def test_card_reward_recommendation_is_character_conditioned_and_masked(self) -> None:
        state = self._base_state()
        state.update(
            {
                "state_type": "card_reward",
                "card_reward": {
                    "cards": [
                        {"index": 0, "id": "BASH"},
                        {"index": 1, "id": "ANGER"},
                    ],
                    "can_skip": True,
                },
            }
        )
        agent = MacroPolicyAgent(_HighestLastModel(), build_id="v-test", save_root="missing")
        result = agent.recommend(state)
        self.assertEqual(result["character"], "CHARACTER.IRONCLAD")
        self.assertEqual(result["chosen"], "SKIP")
        self.assertEqual(result["action"], {"action": "skip_card_reward"})

    def test_ancient_never_invents_a_skip_action(self) -> None:
        state = self._base_state()
        state.update(
            {
                "state_type": "event",
                "event": {
                    "event_id": "NEOW",
                    "is_ancient": True,
                    "in_dialogue": False,
                    "options": [
                        {
                            "index": 4,
                            "title": "New Leaf",
                            "relic_name": "New Leaf",
                            "is_locked": False,
                            "is_proceed": False,
                        },
                        {
                            "index": 7,
                            "title": "Neow's Bones",
                            "relic_name": "Neow's Bones",
                            "is_locked": False,
                            "is_proceed": False,
                        },
                    ],
                },
            }
        )
        agent = MacroPolicyAgent(_HighestFirstModel(), build_id="v-test", save_root="missing")
        result = agent.recommend(state)
        self.assertEqual(result["chosen"], "ANCIENT.NEW_LEAF")
        self.assertEqual(
            result["action"], {"action": "choose_event_option", "index": 4}
        )
        self.assertNotIn("SKIP", [item["option"] for item in result["ranked_options"]])

    @staticmethod
    def _base_state() -> dict:
        return {
            "run": {"act": 1, "floor": 2, "ascension": 10},
            "player": {
                "character": "The Ironclad",
                "hp": 70,
                "max_hp": 80,
                "gold": 100,
                "relics": [{"id": "BURNING_BLOOD"}],
            },
        }


if __name__ == "__main__":
    unittest.main()
