from __future__ import annotations

import unittest

import numpy as np

from stsai.combat import action_key, legal_combat_actions, normalize_combat_observation
from stsai.combat_policy import CombatFeatureHasher, CombatPolicyAgent, prepare_combat_records
from stsai.policy import MaskedMLPPolicy


def _state(character: str = "The Regent") -> dict:
    return {
        "state_type": "elite",
        "run": {"act": 2, "floor": 22, "ascension": 10},
        "player": {
            "character": character,
            "hp": 40,
            "max_hp": 70,
            "energy": 2,
            "max_energy": 3,
            "stars": 3,
            "hand": [
                {
                    "index": 0,
                    "id": "CARD.ATTACK",
                    "type": "Attack",
                    "cost": "1",
                    "target_type": "AnyEnemy",
                    "can_play": True,
                },
                {
                    "index": 1,
                    "id": "CARD.BLOCK",
                    "type": "Skill",
                    "cost": "1",
                    "target_type": "Self",
                    "can_play": True,
                },
            ],
            "relics": [],
            "potions": [],
            "status": [],
            "orbs": [],
            "pets": [],
        },
        "battle": {
            "round": 3,
            "turn": "player",
            "is_play_phase": True,
            "enemies": [
                {
                    "entity_id": "GREMLIN_0",
                    "hp": 25,
                    "max_hp": 40,
                    "block": 0,
                    "status": [],
                    "intents": [{"type": "Attack", "damage": 10, "hits": 1}],
                }
            ],
        },
    }


class CombatPolicyTests(unittest.TestCase):
    def test_preparation_is_character_conditioned_and_candidate_specific(self) -> None:
        state = _state()
        legal = legal_combat_actions(state)
        record = {
            "patch": {"build_id": "v0.107.1"},
            "observation": normalize_combat_observation(state),
            "legal_actions": legal,
            "chosen_action": legal[0],
        }
        hasher = CombatFeatureHasher(128)
        prepared = prepare_combat_records([record], hasher)
        self.assertEqual(prepared[0].character, "CHARACTER.REGENT")
        self.assertFalse(np.array_equal(prepared[0].features[0], prepared[0].features[1]))

    def test_live_recommendation_can_only_select_a_legal_action(self) -> None:
        state = _state("The Defect")
        legal = legal_combat_actions(state)
        agent = CombatPolicyAgent(MaskedMLPPolicy(128, 16, seed=3), build_id="v0.107.1")
        result = agent.recommend(state)
        legal_keys = {action_key(action) for action in legal}
        self.assertEqual(result["character"], "CHARACTER.DEFECT")
        self.assertIn(action_key(result["action"]), legal_keys)
        self.assertAlmostEqual(
            sum(item["probability"] for item in result["ranked_actions"]), 1.0, places=6
        )


if __name__ == "__main__":
    unittest.main()
