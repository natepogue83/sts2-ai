from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from stsai.headless_env import Sts2Env
from stsai.set_policy import (
    DeepSetCombatPolicy,
    HeadlessSetCombatPolicy,
    prepare_set_record,
)


class FakeClient:
    version = "test"

    def __init__(self, state: dict):
        self.state = state

    def start_run(self, **_: object) -> dict:
        return self.state


class FakeMacroModel:
    dimension = 64


def combat_record() -> dict:
    return {
        "decision_type": "combat_action",
        "patch": {"build_id": "v0.107.1"},
        "observation": {
            "state_type": "monster",
            "run": {"act": 1, "floor": 4, "ascension": 10},
            "player": {
                "character": "CHARACTER.DEFECT",
                "hp": 55,
                "max_hp": 75,
                "block": 3,
                "energy": 3,
                "max_energy": 3,
                "hand": [
                    {
                        "index": 0,
                        "id": "CARD.STRIKE_DEFECT",
                        "type": "Attack",
                        "cost": 1,
                        "target_type": "AnyEnemy",
                        "can_play": True,
                    },
                    {
                        "index": 1,
                        "id": "CARD.DEFEND_DEFECT",
                        "type": "Skill",
                        "cost": 1,
                        "target_type": "None",
                        "can_play": True,
                    },
                ],
                "relics": [{"id": "RELIC.CRACKED_CORE"}],
                "potions": [{"slot": 0, "id": "POTION.SWIFT_POTION"}],
                "status": [],
                "orbs": [{"id": "ORB.LIGHTNING", "passive_val": 3}],
                "pets": [],
            },
            "battle": {
                "round": 2,
                "enemies": [
                    {
                        "entity_id": "enemy:0",
                        "id": "MONSTER.A",
                        "hp": 6,
                        "max_hp": 20,
                        "block": 0,
                        "intents": [{"type": "Attack", "damage": 8, "hits": 1}],
                        "status": [],
                    },
                    {
                        "entity_id": "enemy:1",
                        "id": "MONSTER.B",
                        "hp": 18,
                        "max_hp": 18,
                        "block": 0,
                        "intents": [{"type": "Buff"}],
                        "status": [],
                    },
                ],
            },
        },
        "legal_actions": [
            {"action": "play_card", "card_index": 0, "target": "enemy:0"},
            {"action": "play_card", "card_index": 0, "target": "enemy:1"},
            {"action": "play_card", "card_index": 1},
            {"action": "use_potion", "slot": 0},
            {"action": "end_turn"},
        ],
        "chosen_action": {
            "action": "play_card",
            "card_index": 0,
            "target": "enemy:0",
        },
    }


class SetPolicyTests(unittest.TestCase):
    def test_preparation_is_character_aware_and_keeps_dynamic_label(self) -> None:
        prepared = prepare_set_record(combat_record())

        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertEqual(prepared.character, "CHARACTER.DEFECT")
        self.assertEqual(prepared.action_type, "play_card")
        self.assertEqual(prepared.chosen_index, 0)
        self.assertEqual(len(prepared.actions), 5)
        self.assertEqual(len(prepared.groups), 4)

    def test_scores_are_invariant_to_hand_and_enemy_order(self) -> None:
        original = combat_record()
        permuted = deepcopy(original)
        permuted["observation"]["player"]["hand"].reverse()
        permuted["observation"]["battle"]["enemies"].reverse()
        first = prepare_set_record(original)
        second = prepare_set_record(permuted)
        assert first is not None and second is not None
        torch.manual_seed(7)
        model = DeepSetCombatPolicy().eval()

        first_scores, _ = model([first])
        second_scores, _ = model([second])

        self.assertTrue(torch.allclose(first_scores, second_scores, atol=1e-6))

    def test_checkpoint_round_trip_is_atomic_and_compact(self) -> None:
        prepared = prepare_set_record(combat_record())
        assert prepared is not None
        torch.manual_seed(7)
        model = DeepSetCombatPolicy().eval()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "set-policy.pt"
            model.save(output)
            restored = DeepSetCombatPolicy.load(output).eval()
            before, _ = model([prepared])
            after, _ = restored([prepared])

            self.assertTrue(output.is_file())
            self.assertFalse(output.with_suffix(".pt.tmp").exists())

        self.assertLess(model.parameter_count, 500_000)
        self.assertTrue(torch.equal(before, after))

    def test_live_policy_scores_only_legal_character_aware_actions(self) -> None:
        state = {
            "type": "decision",
            "decision": "combat_play",
            "context": {"act": 1, "floor": 2, "room_type": "Monster"},
            "round": 1,
            "energy": 3,
            "max_energy": 3,
            "hand": [
                {
                    "index": 0,
                    "id": "CARD.STRIKE_DEFECT",
                    "type": "Attack",
                    "cost": 1,
                    "can_play": True,
                    "target_type": "AnyEnemy",
                }
            ],
            "enemies": [
                {"index": 0, "id": "MONSTER.A", "hp": 20, "max_hp": 20},
                {"index": 1, "id": "MONSTER.B", "hp": 5, "max_hp": 10},
            ],
            "player": {
                "hp": 50,
                "max_hp": 75,
                "block": 0,
                "gold": 99,
                "deck": [],
                "potions": [],
            },
        }
        env = Sts2Env(client=FakeClient(state))
        env.reset(character="Defect", ascension=10, seed="ABC")
        torch.manual_seed(7)
        policy = HeadlessSetCombatPolicy(FakeMacroModel(), DeepSetCombatPolicy())

        index, info = policy.select(env)

        self.assertGreaterEqual(index, 0)
        self.assertLess(index, len(env.legal_actions))
        self.assertEqual(info["source"], "combat_deepsets")
        self.assertEqual(info["character"], "CHARACTER.DEFECT")
        self.assertEqual(len(info["ranked_actions"]), len(env.legal_actions))


if __name__ == "__main__":
    unittest.main()
