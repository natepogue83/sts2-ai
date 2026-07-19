from __future__ import annotations

import unittest

import numpy as np

from stsai.headless import HeadlessRuntimeError
from stsai.headless_env import Sts2Env
from stsai.headless_policy import HeadlessHybridPolicy


class FakeClient:
    version = "test"

    def __init__(self, state: dict):
        self.state = state

    def start_run(self, **_: object) -> dict:
        return self.state


class FakeModel:
    dimension = 64

    def scores(self, features: np.ndarray) -> np.ndarray:
        return np.arange(len(features), dtype=np.float32)


class CapturingHasher:
    def __init__(self):
        self.records: list[tuple[dict, str]] = []

    def encode(self, record: dict, option: str) -> np.ndarray:
        self.records.append((record, option))
        return np.zeros(64, dtype=np.float32)


class CapturingCombatHasher:
    def __init__(self):
        self.records: list[tuple[dict, dict]] = []

    def encode(self, record: dict, action: dict) -> np.ndarray:
        self.records.append((record, action))
        return np.zeros(64, dtype=np.float32)


def reset_env(state: dict, character: str = "Silent") -> Sts2Env:
    env = Sts2Env(client=FakeClient(state))
    env.reset(character=character, ascension=10, seed="ABC")
    return env


class HybridPolicyTests(unittest.TestCase):
    def test_neow_uses_macro_model_with_character_and_only_legal_options(self) -> None:
        state = {
            "type": "decision",
            "decision": "event_choice",
            "context": {"act": 1, "floor": 1, "room_type": "Event"},
            "event_name": "Neow",
            "options": [
                {
                    "index": 0,
                    "title": "Fishing Rod",
                    "text_key": "NEOW.pages.INITIAL.options.FISHING_ROD",
                    "is_locked": False,
                },
                {
                    "index": 1,
                    "title": "Locked",
                    "text_key": "NEOW.pages.INITIAL.options.LOCKED",
                    "is_locked": True,
                },
                {
                    "index": 2,
                    "title": "Hefty Tablet",
                    "text_key": "NEOW.pages.INITIAL.options.HEFTY_TABLET",
                    "is_locked": False,
                },
            ],
            "player": {
                "hp": 45,
                "max_hp": 60,
                "gold": 99,
                "deck": [{"id": "CARD.BIND"}],
                "relics": [{"name": "Bound Phylactery"}],
            },
        }
        env = reset_env(state, "Necrobinder")
        policy = HeadlessHybridPolicy(FakeModel())
        capture = CapturingHasher()
        policy.hasher = capture

        index, info = policy.select(env)

        self.assertEqual(index, 1)
        self.assertEqual(info["source"], "macro_bc")
        self.assertEqual(info["chosen_option"], "ANCIENT.HEFTY_TABLET")
        self.assertEqual(
            [option for _, option in capture.records],
            ["ANCIENT.FISHING_ROD", "ANCIENT.HEFTY_TABLET"],
        )
        record = capture.records[0][0]
        self.assertEqual(record["observation"]["character"], "CHARACTER.NECROBINDER")
        self.assertEqual(record["observation"]["deck"], ["CARD.BIND"])

    def test_card_reward_model_can_choose_skip(self) -> None:
        state = {
            "type": "decision",
            "decision": "card_reward",
            "context": {"act": 1, "floor": 2, "room_type": "Monster"},
            "cards": [
                {"index": 0, "id": "CARD.A"},
                {"index": 1, "id": "CARD.B"},
            ],
            "can_skip": True,
            "player": {"hp": 50, "max_hp": 70, "gold": 110, "deck": []},
        }
        env = reset_env(state)
        policy = HeadlessHybridPolicy(FakeModel())

        index, info = policy.select(env)

        self.assertEqual(index, 2)
        self.assertEqual(env.legal_actions[index].command, "skip_card_reward")
        self.assertEqual(info["chosen_option"], "SKIP")

    def test_heuristic_ablation_bypasses_macro_scoring(self) -> None:
        state = {
            "type": "decision",
            "decision": "card_reward",
            "context": {"act": 1, "floor": 2, "room_type": "Monster"},
            "cards": [
                {"index": 0, "id": "CARD.A"},
                {"index": 1, "id": "CARD.B"},
            ],
            "can_skip": True,
            "player": {"hp": 50, "max_hp": 70, "gold": 110, "deck": []},
        }
        env = reset_env(state)
        policy = HeadlessHybridPolicy(FakeModel(), use_macro=False)

        index, info = policy.select(env)

        self.assertEqual(index, 0)
        self.assertEqual(info["source"], "heuristic")
        self.assertEqual(env.legal_actions[index].command, "select_card_reward")

    def test_combat_heuristic_prefers_a_lethal_target_over_end_turn(self) -> None:
        state = {
            "type": "decision",
            "decision": "combat_play",
            "context": {"act": 1, "floor": 2, "room_type": "Monster"},
            "hand": [
                {
                    "index": 0,
                    "id": "CARD.STRIKE_SILENT",
                    "type": "Attack",
                    "cost": 1,
                    "can_play": True,
                    "target_type": "AnyEnemy",
                    "stats": {"damage": 6},
                    "damage_by_target": [
                        {"target_index": 0, "damage": 6},
                        {"target_index": 1, "damage": 6},
                    ],
                }
            ],
            "enemies": [
                {"index": 0, "hp": 20, "block": 0},
                {"index": 1, "hp": 5, "block": 0},
            ],
            "player": {
                "hp": 50,
                "max_hp": 70,
                "block": 0,
                "gold": 99,
                "deck": [],
                "potions": [],
            },
        }
        env = reset_env(state)
        policy = HeadlessHybridPolicy(FakeModel())

        index, info = policy.select(env)

        self.assertEqual(info["source"], "heuristic")
        self.assertEqual(env.legal_actions[index].command, "play_card")
        self.assertEqual(env.legal_actions[index].argument_dict()["target_index"], 1)

    def test_chip_aware_heuristic_blocks_before_nonlethal_damage(self) -> None:
        state = {
            "type": "decision",
            "decision": "combat_play",
            "context": {"act": 1, "floor": 4, "room_type": "Monster"},
            "hand": [
                {
                    "index": 0,
                    "id": "CARD.STRIKE_SILENT",
                    "type": "Attack",
                    "cost": 1,
                    "can_play": True,
                    "target_type": "AnyEnemy",
                    "stats": {"damage": 6},
                },
                {
                    "index": 1,
                    "id": "CARD.DEFEND_SILENT",
                    "type": "Skill",
                    "cost": 1,
                    "can_play": True,
                    "target_type": "Self",
                    "stats": {"block": 5},
                },
            ],
            "enemies": [
                {
                    "index": 0,
                    "hp": 30,
                    "block": 0,
                    "intents": [{"damage": 8, "hits": 1}],
                }
            ],
            "player": {
                "hp": 50,
                "max_hp": 70,
                "block": 0,
                "gold": 99,
                "deck": [],
                "potions": [],
            },
        }
        original_env = reset_env(state)
        original = HeadlessHybridPolicy(FakeModel(), heuristic_version="v1")
        chip_env = reset_env(state)
        chip_aware = HeadlessHybridPolicy(FakeModel(), heuristic_version="v2_chip_aware")

        original_index, _ = original.select(original_env)
        chip_index, chip_info = chip_aware.select(chip_env)

        self.assertTrue(
            original_env.legal_actions[original_index].option_id.startswith("CARD.STRIKE_SILENT")
        )
        self.assertEqual(chip_env.legal_actions[chip_index].option_id, "CARD.DEFEND_SILENT")
        self.assertEqual(chip_info["policy"], "greedy_rules_v2_chip_aware")

    def test_opt_in_combat_model_scores_only_live_character_aware_actions(self) -> None:
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
        env = reset_env(state, "Defect")
        policy = HeadlessHybridPolicy(FakeModel(), combat_model=FakeModel())
        capture = CapturingCombatHasher()
        policy.combat_hasher = capture

        index, info = policy.select(env)

        self.assertEqual(info["source"], "combat_bc")
        self.assertEqual(info["character"], "CHARACTER.DEFECT")
        self.assertEqual(env.legal_actions[index].command, "end_turn")
        self.assertEqual(
            [action for _, action in capture.records],
            [
                {"action": "play_card", "card_index": 0, "target": "enemy:0"},
                {"action": "play_card", "card_index": 0, "target": "enemy:1"},
                {"action": "end_turn"},
            ],
        )
        self.assertTrue(
            all(
                record["observation"]["player"]["character"] == "CHARACTER.DEFECT"
                for record, _ in capture.records
            )
        )

    def test_low_hp_rest_heuristic_chooses_heal(self) -> None:
        state = {
            "type": "decision",
            "decision": "rest_site",
            "context": {"act": 1, "floor": 7, "room_type": "RestSite"},
            "options": [
                {"index": 0, "option_id": "SMITH", "is_enabled": True},
                {"index": 1, "option_id": "HEAL", "is_enabled": True},
            ],
            "player": {"hp": 20, "max_hp": 70, "gold": 99, "deck": []},
        }
        env = reset_env(state)
        policy = HeadlessHybridPolicy(FakeModel())

        index, _ = policy.select(env)

        self.assertEqual(env.legal_actions[index].argument_dict()["option_index"], 1)

    def test_tiny_mailbox_overflow_guard_chooses_smith(self) -> None:
        state = {
            "type": "decision",
            "decision": "rest_site",
            "context": {"act": 1, "floor": 12, "room_type": "RestSite"},
            "options": [
                {"index": 0, "option_id": "HEAL", "is_enabled": True},
                {"index": 1, "option_id": "SMITH", "is_enabled": True},
            ],
            "player": {
                "hp": 20,
                "max_hp": 70,
                "potion_slot_count": 3,
                "potions": [{"index": 0}, {"index": 1}],
                "relics": [{"id": "RELIC.TINY_MAILBOX"}],
                "deck": [],
            },
        }
        env = reset_env(state)
        policy = HeadlessHybridPolicy(FakeModel())

        index, info = policy.select(env)

        self.assertEqual(env.legal_actions[index].argument_dict()["option_index"], 1)
        self.assertEqual(info["runtime_guard"], "tiny_mailbox_overflow_avoid_heal_v0.107.1")

    def test_repeated_event_heuristic_takes_explicit_safe_exit(self) -> None:
        state = {
            "type": "decision",
            "decision": "event_choice",
            "context": {"act": 1, "floor": 4, "room_type": "Event"},
            "event_name": "Tablet of Truth",
            "options": [
                {"index": 0, "title": "Keep Deciphering", "is_locked": False},
                {"index": 1, "title": "Give Up", "is_locked": False},
            ],
            "player": {"hp": 47, "max_hp": 80, "gold": 99, "deck": []},
        }
        env = reset_env(state, "Ironclad")
        policy = HeadlessHybridPolicy(FakeModel())

        index, info = policy.select(env)

        self.assertEqual(info["source"], "heuristic")
        self.assertEqual(env.legal_actions[index].argument_dict()["option_index"], 1)

    def test_repeated_noop_event_action_uses_a_different_legal_option(self) -> None:
        state = {
            "type": "decision",
            "decision": "event_choice",
            "context": {"act": 1, "floor": 12, "room_type": "Event"},
            "event_name": "Whispering Hollow",
            "options": [
                {"index": 0, "title": "Exchange Gold", "is_locked": False},
                {"index": 1, "title": "Hug the Tree", "is_locked": False},
            ],
            "player": {"hp": 20, "max_hp": 80, "gold": 338, "deck": []},
        }
        env = reset_env(state, "Ironclad")
        policy = HeadlessHybridPolicy(FakeModel())

        first_index, first_info = policy.select(env)
        second_index, second_info = policy.select(env)

        self.assertEqual(env.legal_actions[first_index].argument_dict()["option_index"], 0)
        self.assertNotEqual(second_index, first_index)
        self.assertEqual(
            second_info["runtime_guard"],
            "repeated_noop_event_alternate_v0.107.1",
        )
        self.assertNotIn("runtime_guard", first_info)
        with self.assertRaisesRegex(HeadlessRuntimeError, "Every legal event action"):
            policy.select(env)

    def test_shop_runtime_guard_leaves_without_purchasing(self) -> None:
        state = {
            "type": "decision",
            "decision": "shop",
            "context": {"act": 1, "floor": 6, "room_type": "Shop"},
            "cards": [{"index": 0, "name": "Strike", "cost": 10}],
            "relics": [{"index": 0, "name": "Petrified Toad", "cost": 10}],
            "potions": [],
            "card_removal_cost": 10,
            "player": {"hp": 50, "max_hp": 80, "gold": 400, "deck": []},
        }
        env = reset_env(state, "Ironclad")
        policy = HeadlessHybridPolicy(FakeModel())

        index, _ = policy.select(env)

        self.assertEqual(env.legal_actions[index].command, "leave_room")


if __name__ == "__main__":
    unittest.main()
