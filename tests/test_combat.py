from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from stsai.combat import (
    build_combat_dataset,
    build_headless_combat_dataset,
    headless_combat_trajectory_key,
    legal_combat_actions,
    normalize_combat_observation,
    safe_potion_indexes,
)


def _combat_state() -> dict:
    return {
        "state_type": "monster",
        "run": {"act": 1, "floor": 2, "ascension": 10, "seed": "do-not-copy"},
        "player": {
            "character": "The Necrobinder",
            "player_name": "do-not-copy",
            "hp": 50,
            "max_hp": 66,
            "energy": 2,
            "hand": [
                {
                    "index": 0,
                    "id": "STRIKE_N",
                    "type": "Attack",
                    "cost": "1",
                    "target_type": "AnyEnemy",
                    "can_play": True,
                },
                {
                    "index": 1,
                    "id": "EXPENSIVE",
                    "target_type": "AnyEnemy",
                    "can_play": False,
                },
                {
                    "index": 2,
                    "id": "DEFEND_N",
                    "target_type": "Self",
                    "can_play": True,
                },
            ],
            "potions": [
                {
                    "slot": 0,
                    "id": "FIRE_POTION",
                    "can_use_in_combat": True,
                    "target_type": "AnyEnemy",
                }
            ],
            "pets": [{"id": "OSTY", "hp": 12, "max_hp": 12, "block": 0}],
        },
        "battle": {
            "round": 1,
            "turn": "player",
            "is_play_phase": True,
            "enemies": [
                {"entity_id": "LOUSE_0", "hp": 10, "max_hp": 10},
                {"entity_id": "LOUSE_1", "hp": 0, "max_hp": 10},
            ],
        },
    }


class CombatDatasetTests(unittest.TestCase):
    def test_legal_mask_expands_targets_and_excludes_unplayable_cards(self) -> None:
        actions = legal_combat_actions(_combat_state())
        self.assertEqual(
            actions,
            [
                {"action": "play_card", "card_index": 0, "target": "LOUSE_0"},
                {"action": "play_card", "card_index": 2},
                {"action": "use_potion", "slot": 0, "target": "LOUSE_0"},
                {"action": "end_turn"},
            ],
        )

    def test_character_and_necrobinder_pet_are_kept_but_identity_is_removed(self) -> None:
        normalized = normalize_combat_observation(_combat_state())
        self.assertEqual(normalized["player"]["character"], "CHARACTER.NECROBINDER")
        self.assertEqual(normalized["player"]["pets"][0]["id"], "OSTY")
        self.assertNotIn("player_name", json.dumps(normalized))
        self.assertNotIn("do-not-copy", json.dumps(normalized))

    def test_hand_selection_has_its_own_legal_mask(self) -> None:
        state = {
            "state_type": "hand_select",
            "hand_select": {
                "cards": [{"index": 0}, {"index": 3}],
                "can_confirm": True,
            },
        }
        self.assertEqual(
            legal_combat_actions(state),
            [
                {"action": "combat_select_card", "card_index": 0},
                {"action": "combat_select_card", "card_index": 3},
                {"action": "combat_confirm_selection"},
            ],
        )

    def test_necrobinder_can_target_self_or_osty(self) -> None:
        state = _combat_state()
        state["player"]["combat_id"] = 7
        state["player"]["pets"][0]["combat_id"] = 8
        state["player"]["hand"] = [
            {
                "index": 0,
                "id": "PROTECT_OSTY",
                "target_type": "AnyAlly",
                "can_play": True,
            }
        ]
        state["player"]["potions"] = []
        self.assertEqual(
            legal_combat_actions(state),
            [
                {"action": "play_card", "card_index": 0, "target": "7"},
                {"action": "play_card", "card_index": 0, "target": "8"},
                {"action": "end_turn"},
            ],
        )

    def test_builder_keeps_only_successful_legal_combat_actions(self) -> None:
        state = _combat_state()
        events = [
            {
                "observation": state,
                "action": {"action": "play_card", "card_index": 0, "target": "LOUSE_0"},
                "bridge_response": {"status": "ok"},
                "next_observation": {**state, "player": {**state["player"], "energy": 1}},
                "state_changed": True,
                "terminal": False,
            },
            {
                "observation": state,
                "action": {"action": "play_card", "card_index": 1, "target": "LOUSE_0"},
                "bridge_response": {"status": "ok"},
                "next_observation": state,
                "state_changed": True,
            },
            {
                "observation": state,
                "action": {"action": "end_turn"},
                "bridge_response": {"status": "error"},
                "next_observation": state,
                "state_changed": False,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trajectory.jsonl"
            output = Path(directory) / "combat.jsonl"
            source.write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            summary = build_combat_dataset([source], output, build_id="v0.107.1")
            records = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(summary.decisions_written, 1)
        self.assertEqual(summary.skipped_illegal, 1)
        self.assertEqual(summary.skipped_unsuccessful, 1)
        self.assertEqual(records[0]["chosen_action"]["card_index"], 0)
        self.assertIn(records[0]["split"], {"train", "validation", "test"})

    def test_trajectory_split_stays_stable_when_file_grows(self) -> None:
        state = _combat_state()
        event = {
            "trajectory_id": "stable-session",
            "observation": state,
            "action": {"action": "end_turn"},
            "bridge_response": {"status": "ok"},
            "next_observation": state,
            "state_changed": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trajectory.jsonl"
            first_output = Path(directory) / "first.jsonl"
            second_output = Path(directory) / "second.jsonl"
            source.write_text(json.dumps(event) + "\n", encoding="utf-8")
            build_combat_dataset([source], first_output, build_id="v0.107.1")
            first = json.loads(first_output.read_text())
            source.write_text(json.dumps(event) + "\n" + json.dumps(event) + "\n")
            build_combat_dataset([source], second_output, build_id="v0.107.1")
            second = json.loads(second_output.read_text().splitlines()[0])
        self.assertEqual(first["trajectory_key"], second["trajectory_key"])
        self.assertEqual(first["split"], second["split"])

    def test_passive_human_ui_action_does_not_require_a_synthetic_after_state(self) -> None:
        event = {
            "event_type": "human_combat_action",
            "trajectory_id": "human-session",
            "source": "game_ui",
            "observation": _combat_state(),
            "action": {"action": "end_turn"},
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "human.jsonl"
            output = Path(directory) / "combat.jsonl"
            source.write_text(json.dumps(event) + "\n", encoding="utf-8")
            summary = build_combat_dataset([source], output, build_id="v0.107.1")
            record = json.loads(output.read_text())
        self.assertEqual(summary.decisions_written, 1)
        self.assertEqual(summary.sources, {"human_ui": 1})
        self.assertIsNone(record["next_observation"])

    def test_headless_teacher_trajectory_maps_to_shared_combat_schema(self) -> None:
        observation = {
            "source": "sts2-cli",
            "character": "CHARACTER.REGENT",
            "ascension": 10,
            "seed": "CBTR000001",
            "decision_type": "combat_play",
            "context": {"act": 1, "floor": 3, "room_type": "Monster"},
            "player": {
                "hp": 64,
                "max_hp": 72,
                "block": 2,
                "gold": 99,
                "relics": [{"id": "RELIC.REGAL_PILLOW"}],
                "potions": [],
            },
            "decision_state": {
                "encounter_id": "ENCOUNTER.SLIMES_WEAK",
                "round": 2,
                "energy": 2,
                "max_energy": 3,
                "hand": [
                    {
                        "index": 0,
                        "id": "CARD.STRIKE_REGENT",
                        "type": "Attack",
                        "cost": 1,
                        "can_play": True,
                        "target_type": "AnyEnemy",
                        "upgraded": False,
                    }
                ],
                "enemies": [
                    {
                        "index": 0,
                        "id": "MONSTER.TWIG_SLIME_M",
                        "hp": 9,
                        "max_hp": 14,
                        "block": 0,
                        "intents": [{"type": "Attack", "damage": 5}],
                        "powers": [],
                    }
                ],
                "player_powers": [],
                "draw_pile_count": 6,
                "discard_pile_count": 3,
            },
        }
        events = [
            {
                "event_type": "episode_start",
                "character": "CHARACTER.REGENT",
                "seed": "CBTR000001",
                "policy": "greedy_rules_v1",
            },
            {
                "event_type": "transition",
                "observation": observation,
                "legal_actions": [
                    {
                        "action": "play_card",
                        "option_id": "CARD.STRIKE_REGENT|TARGET.0",
                        "args": {"card_index": 0, "target_index": 0},
                    },
                    {"action": "end_turn", "option_id": "END_TURN"},
                ],
                "action": {
                    "action": "play_card",
                    "option_id": "CARD.STRIKE_REGENT|TARGET.0",
                    "args": {"card_index": 0, "target_index": 0},
                },
                "next_observation": observation,
                "terminated": False,
                "info": {"policy": {"source": "heuristic"}},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "teacher.jsonl"
            output = Path(directory) / "combat.jsonl"
            source.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            summary = build_headless_combat_dataset(
                [source], output, build_id="v0.107.1"
            )
            record = json.loads(output.read_text())

        expected_key = headless_combat_trajectory_key(
            build_id="v0.107.1",
            character="Regent",
            seed="CBTR000001",
        )
        self.assertEqual(summary.decisions_written, 1)
        self.assertEqual(record["trajectory_key"], expected_key)
        self.assertEqual(
            record["observation"]["player"]["character"], "CHARACTER.REGENT"
        )
        self.assertEqual(record["chosen_action"]["target"], "enemy:0")
        self.assertEqual(
            record["observation"]["battle"]["enemies"][0]["entity_id"],
            "enemy:0",
        )
        self.assertEqual(record["patch"]["label_source"], "heuristic_distillation")

    def test_headless_builder_rejects_non_teacher_policy_sources(self) -> None:
        events = [
            {
                "event_type": "episode_start",
                "character": "CHARACTER.SILENT",
                "seed": "CBTS000001",
            },
            {
                "event_type": "transition",
                "observation": {"decision_type": "combat_play"},
                "info": {"policy": {"source": "legal_random"}},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "random.jsonl"
            output = Path(directory) / "combat.jsonl"
            source.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            summary = build_headless_combat_dataset(
                [source], output, build_id="v0.107.1"
            )
        self.assertEqual(summary.decisions_written, 0)
        self.assertEqual(summary.skipped_wrong_source, 1)


class SafePotionIndexTests(unittest.TestCase):
    def test_uses_raw_index_not_list_position_after_consumption(self) -> None:
        # Slot 0 was consumed: remaining potions carry raw indexes 1 and 2 at
        # list positions 0 and 1. The allowlist must key raw indexes.
        player = {
            "potions": [
                {"index": 1, "id": "POTION.ATTACK_POTION"},
                {"index": 2, "id": "POTION.FEAR_POTION"},
            ]
        }
        allowed = safe_potion_indexes(player, frozenset({"POTION.ATTACK_POTION"}))
        self.assertEqual(allowed, {1})

    def test_canonicalizes_raw_ids_and_allowlist_entries(self) -> None:
        player = {"potions": [{"index": 0, "id": "Attack Potion"}]}
        self.assertEqual(
            safe_potion_indexes(player, frozenset({"POTION.ATTACK_POTION"})), {0}
        )
        self.assertEqual(safe_potion_indexes(player, ("attack potion",)), {0})

    def test_unknown_or_missing_ids_fail_closed(self) -> None:
        player = {"potions": [{"index": 0}, {"index": 1, "id": ""}]}
        self.assertEqual(
            safe_potion_indexes(player, frozenset({"POTION.ATTACK_POTION"})), set()
        )
        self.assertEqual(safe_potion_indexes({}, frozenset({"POTION.A"})), set())


if __name__ == "__main__":
    unittest.main()
