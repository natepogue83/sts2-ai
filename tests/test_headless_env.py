from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from stsai.headless import HeadlessConfig
from stsai.headless_env import (
    AtomicTrajectoryRecorder,
    HeadlessAction,
    RewardConfig,
    StepResult,
    Sts2Env,
    legal_headless_actions,
    normalize_headless_observation,
)


FORKABLE_RUNTIME = r"""
import json
import sys

print(json.dumps({"type": "ready", "version": "env-test"}), flush=True)
for line in sys.stdin:
    command = json.loads(line)
    if command["cmd"] == "quit":
        print(json.dumps({"type": "quit_result", "success": True}), flush=True)
        break
    if command["cmd"] == "snapshot_checkpoint":
        print(json.dumps({
            "type": "snapshot_checkpoint",
            "schema_version": 1,
            "snapshot_safe": True,
            "stage": "test",
            "reasons": [],
            "semantic": {"test": True},
        }), flush=True)
        continue
    if command["cmd"] == "start_run":
        state = {
            "type": "decision",
            "decision": "event_choice",
            "context": {"act": 1, "floor": 1, "room_type": "Event"},
            "options": [{"index": 0, "text_key": "SAFE", "is_locked": False}],
            "player": {"name": "The Silent", "hp": 70, "max_hp": 70},
        }
    else:
        state = {
            "type": "decision",
            "decision": "game_over",
            "context": {"act": 1, "floor": 1, "room_type": "Event"},
            "victory": True,
            "player": {"name": "The Silent", "hp": 70, "max_hp": 70},
        }
    print(json.dumps(state), flush=True)
"""


class FakeClient:
    version = "test"

    def __init__(self, start: dict, responses: list[dict]):
        self.start = start
        self.responses = list(responses)
        self.actions: list[tuple[str, dict]] = []

    def start_run(self, **_: object) -> dict:
        return self.start

    def action(self, action: str, **arguments: object) -> dict:
        self.actions.append((action, dict(arguments)))
        return self.responses.pop(0)


def event_state() -> dict:
    return {
        "type": "decision",
        "decision": "event_choice",
        "context": {"act": 1, "floor": 1, "room_type": "Event"},
        "options": [
            {"index": 0, "text_key": "SAFE", "is_locked": False},
            {"index": 1, "text_key": "LOCKED", "is_locked": True},
        ],
        "player": {"name": "The Silent", "hp": 70, "max_hp": 70},
    }


class LegalActionTests(unittest.TestCase):
    def test_event_excludes_locked_options(self) -> None:
        actions = legal_headless_actions(event_state())
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].command, "choose_option")
        self.assertEqual(actions[0].argument_dict(), {"option_index": 0})

    def test_combat_expands_targets_and_masks_unplayable_cards(self) -> None:
        state = {
            "type": "decision",
            "decision": "combat_play",
            "hand": [
                {
                    "index": 0,
                    "id": "CARD.STRIKE_SILENT",
                    "can_play": True,
                    "target_type": "AnyEnemy",
                },
                {
                    "index": 1,
                    "id": "CARD.ASCENDERS_BANE",
                    "can_play": False,
                    "target_type": "None",
                },
            ],
            "enemies": [{"index": 0}, {"index": 1}],
            "player": {
                "potions": [
                    {
                        "index": 2,
                        "name": "Fire Potion",
                        "target_type": "AnyEnemy",
                    }
                ]
            },
        }
        actions = legal_headless_actions(state)
        self.assertEqual(
            [action.command for action in actions],
            ["play_card", "play_card", "use_potion", "use_potion", "end_turn"],
        )
        self.assertEqual(
            actions[1].argument_dict(), {"card_index": 0, "target_index": 1}
        )
        self.assertEqual(
            actions[3].argument_dict(), {"potion_index": 2, "target_index": 1}
        )

    def test_card_selection_enumerates_every_allowed_combination(self) -> None:
        state = {
            "decision": "card_select",
            "cards": [{"index": 4}, {"index": 8}, {"index": 9}],
            "min_select": 1,
            "max_select": 2,
        }
        actions = legal_headless_actions(state)
        self.assertEqual(len(actions), 6)
        self.assertEqual(
            {action.argument_dict()["indices"] for action in actions},
            {"4", "8", "9", "4,8", "4,9", "8,9"},
        )

    def test_shop_only_exposes_stocked_affordable_purchases(self) -> None:
        state = {
            "decision": "shop",
            "player": {"gold": 100},
            "cards": [
                {"index": 0, "name": "Cheap", "cost": 50, "is_stocked": True},
                {"index": 1, "name": "Dear", "cost": 150, "is_stocked": True},
            ],
            "relics": [{"index": 0, "name": "Gone", "cost": 10, "is_stocked": False}],
            "potions": [],
            "card_removal_cost": 75,
        }
        actions = legal_headless_actions(state)
        self.assertEqual(
            [action.command for action in actions],
            ["buy_card", "remove_card", "leave_room"],
        )

    def test_merchant_ready_exposes_true_slot_potions_then_open(self) -> None:
        state = {
            "decision": "shop",
            "shop_kind": "merchant_room",
            "shop_stage": "merchant_ready",
            "player": {
                "gold": 0,
                "potions": [
                    {
                        "index": 1,
                        "id": "POTION.FOUL_POTION",
                        "target_type": "TargetedNoCreature",
                        "can_use_now": True,
                    },
                    {
                        "index": 2,
                        "id": "POTION.ATTACK_POTION",
                        "target_type": "Self",
                        "can_use_now": False,
                    },
                ],
            },
            "cards": [{"index": 0, "cost": 0, "is_stocked": True}],
            "relics": [],
            "potions": [],
            "card_removal_cost": 0,
        }
        actions = legal_headless_actions(state)
        self.assertEqual(
            [action.command for action in actions],
            ["use_potion", "open_shop", "leave_room"],
        )
        self.assertEqual(actions[0].argument_dict(), {"potion_index": 1})

        state["shop_stage"] = "inventory_open"
        actions = legal_headless_actions(state)
        self.assertNotIn("use_potion", [action.command for action in actions])
        self.assertNotIn("open_shop", [action.command for action in actions])
        self.assertIn("buy_card", [action.command for action in actions])

    def test_event_substages_export_only_typed_actions(self) -> None:
        reward_state = {
            "decision": "event_reward",
            "rewards": [
                {
                    "index": 0,
                    "reward_type": "PotionReward",
                    "entity_id": "POTION.FOUL_POTION",
                },
                {"index": 1, "reward_type": "GoldReward", "amount": 20},
            ],
            "can_skip": True,
        }
        reward_actions = legal_headless_actions(reward_state)
        self.assertEqual(
            [action.command for action in reward_actions],
            ["claim_reward", "claim_reward", "skip_rewards"],
        )
        self.assertEqual(
            reward_actions[1].argument_dict(), {"reward_index": 1}
        )

        fake_shop = {
            "decision": "fake_shop",
            "event_stage": "fake_shop_ready",
            "inventory_present": True,
            "started_fight": False,
        }
        self.assertEqual(
            [action.command for action in legal_headless_actions(fake_shop)],
            ["open_fake_shop"],
        )
        fake_shop["started_fight"] = True
        self.assertEqual(legal_headless_actions(fake_shop), ())
        self.assertEqual(
            legal_headless_actions({"decision": "event_blocked"}), ()
        )

    def test_finished_event_keeps_proceed_alongside_potion_discard(self) -> None:
        state = {
            "decision": "event_choice",
            "event_stage": "proceed",
            "options": [],
            "player": {"potions": [{"index": 0, "id": "POTION.FIRE"}]},
        }
        self.assertEqual(
            [action.command for action in legal_headless_actions(state)],
            ["discard_potion", "leave_room"],
        )

    def test_treasure_actions_follow_native_stages(self) -> None:
        unopened = {"decision": "treasure", "treasure_stage": "unopened"}
        self.assertEqual(
            [action.command for action in legal_headless_actions(unopened)],
            ["open_chest"],
        )

        ready = {
            "decision": "treasure",
            "treasure_stage": "relic_ready",
            "relics": [{"index": 0, "id": "RELIC.TEST"}],
        }
        take = legal_headless_actions(ready)
        self.assertEqual([action.command for action in take], ["take_chest_relic"])
        self.assertEqual(take[0].argument_dict(), {"relic_index": 0})

        proceed = {"decision": "treasure", "treasure_stage": "proceed"}
        self.assertEqual(
            [action.command for action in legal_headless_actions(proceed)],
            ["leave_room"],
        )
        self.assertEqual(
            legal_headless_actions({"decision": "treasure_blocked"}), ()
        )

        ready["relics"] = []
        self.assertEqual(legal_headless_actions(ready), ())

    def test_observation_has_explicit_canonical_character(self) -> None:
        state = event_state()
        state["player"]["deck"] = [
            {"id": "STRIKE_SILENT", "description": "localized display text"}
        ]
        observation = normalize_headless_observation(
            state, character="Silent", ascension=10, seed="ABC"
        )
        self.assertEqual(observation["character"], "CHARACTER.SILENT")
        self.assertEqual(observation["player"]["character"], "CHARACTER.SILENT")
        self.assertEqual(observation["player"]["deck"][0]["id"], "CARD.STRIKE_SILENT")
        self.assertNotIn("description", observation["player"]["deck"][0])


class EnvironmentTests(unittest.TestCase):
    def test_snapshot_fork_preserves_observation_actions_and_parent_isolation(self) -> None:
        config = HeadlessConfig(
            runtime=Path("unused"),
            command=(sys.executable, "-u", "-c", FORKABLE_RUNTIME),
            startup_timeout=5,
            command_timeout=5,
        )
        with Sts2Env(config) as parent:
            original = parent.reset(character="Silent", ascension=10, seed="ABC")
            original_actions = parent.legal_actions
            snapshot = parent.snapshot()

            self.assertTrue(parent.step(0).terminated)
            with parent.fork(snapshot) as child:
                self.assertNotEqual(parent.client.pid, child.client.pid)
                self.assertEqual(child.observation, original)
                self.assertEqual(child.legal_actions, original_actions)
                self.assertFalse(child.terminated)
                self.assertTrue(child.step(0).terminated)

    def test_reset_step_mask_and_terminal_reward(self) -> None:
        terminal = {
            "type": "decision",
            "decision": "game_over",
            "victory": True,
            "act": 3,
            "floor": 17,
            "player": {"name": "The Silent", "hp": 4, "max_hp": 70},
        }
        client = FakeClient(event_state(), [terminal])
        env = Sts2Env(client=client)
        observation = env.reset(character="Silent", ascension=10, seed="ABC")
        self.assertEqual(observation["decision_type"], "event_choice")
        self.assertEqual(env.action_mask, (True,))

        result = env.step(0)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(client.actions, [("choose_option", {"option_index": 0})])
        self.assertEqual(env.legal_actions, ())
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_defeat_reward_is_configurable(self) -> None:
        terminal = {
            "type": "decision",
            "decision": "game_over",
            "victory": False,
            "player": {},
        }
        env = Sts2Env(
            client=FakeClient(event_state(), [terminal]),
            rewards=RewardConfig(victory=5, defeat=-3),
        )
        env.reset(character="Silent", ascension=0, seed="ABC")
        self.assertEqual(env.step(0).reward, -3)

    def test_card_reward_carries_forward_exact_combat_context(self) -> None:
        combat = {
            "type": "decision",
            "decision": "combat_play",
            "context": {"act": 1, "floor": 2, "room_type": "Monster"},
            "encounter_id": "ENCOUNTER.FUZZY_WURM_CRAWLER_WEAK",
            "round": 3,
            "hand": [],
            "enemies": [
                {"index": 0, "id": "MONSTER.FUZZY_WURM_CRAWLER", "hp": 1}
            ],
            "player": {"hp": 50, "max_hp": 70, "gold": 99, "deck": []},
        }
        reward = {
            "type": "decision",
            "decision": "card_reward",
            "context": {"act": 1, "floor": 2, "room_type": "Monster"},
            "encounter_id": "ENCOUNTER.FUZZY_WURM_CRAWLER_WEAK",
            "turns_taken": 3,
            "cards": [{"index": 0, "id": "CARD.DODGE_AND_ROLL"}],
            "can_skip": True,
            "player": {"hp": 50, "max_hp": 70, "gold": 107, "deck": []},
        }
        env = Sts2Env(client=FakeClient(combat, [reward]))
        env.reset(character="Silent", ascension=10, seed="ABC")

        result = env.step(0)

        self.assertEqual(
            result.observation["previous_combat"],
            {
                "encounter_id": "ENCOUNTER.FUZZY_WURM_CRAWLER_WEAK",
                "monsters": ["MONSTER.FUZZY_WURM_CRAWLER"],
                "turns_taken": 3,
            },
        )


class RecorderTests(unittest.TestCase):
    def test_trajectory_is_only_published_on_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "episode.jsonl"
            recorder = AtomicTrajectoryRecorder(output, {"seed": "ABC"})
            action = HeadlessAction.create(
                "choose_option", "OPTION.SAFE", option_index=0
            )
            observation = {"decision_type": "event_choice"}
            recorder.record_reset(observation, [action])
            result = StepResult(
                {"decision_type": "game_over"},
                1.0,
                True,
                False,
                {"victory": True},
            )
            recorder.record_transition(
                step=1,
                observation=observation,
                legal_actions=[action],
                action_index=0,
                action=action,
                result=result,
                next_legal_actions=[],
            )
            self.assertFalse(output.exists())
            self.assertTrue(recorder.temporary.exists())

            recorder.commit({"steps": 1, "victory": True})
            self.assertTrue(output.exists())
            self.assertFalse(recorder.temporary.exists())
            events = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(
                [event["event_type"] for event in events],
                ["episode_start", "reset", "transition", "episode_end"],
            )
            self.assertEqual(events[2]["action_index"], 0)


if __name__ == "__main__":
    unittest.main()
