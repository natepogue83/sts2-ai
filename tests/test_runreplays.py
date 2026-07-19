from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from stsai.headless_env import HeadlessAction, legal_headless_actions
from stsai.runreplays import (
    ReplayMappingError,
    _prefix_alignment,
    audit_runreplays,
    map_runreplays_command,
    probe_runreplays_reconstruction,
    resolve_runreplays_followup_action,
)


def _snapshot(commands: list[str], *, build: str = "v0.107.1") -> str:
    header = [
        "# Character: DEFECT",
        "# Seed: SECRETSEED",
        "# Ascension: 10",
        f"# Game: {build}",
        "# Mod: 0.1.0",
    ]
    return "\n".join(header + commands) + "\n"


def _run_save(floor: int) -> dict:
    return {
        "schema_version": 16,
        "platform_type": "steam",
        "game_mode": "standard",
        "ascension": 10,
        "rng": {"seed": "SECRETSEED"},
        "players": [{"character_id": "CHARACTER.DEFECT"}],
        "acts": [],
        "map_point_history": [[{} for _ in range(floor - 1)]],
    }


class RunReplaysAuditTests(unittest.TestCase):
    def test_prefix_alignment_quarantines_only_audited_legacy_commands(self) -> None:
        base = [
            "OpenChest",
            "NetPickRelicAction 1",
            "MoveToMapCoord 2",
        ]
        target = ["OpenChest", "MoveToMapCoord 2", "EndTurn"]

        aligned = _prefix_alignment(base, target, max_overlap=0)
        self.assertTrue(aligned["prefix_gate"])
        self.assertEqual(aligned["mode"], "canonical_quarantine")
        self.assertEqual(aligned["target_boundary"], 2)
        self.assertTrue(aligned["no_unknown_commands"])

        with_overlap = _prefix_alignment(base, target, max_overlap=1)
        self.assertFalse(with_overlap["overlap_compatible"])

    def test_prefix_alignment_rejects_known_conflicts_and_unknowns(self) -> None:
        conflict = _prefix_alignment(["EndTurn"], ["PlayCard 1"], max_overlap=0)
        self.assertFalse(conflict["prefix_gate"])

        unknown = _prefix_alignment(
            ["NetPickRelicAction 1", "EndTurn"],
            ["EndTurn", "MysteryCommand 2"],
            max_overlap=0,
        )
        self.assertFalse(unknown["prefix_gate"])
        self.assertFalse(unknown["no_unknown_commands"])

    def test_mapper_uses_persistent_card_and_target_ids(self) -> None:
        state = {
            "decision": "combat_play",
            "hand": [
                {
                    "index": 3,
                    "id": "CARD.STRIKE_IRONCLAD",
                    "combat_card_id": 19,
                    "can_play": True,
                    "target_type": "AnyEnemy",
                },
                {
                    "index": 4,
                    "id": "CARD.DEFEND_IRONCLAD",
                    "combat_card_id": 2,
                    "can_play": True,
                    "target_type": "Self",
                },
            ],
            "enemies": [
                {"index": 0, "id": "MONSTER.A", "combat_id": 2},
                {"index": 1, "id": "MONSTER.B", "combat_id": 7},
            ],
            "player": {"potions": []},
        }
        actions = legal_headless_actions(state)

        targeted = map_runreplays_command(
            state, "PlayCard 19 7", legal_actions=actions
        )
        self.assertEqual(
            targeted.argument_dict(), {"card_index": 3, "target_index": 1}
        )
        untargeted = map_runreplays_command(
            state, "PlayCard 2", legal_actions=actions
        )
        self.assertEqual(untargeted.argument_dict(), {"card_index": 4})
        self.assertEqual(
            map_runreplays_command(state, "EndTurn", legal_actions=actions).command,
            "end_turn",
        )

    def test_mapper_fails_closed_on_missing_or_ambiguous_ids(self) -> None:
        state = {
            "decision": "combat_play",
            "hand": [
                {
                    "index": 0,
                    "id": "CARD.A",
                    "combat_card_id": 4,
                    "can_play": True,
                    "target_type": "Self",
                },
                {
                    "index": 1,
                    "id": "CARD.B",
                    "combat_card_id": 4,
                    "can_play": True,
                    "target_type": "Self",
                },
            ],
            "enemies": [],
            "player": {"potions": []},
        }
        with self.assertRaisesRegex(ReplayMappingError, "not unique") as raised:
            map_runreplays_command(state, "PlayCard 4")
        self.assertEqual(raised.exception.code, "ambiguous_card_id")

        with self.assertRaises(ReplayMappingError) as missing:
            map_runreplays_command(state, "PlayCard 99")
        self.assertEqual(missing.exception.code, "missing_card_id")

        with self.assertRaises(ReplayMappingError) as unsupported:
            map_runreplays_command(state, "OpenChest")
        self.assertEqual(unsupported.exception.code, "wrong_decision")

    def test_event_proceed_sentinel_is_noop_only_after_event_has_left(self) -> None:
        map_state = {
            "decision": "map_select",
            "choices": [{"col": 2, "row": 3, "type": "Monster"}],
        }
        noop = map_runreplays_command(map_state, "ChooseEventOption -1")
        self.assertEqual(noop.command, "replay_noop")

        live_event = {
            "decision": "event_choice",
            "options": [{"index": 0, "text_key": "EVENT.options.ONE"}],
        }
        with self.assertRaises(ReplayMappingError) as still_live:
            map_runreplays_command(live_event, "ChooseEventOption -1")
        self.assertEqual(still_live.exception.code, "illegal_action")

        finished_event = {"decision": "event_choice", "options": []}
        leave = map_runreplays_command(finished_event, "ChooseEventOption -1")
        self.assertEqual(leave.command, "leave_room")

        reward_event = {
            "decision": "event_reward",
            "rewards": [
                {
                    "index": 0,
                    "reward_type": "PotionReward",
                    "entity_id": "POTION.FOUL_POTION",
                }
            ],
            "can_skip": True,
        }
        claim = map_runreplays_command(reward_event, "ClaimReward 0")
        self.assertEqual(claim.command, "claim_reward")
        self.assertEqual(claim.argument_dict(), {"reward_index": 0})
        skip = map_runreplays_command(reward_event, "ChooseEventOption -1")
        self.assertEqual(skip.command, "skip_rewards")

        with self.assertRaises(ReplayMappingError) as absent_reward:
            map_runreplays_command(reward_event, "ClaimReward 1")
        self.assertEqual(absent_reward.exception.code, "illegal_action")

    def test_rest_site_option_requires_one_exact_advertised_id(self) -> None:
        state = {
            "decision": "rest_site",
            "options": [
                {"index": 0, "option_id": "HEAL"},
                {"index": 1, "option_id": "SMITH"},
            ],
        }
        actions = legal_headless_actions(state)
        mapped = map_runreplays_command(
            state, "ChooseRestSiteOption HEAL", legal_actions=actions
        )
        self.assertEqual(mapped.command, "choose_option")
        self.assertEqual(mapped.option_id, "OPTION.HEAL")
        self.assertEqual(mapped.argument_dict(), {"option_index": 0})

        with self.assertRaises(ReplayMappingError) as wrong_state:
            map_runreplays_command(
                {"decision": "event_choice", "options": []},
                "ChooseRestSiteOption HEAL",
            )
        self.assertEqual(wrong_state.exception.code, "wrong_decision")

        with self.assertRaises(ReplayMappingError) as missing_id:
            map_runreplays_command(state, "ChooseRestSiteOption")
        self.assertEqual(missing_id.exception.code, "invalid_command")

        with self.assertRaises(ReplayMappingError) as fuzzy_text:
            map_runreplays_command(state, "ChooseRestSiteOption HEAL NOW")
        self.assertEqual(fuzzy_text.exception.code, "invalid_command")

        duplicate = [
            HeadlessAction.create("choose_option", "OPTION.HEAL", option_index=0),
            HeadlessAction.create("choose_option", "OPTION.HEAL", option_index=1),
        ]
        with self.assertRaises(ReplayMappingError) as ambiguous:
            map_runreplays_command(
                state,
                "ChooseRestSiteOption HEAL",
                legal_actions=duplicate,
            )
        self.assertEqual(ambiguous.exception.code, "ambiguous_action")

    def test_open_shop_requires_native_merchant_ready_stage(self) -> None:
        shop_state = {
            "decision": "shop",
            "shop_kind": "merchant_room",
            "shop_stage": "merchant_ready",
            "player": {"gold": 0},
            "cards": [],
            "relics": [],
            "potions": [],
        }
        mapped = map_runreplays_command(shop_state, "OpenShop")
        self.assertEqual(mapped.command, "open_shop")
        self.assertEqual(mapped.option_id, "OPEN_SHOP")

        with self.assertRaises(ReplayMappingError) as wrong_state:
            map_runreplays_command({"decision": "map_select"}, "OpenShop")
        self.assertEqual(wrong_state.exception.code, "wrong_decision")

        with self.assertRaises(ReplayMappingError) as extra_argument:
            map_runreplays_command(shop_state, "OpenShop 0")
        self.assertEqual(extra_argument.exception.code, "invalid_command")

        opened = {**shop_state, "shop_stage": "inventory_open"}
        with self.assertRaises(ReplayMappingError) as duplicate:
            map_runreplays_command(opened, "OpenShop")
        self.assertEqual(duplicate.exception.code, "wrong_decision")

    def test_shop_foul_potion_uses_true_belt_slot(self) -> None:
        state = {
            "decision": "shop",
            "shop_kind": "merchant_room",
            "shop_stage": "merchant_ready",
            "player": {
                "gold": 0,
                "potions": [
                    {
                        "index": 2,
                        "id": "POTION.FOUL_POTION",
                        "target_type": "TargetedNoCreature",
                        "can_use_now": True,
                    }
                ],
            },
            "cards": [],
            "relics": [],
            "potions": [],
        }
        mapped = map_runreplays_command(state, "UsePotion 2")
        self.assertEqual(mapped.command, "use_potion")
        self.assertEqual(mapped.argument_dict(), {"potion_index": 2})
        with self.assertRaises(ReplayMappingError):
            map_runreplays_command(state, "UsePotion 0")

    def test_select_card_from_screen_requires_attack_potion_proof(self) -> None:
        proof = {
            "stage": "selection_pending",
            "selection_source": "potion",
            "source_model_id": "POTION.ATTACK_POTION",
            "potion_index": 2,
            "selection_generation": 1,
            "option_count": 3,
            "can_skip": True,
            "potion_removed": True,
            "postcondition_verified": True,
        }
        state = {
            "decision": "card_select",
            "selection_source": "potion",
            "min_select": 0,
            "max_select": 1,
            "cards": [
                {"index": 0, "id": "CARD.A", "type": "Attack"},
                {"index": 1, "id": "CARD.B", "type": "Attack"},
                {"index": 2, "id": "CARD.C", "type": "Attack"},
            ],
            "potion_use_proof": proof,
        }
        mapped = map_runreplays_command(state, "SelectCardFromScreen 2")
        self.assertEqual(mapped.command, "select_cards")
        self.assertEqual(mapped.argument_dict(), {"indices": "2"})

        for broken in (
            {**state, "selection_source": None},
            {**state, "cards": state["cards"][:2]},
            {
                **state,
                "potion_use_proof": {**proof, "postcondition_verified": False},
            },
        ):
            with self.assertRaises(ReplayMappingError) as unproven:
                map_runreplays_command(broken, "SelectCardFromScreen 2")
            self.assertEqual(unproven.exception.code, "unproven_selection")

        with self.assertRaises(ReplayMappingError) as skip:
            map_runreplays_command(state, "SelectCardFromScreen -1")
        self.assertEqual(skip.exception.code, "invalid_command")

    def test_open_fake_shop_requires_exact_native_ready_stage(self) -> None:
        ready = {
            "decision": "fake_shop",
            "event_stage": "fake_shop_ready",
            "inventory_present": True,
            "started_fight": False,
        }
        mapped = map_runreplays_command(ready, "OpenFakeShop")
        self.assertEqual(mapped.command, "open_fake_shop")

        for wrong in (
            {"decision": "map_select"},
            {
                "decision": "fake_shop",
                "event_stage": "fake_shop_ready",
                "inventory_present": False,
                "started_fight": False,
            },
            {
                "decision": "fake_shop",
                "event_stage": "fake_shop_ready",
                "inventory_present": True,
                "started_fight": True,
            },
        ):
            with self.assertRaises(ReplayMappingError):
                map_runreplays_command(wrong, "OpenFakeShop")

    def test_chest_commands_require_exact_native_stages(self) -> None:
        unopened = {"decision": "treasure", "treasure_stage": "unopened"}
        opened = map_runreplays_command(unopened, "OpenChest")
        self.assertEqual(opened.command, "open_chest")

        ready = {
            "decision": "treasure",
            "treasure_stage": "relic_ready",
            "relics": [{"index": 0, "id": "RELIC.TEST"}],
        }
        taken = map_runreplays_command(ready, "TakeChestRelic")
        self.assertEqual(taken.command, "take_chest_relic")
        self.assertEqual(taken.argument_dict(), {"relic_index": 0})

        for command, wrong in (
            ("OpenChest", ready),
            ("TakeChestRelic", unopened),
            ("TakeChestRelic", {"decision": "treasure", "treasure_stage": "proceed"}),
        ):
            with self.assertRaises(ReplayMappingError) as raised:
                map_runreplays_command(wrong, command)
            self.assertEqual(raised.exception.code, "wrong_decision")

        with self.assertRaises(ReplayMappingError) as extra:
            map_runreplays_command(unopened, "OpenChest 0")
        self.assertEqual(extra.exception.code, "invalid_command")

    def test_treasure_to_map_uses_verified_two_stage_adapter(self) -> None:
        treasure = {"decision": "treasure", "treasure_stage": "proceed"}
        adapter = map_runreplays_command(treasure, "MoveToMapCoord 2")
        self.assertEqual(adapter.command, "replay_two_stage")
        self.assertEqual(adapter.option_id, "RUNREPLAYS.TREASURE_TO_MAP.2")
        self.assertEqual(
            adapter.argument_dict(), {"col": 2, "first_action": "leave_room"}
        )

        map_state = {
            "decision": "map_select",
            "choices": [{"col": 2, "row": 4, "type": "Monster"}],
        }
        followup = resolve_runreplays_followup_action(adapter, map_state)
        self.assertEqual(followup.command, "select_map_node")
        self.assertEqual(followup.argument_dict(), {"col": 2, "row": 4})

    def test_legacy_net_pick_echo_requires_complete_treasure_proof(self) -> None:
        proven = {
            "decision": "treasure",
            "treasure_stage": "proceed",
            "treasure_proof": {
                "stage": "relic_awarded",
                "result_player_verified": True,
                "session_closed": True,
                "inventory_delta": 1,
                "postcondition_verified": True,
            },
        }
        echo = map_runreplays_command(
            proven, "NetPickRelicAction for player 1 index 0"
        )
        self.assertEqual(echo.command, "replay_noop")
        self.assertEqual(
            echo.option_id, "RUNREPLAYS.VERIFIED_NET_PICK_RELIC_ECHO"
        )

        malformed = "NetPickRelicAction for player 2 index 0"
        with self.assertRaises(ReplayMappingError) as wrong_player:
            map_runreplays_command(proven, malformed)
        self.assertEqual(wrong_player.exception.code, "quarantined_command")

        for key in (
            "result_player_verified",
            "session_closed",
            "inventory_delta",
            "postcondition_verified",
        ):
            incomplete = {
                **proven,
                "treasure_proof": dict(proven["treasure_proof"]),
            }
            incomplete["treasure_proof"].pop(key)
            with self.assertRaises(ReplayMappingError) as unproven:
                map_runreplays_command(
                    incomplete, "NetPickRelicAction for player 1 index 0"
                )
            self.assertEqual(unproven.exception.code, "quarantined_command")

        wrong_stage = {**proven, "treasure_stage": "relic_ready"}
        with self.assertRaises(ReplayMappingError) as early:
            map_runreplays_command(
                wrong_stage, "NetPickRelicAction for player 1 index 0"
            )
        self.assertEqual(early.exception.code, "quarantined_command")

    def test_shop_commands_require_unique_advertised_exact_matches(self) -> None:
        state = {
            "decision": "shop",
            "player": {"gold": 200},
            "cards": [
                {
                    "index": 1,
                    "id": "CARD.FEAR",
                    "name": "Fear",
                    "cost": 60,
                }
            ],
            "relics": [
                {
                    "index": 0,
                    "id": "RELIC.ALPHA",
                    "name": "Alpha Stone",
                    "cost": 50,
                },
                {
                    "index": 2,
                    "id": "RELIC.BETA",
                    "name": "Beta Idol",
                    "cost": 75,
                },
            ],
            "potions": [],
            "card_removal_cost": 100,
        }
        actions = legal_headless_actions(state)

        relic = map_runreplays_command(
            state, "BuyRelic Beta Idol", legal_actions=actions
        )
        self.assertEqual(relic.command, "buy_relic")
        self.assertEqual(relic.argument_dict(), {"relic_index": 2})

        removal = map_runreplays_command(
            state, "BuyCardRemoval", legal_actions=actions
        )
        self.assertEqual(removal.command, "remove_card")

        card = map_runreplays_command(state, "BuyCard Fear", legal_actions=actions)
        self.assertEqual(card.command, "buy_card")
        self.assertEqual(card.argument_dict(), {"card_index": 1})

        with self.assertRaises(ReplayMappingError) as missing_card_title:
            map_runreplays_command(state, "BuyCard", legal_actions=actions)
        self.assertEqual(missing_card_title.exception.code, "invalid_command")

        with self.assertRaises(ReplayMappingError) as wrong_card_state:
            map_runreplays_command(
                {"decision": "map_select", "cards": state["cards"]},
                "BuyCard Fear",
            )
        self.assertEqual(wrong_card_state.exception.code, "wrong_decision")

        duplicate_card = dict(state)
        duplicate_card["cards"] = [
            state["cards"][0],
            {"index": 2, "id": "CARD.FEAR_2", "name": "Fear", "cost": 60},
        ]
        with self.assertRaises(ReplayMappingError) as ambiguous_card:
            map_runreplays_command(duplicate_card, "BuyCard Fear")
        self.assertEqual(ambiguous_card.exception.code, "ambiguous_action")

        unaffordable_card = dict(state)
        unaffordable_card["cards"] = [
            {"index": 1, "id": "CARD.FEAR", "name": "Fear", "cost": 250}
        ]
        with self.assertRaises(ReplayMappingError) as card_too_expensive:
            map_runreplays_command(unaffordable_card, "BuyCard Fear")
        self.assertEqual(card_too_expensive.exception.code, "illegal_action")

        with self.assertRaises(ReplayMappingError) as wrong_case:
            map_runreplays_command(
                state, "BuyRelic beta idol", legal_actions=actions
            )
        self.assertEqual(wrong_case.exception.code, "illegal_action")

        with self.assertRaises(ReplayMappingError) as missing_title:
            map_runreplays_command(state, "BuyRelic", legal_actions=actions)
        self.assertEqual(missing_title.exception.code, "invalid_command")

        duplicate = dict(state)
        duplicate["relics"] = [
            state["relics"][0],
            {"index": 1, "id": "RELIC.GAMMA", "name": "Alpha Stone", "cost": 50},
        ]
        with self.assertRaises(ReplayMappingError) as ambiguous:
            map_runreplays_command(duplicate, "BuyRelic Alpha Stone")
        self.assertEqual(ambiguous.exception.code, "ambiguous_action")

        unavailable_removal = dict(state)
        unavailable_removal["card_removal_cost"] = 250
        with self.assertRaises(ReplayMappingError) as unaffordable:
            map_runreplays_command(unavailable_removal, "BuyCardRemoval")
        self.assertEqual(unaffordable.exception.code, "illegal_action")

    def test_shop_map_move_is_a_fail_closed_two_stage_adapter(self) -> None:
        shop_state = {
            "decision": "shop",
            "player": {"gold": 0},
            "cards": [],
            "relics": [],
            "potions": [],
        }
        adapter = map_runreplays_command(shop_state, "MoveToMapCoord 2")
        self.assertEqual(adapter.command, "replay_two_stage")
        self.assertEqual(
            adapter.argument_dict(), {"col": 2, "first_action": "leave_room"}
        )

        map_state = {
            "decision": "map_select",
            "choices": [
                {"col": 1, "row": 4, "type": "Monster"},
                {"col": 2, "row": 4, "type": "Event"},
            ],
        }
        followup = resolve_runreplays_followup_action(adapter, map_state)
        self.assertEqual(followup.command, "select_map_node")
        self.assertEqual(followup.argument_dict(), {"col": 2, "row": 4})

        no_leave = [HeadlessAction.create("buy_card", "CARD.A", card_index=0)]
        with self.assertRaises(ReplayMappingError) as missing_first_stage:
            map_runreplays_command(
                shop_state, "MoveToMapCoord 2", legal_actions=no_leave
            )
        self.assertEqual(missing_first_stage.exception.code, "illegal_action")

        duplicate_leave = [
            HeadlessAction.create("leave_room", "LEAVE_ROOM.A"),
            HeadlessAction.create("leave_room", "LEAVE_ROOM.B"),
        ]
        with self.assertRaises(ReplayMappingError) as ambiguous_first_stage:
            map_runreplays_command(
                shop_state, "MoveToMapCoord 2", legal_actions=duplicate_leave
            )
        self.assertEqual(ambiguous_first_stage.exception.code, "ambiguous_action")

        with self.assertRaises(ReplayMappingError) as wrong_followup_state:
            resolve_runreplays_followup_action(adapter, shop_state)
        self.assertEqual(
            wrong_followup_state.exception.code, "wrong_followup_decision"
        )

        duplicate_column = {
            "decision": "map_select",
            "choices": [
                {"col": 2, "row": 4, "type": "Event"},
                {"col": 2, "row": 5, "type": "Monster"},
            ],
        }
        with self.assertRaises(ReplayMappingError) as ambiguous_second_stage:
            resolve_runreplays_followup_action(adapter, duplicate_column)
        self.assertEqual(ambiguous_second_stage.exception.code, "ambiguous_action")

    def test_discard_potion_requires_exact_advertised_event_slot(self) -> None:
        state = {
            "decision": "event_choice",
            "options": [{"index": 0, "text_key": "EVENT.options.ONE"}],
            "player": {
                "potions": [
                    {"index": 0, "id": "POTION.FIRE"},
                    {"index": 2, "id": "POTION.BLOCK"},
                ]
            },
        }
        actions = legal_headless_actions(state)
        mapped = map_runreplays_command(
            state, "DiscardPotion 2", legal_actions=actions
        )
        self.assertEqual(mapped.command, "discard_potion")
        self.assertEqual(mapped.argument_dict(), {"potion_index": 2})

        with self.assertRaises(ReplayMappingError) as missing_slot:
            map_runreplays_command(
                state, "DiscardPotion 1", legal_actions=actions
            )
        self.assertEqual(missing_slot.exception.code, "illegal_action")

        non_event_state = {
            "decision": "map_select",
            "choices": [],
            "player": state["player"],
        }
        with self.assertRaises(ReplayMappingError) as unadvertised:
            map_runreplays_command(non_event_state, "DiscardPotion 2")
        self.assertEqual(unadvertised.exception.code, "illegal_action")

    def test_reward_commands_require_typed_native_queue(self) -> None:
        state = {
            "decision": "card_reward",
            "cards": [
                {"index": 0, "id": "CARD.A"},
                {"index": 1, "id": "CARD.B"},
                {"index": 2, "id": "CARD.C"},
            ],
            "can_skip": True,
            "replay_reward_queue": [
                {
                    "reward_type": "GoldReward",
                    "disposition": "auto_collected",
                    "selection_result": True,
                    "successfully_selected": True,
                    "postcondition_verified": True,
                },
                {
                    "reward_type": "CardReward",
                    "disposition": "card_reward_open",
                    "postcondition_verified": True,
                },
            ],
        }
        claim = map_runreplays_command(state, "ClaimReward 0")
        self.assertEqual(claim.command, "replay_noop")
        self.assertEqual(
            claim.argument_dict()["disposition"], "auto_collected"
        )

        state["_runreplays_reward_queue"] = []
        take = map_runreplays_command(state, "TakeCard 1")
        self.assertEqual(take.command, "select_card_reward")
        self.assertEqual(take.argument_dict(), {"card_index": 1})
        skip = map_runreplays_command(state, "TakeCard skip")
        self.assertEqual(skip.command, "skip_card_reward")

    def test_reward_queue_preserves_proven_retained_visible_index(self) -> None:
        retained = {
            "reward_type": "PotionReward",
            "disposition": "retained_unselected",
            "selection_result": False,
            "successfully_selected": False,
            "postcondition_verified": True,
        }
        card = {
            "reward_type": "CardReward",
            "disposition": "card_reward_open",
            "postcondition_verified": True,
        }
        state = {
            "decision": "card_reward",
            "cards": [{"index": 0, "id": "CARD.A"}],
            "can_skip": True,
            "_runreplays_reward_queue": [retained, card],
        }
        rejected = map_runreplays_command(state, "ClaimReward 0")
        self.assertEqual(rejected.argument_dict()["disposition"], "retained_unselected")
        opened = map_runreplays_command(state, "ClaimReward 1")
        self.assertEqual(opened.argument_dict()["disposition"], "card_reward_open")

        state["_runreplays_reward_queue"] = [retained]
        take = map_runreplays_command(state, "TakeCard 0")
        self.assertEqual(take.command, "select_card_reward")

        unproven = dict(retained)
        unproven.pop("postcondition_verified")
        state["_runreplays_reward_queue"] = [unproven]
        with self.assertRaises(ReplayMappingError) as missing_proof:
            map_runreplays_command(state, "ClaimReward 0")
        self.assertEqual(missing_proof.exception.code, "unproven_reward")

    def test_reward_commands_fail_on_pending_or_unhandled_rewards(self) -> None:
        state = {
            "decision": "card_reward",
            "cards": [{"index": 0, "id": "CARD.A"}],
            "can_skip": True,
            "replay_reward_queue": [
                {
                    "reward_type": "SpecialCardReward",
                    "disposition": "unhandled",
                    "postcondition_verified": False,
                }
            ],
        }
        with self.assertRaises(ReplayMappingError) as unhandled:
            map_runreplays_command(state, "ClaimReward 0")
        self.assertEqual(unhandled.exception.code, "unhandled_reward")

        with self.assertRaises(ReplayMappingError) as pending:
            map_runreplays_command(state, "TakeCard 0")
        self.assertEqual(pending.exception.code, "pending_reward_claims")

        state.pop("replay_reward_queue")
        with self.assertRaises(ReplayMappingError) as missing:
            map_runreplays_command(state, "ClaimReward 0")
        self.assertEqual(missing.exception.code, "missing_reward_provenance")

    def test_probe_rejects_non_prefix_stream_before_starting_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "floor_2"
            target = root / "floor_3"
            base.mkdir()
            target.mkdir()
            (base / "actions.sts2replay").write_text(
                _snapshot(["EndTurn"]), encoding="utf-8"
            )
            (target / "actions.sts2replay").write_text(
                _snapshot(["PlayCard 1"]), encoding="utf-8"
            )
            (base / "run.save").write_text(
                json.dumps(_run_save(2)), encoding="utf-8"
            )
            (target / "run.save").write_text(
                json.dumps(_run_save(3)), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "prefix_gate"):
                probe_runreplays_reconstruction(
                    base / "actions.sts2replay",
                    target / "actions.sts2replay",
                    root / "probe.json",
                    expected_build="v0.107.1",
                )
            self.assertFalse((root / "probe.json").exists())

    def test_audit_deduplicates_cumulative_snapshots_without_leaking_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "logs" / "SECRETSEED" / "floor_2"
            second = root / "logs" / "SECRETSEED" / "floor_3"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "actions.sts2replay").write_text(
                _snapshot(
                    [
                        "PlayCard 7 1 || Hand: [Zap]",
                        "VoteForMapCoordAction 1",
                        "EndTurn",
                    ]
                ),
                encoding="utf-8",
            )
            (second / "actions.sts2replay").write_text(
                _snapshot(
                    [
                        "PlayCard 7 1 # CARD.ZAP",
                        "EndTurn",
                        "UsePotion 0 1 # POTION.FIRE",
                    ]
                ),
                encoding="utf-8",
            )
            (first / "run.save").write_text(
                json.dumps(_run_save(2)), encoding="utf-8"
            )
            (second / "run.save").write_text(
                json.dumps(_run_save(3)), encoding="utf-8"
            )
            license_file = root / "LICENSE"
            license_file.write_text("MIT License", encoding="utf-8")
            output = root / "audit.json"

            report = audit_runreplays(
                [root / "logs"],
                output,
                expected_build="v0.107.1",
                source_uri="https://example.invalid/replays",
                source_revision="abc123",
                license_spdx="MIT",
                license_file=license_file,
            )

            overall = report["overall"]
            self.assertEqual(overall["raw_commands"], 6)
            self.assertEqual(overall["supported_commands"], 5)
            self.assertEqual(overall["unique_commands_after_snapshot_dedup"], 3)
            self.assertEqual(overall["duplicate_snapshot_commands"], 2)
            self.assertEqual(overall["snapshot_dedup_rate"], 0.4)
            self.assertEqual(overall["prefix_conforming_snapshots"], 2)
            self.assertEqual(overall["unsupported_commands"], 1)
            self.assertEqual(
                overall["raw_diagnostic_prestate_coverage"][
                    "with_diagnostic_prestate"
                ],
                1,
            )
            self.assertEqual(
                overall["combat_action_identity"]["syntactically_parsed"], 3
            )
            self.assertEqual(overall["legal_action_reconstruction"]["success"], 0)
            self.assertEqual(overall["trainable_combat_decisions"], 0)
            self.assertFalse(report["training_authorized"])
            self.assertTrue(report["exact_build_gate"]["passed"])
            self.assertEqual(overall["run_save_sidecars"]["valid"], 2)
            self.assertTrue(
                report["provenance"]["license"]["repository_license_verified"]
            )
            self.assertNotIn("SECRETSEED", output.read_text(encoding="utf-8"))

    def test_audit_reports_wrong_build_unknown_and_invalid_combat_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "floor_1"
            snapshot.mkdir()
            (snapshot / "actions.sts2replay").write_text(
                _snapshot(["PlayCard nope", "MysteryCommand 3"], build="v0.108.0"),
                encoding="utf-8",
            )

            report = audit_runreplays(
                [root], root / "audit.json", expected_build="v0.107.1"
            )

            self.assertEqual(report["overall"]["files_matching_expected_build"], 0)
            self.assertEqual(report["overall"]["unknown_commands"], 1)
            self.assertEqual(report["overall"]["invalid_combat_commands"], 1)
            encoded = json.dumps(report)
            self.assertNotIn("PlayCard nope", encoded)
            self.assertNotIn("MysteryCommand", encoded)


if __name__ == "__main__":
    unittest.main()
