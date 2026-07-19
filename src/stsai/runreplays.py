"""Privacy-safe audit tooling for RunReplays command snapshots."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, NoReturn, Sequence

from stsai.headless import HeadlessConfig, Sts2HeadlessClient
from stsai.headless_env import (
    HeadlessAction,
    legal_headless_actions,
    normalize_headless_observation,
)
from stsai.ids import canonical_character


REPLAY_FILENAME = "actions.sts2replay"
COMBAT_COMMANDS = frozenset({"PlayCard", "EndTurn", "UsePotion"})
AUDITED_UNSUPPORTED_COMMANDS = frozenset(
    {"NetPickRelicAction", "VoteForMapCoordAction"}
)
KNOWN_COMMANDS = frozenset(
    {
        "BuyCard",
        "BuyCardRemoval",
        "BuyPotion",
        "BuyRelic",
        "CancelGridSelection",
        "ChooseEventOption",
        "ChooseRestSiteOption",
        "ClaimReward",
        "ClickGridCard",
        "CloseShop",
        "ConfirmGridSelection",
        "CrystalSphereClick",
        "DiscardPotion",
        "EndTurn",
        "MoveToMapCoord",
        "OpenChest",
        "OpenFakeShop",
        "OpenShop",
        "PlayCard",
        "ProceedToMap",
        "ProceedToNextAct",
        "SelectCardFromScreen",
        "SelectGridCard",
        "SelectHandCards",
        "SkipRewards",
        "TakeCard",
        "TakeChestRelic",
        "UsePotion",
    }
)
REQUIRED_HEADERS = frozenset(
    {"character", "seed", "ascension", "game", "mod"}
)


class ReplayMappingError(ValueError):
    """A replay command cannot be mapped to exactly one advertised legal action."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _objects(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _run_save_sidecar(
    path: Path,
    *,
    headers: Mapping[str, str],
    character: str,
    snapshot_floor: int | None,
) -> dict[str, Any]:
    save_path = path.with_name("run.save")
    if not save_path.is_file():
        return {"status": "missing", "all_crosschecks_passed": False}
    try:
        value = json.loads(save_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {
            "status": "invalid",
            "file_sha256": _sha256_file(save_path),
            "all_crosschecks_passed": False,
        }
    if not isinstance(value, Mapping):
        return {
            "status": "invalid",
            "file_sha256": _sha256_file(save_path),
            "all_crosschecks_passed": False,
        }

    players = _objects(value.get("players"))
    rng = value.get("rng") if isinstance(value.get("rng"), Mapping) else {}
    acts = [str(act.get("id")) for act in _objects(value.get("acts"))]
    history = value.get("map_point_history")
    history = history if isinstance(history, list) else []
    save_floor = sum(len(column) for column in history if isinstance(column, list)) + 1
    declared_acts = [
        item.strip() for item in headers.get("acts", "").split(",") if item.strip()
    ]
    checks: dict[str, bool] = {
        "seed": str(rng.get("seed", "")) == headers.get("seed", ""),
        "character": bool(players)
        and canonical_character(players[0].get("character_id")) == character,
        "ascension": str(value.get("ascension", ""))
        == headers.get("ascension", ""),
        "floor": snapshot_floor is not None and save_floor == snapshot_floor,
    }
    if declared_acts:
        checks["acts"] = acts == declared_acts
    return {
        "status": "checked",
        "file_sha256": _sha256_file(save_path),
        "schema_version": value.get("schema_version"),
        "platform_type": value.get("platform_type"),
        "game_mode": value.get("game_mode"),
        "player_count": len(players),
        "crosschecks": checks,
        "all_crosschecks_passed": all(checks.values()),
    }


def _discover(inputs: Sequence[str | os.PathLike[str]]) -> list[Path]:
    paths: dict[str, Path] = {}
    for value in inputs:
        root = Path(value)
        if root.is_file():
            candidates: Iterable[Path] = [root]
        elif root.is_dir():
            candidates = root.rglob(REPLAY_FILENAME)
        else:
            raise FileNotFoundError(root)
        for path in candidates:
            if path.is_file() and path.name.casefold() == REPLAY_FILENAME:
                paths[str(path.resolve()).casefold()] = path
    return sorted(paths.values(), key=lambda path: str(path).casefold())


def _floor(path: Path) -> int | None:
    match = re.fullmatch(r"floor_(\d+)", path.parent.name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _command_name(command: str) -> str:
    return command.split(maxsplit=1)[0] if command else ""


def _valid_uint(value: str) -> bool:
    return value.isdigit() and int(value) >= 0


def _valid_combat_command(command: str) -> bool:
    parts = command.split()
    if not parts:
        return False
    if parts[0] == "EndTurn":
        return len(parts) == 1
    if parts[0] in {"PlayCard", "UsePotion"}:
        return len(parts) in {2, 3} and all(_valid_uint(part) for part in parts[1:])
    return False


def _prefix_alignment(
    base_commands: Sequence[str],
    target_commands: Sequence[str],
    *,
    max_overlap: int,
) -> dict[str, Any]:
    raw_prefix = list(base_commands) == list(target_commands[: len(base_commands)])
    base_recognized = [
        command for command in base_commands if _command_name(command) in KNOWN_COMMANDS
    ]
    target_recognized = [
        command for command in target_commands if _command_name(command) in KNOWN_COMMANDS
    ]
    canonical_prefix = base_recognized == target_recognized[: len(base_recognized)]
    no_unknown_commands = all(
        _command_name(command) in KNOWN_COMMANDS
        or _command_name(command) in AUDITED_UNSUPPORTED_COMMANDS
        for command in (*base_commands, *target_commands)
    )
    mode = "raw" if raw_prefix else "canonical_quarantine"
    prefix_gate = raw_prefix or (canonical_prefix and no_unknown_commands)
    overlap_compatible = max_overlap == 0 or raw_prefix

    target_boundary: int | None = None
    if raw_prefix:
        target_boundary = len(base_commands)
    elif prefix_gate:
        recognized = 0
        if not base_recognized:
            target_boundary = 0
        else:
            for index, command in enumerate(target_commands):
                if _command_name(command) in KNOWN_COMMANDS:
                    recognized += 1
                if recognized == len(base_recognized):
                    target_boundary = index + 1
                    break
    return {
        "mode": mode,
        "raw_prefix_exact": raw_prefix,
        "canonical_prefix_exact": canonical_prefix,
        "no_unknown_commands": no_unknown_commands,
        "prefix_gate": prefix_gate,
        "overlap_compatible": overlap_compatible,
        "base_raw_commands": len(base_commands),
        "base_recognized_commands": len(base_recognized),
        "target_raw_commands": len(target_commands),
        "target_recognized_commands": len(target_recognized),
        "target_boundary": target_boundary,
    }


def _mapping_error(code: str, message: str) -> NoReturn:
    raise ReplayMappingError(code, message)


def _int_argument(value: str, *, command: str, name: str) -> int:
    if not _valid_uint(value):
        _mapping_error(
            "invalid_command",
            f"{command} requires an unsigned integer {name}",
        )
    return int(value)


def _unique_action(
    candidates: Sequence[HeadlessAction],
    *,
    command: str,
) -> HeadlessAction:
    if not candidates:
        _mapping_error(
            "illegal_action",
            f"{command} did not match any advertised legal action",
        )
    if len(candidates) != 1:
        _mapping_error(
            "ambiguous_action",
            f"{command} matched {len(candidates)} advertised legal actions",
        )
    return candidates[0]


def _matching_creature_index(
    state: Mapping[str, Any], *, combat_id: int, command: str
) -> int:
    matches = [
        int(enemy.get("index", fallback))
        for fallback, enemy in enumerate(_objects(state.get("enemies")))
        if enemy.get("combat_id") is not None
        and int(enemy["combat_id"]) == combat_id
    ]
    if not matches:
        _mapping_error(
            "missing_target_id",
            f"{command} target combat id is absent from the current enemies",
        )
    if len(matches) != 1:
        _mapping_error(
            "ambiguous_target_id",
            f"{command} target combat id is not unique",
        )
    return matches[0]


def _reward_replay_queue(state: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    if "_runreplays_reward_queue" in state:
        value = state.get("_runreplays_reward_queue")
    elif "replay_reward_queue" in state:
        value = state.get("replay_reward_queue")
    else:
        return None
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        return None
    return [item for item in value if isinstance(item, Mapping)]


def _reward_disposition_is_proven(item: Mapping[str, Any]) -> bool:
    disposition = str(item.get("disposition", ""))
    if disposition in {"auto_collected", "direct_collected"}:
        return (
            item.get("selection_result") is True
            and item.get("successfully_selected") is True
            and item.get("postcondition_verified") is True
        )
    if disposition == "retained_unselected":
        return (
            item.get("selection_result") is False
            and item.get("successfully_selected") is False
            and item.get("postcondition_verified") is True
        )
    if disposition == "card_reward_open":
        return item.get("postcondition_verified") is True
    return False


def _event_reward_claim_is_proven(
    state: Mapping[str, Any], *, reward_index: int
) -> bool:
    proof = state.get("reward_claim_proof")
    if not isinstance(proof, Mapping) or proof.get("reward_index") != reward_index:
        return False
    disposition = str(proof.get("disposition", ""))
    if disposition == "native_collected":
        return (
            proof.get("selection_result") is True
            and proof.get("successfully_selected") is True
            and proof.get("postcondition_verified") is True
        )
    if disposition == "retained_unselected":
        return (
            proof.get("selection_result") is False
            and proof.get("successfully_selected") is False
            and proof.get("postcondition_verified") is True
        )
    if disposition == "card_reward_open":
        return proof.get("postcondition_verified") is True
    return False


def _treasure_relic_award_is_proven(state: Mapping[str, Any]) -> bool:
    proof = state.get("treasure_proof")
    return (
        str(state.get("decision", "")) == "treasure"
        and state.get("treasure_stage") == "proceed"
        and isinstance(proof, Mapping)
        and proof.get("stage") == "relic_awarded"
        and proof.get("result_player_verified") is True
        and proof.get("session_closed") is True
        and proof.get("inventory_delta") == 1
        and proof.get("postcondition_verified") is True
    )


def _potion_use_is_proven(state: Mapping[str, Any], *, potion_index: int) -> bool:
    proof = state.get("potion_use_proof")
    return (
        isinstance(proof, Mapping)
        and proof.get("stage") in {"completed", "selection_pending"}
        and proof.get("potion_index") == potion_index
        and proof.get("potion_removed") is True
        and proof.get("postcondition_verified") is True
    )


def _potion_screen_selection_is_proven(
    state: Mapping[str, Any], *, selected_index: int
) -> bool:
    proof = state.get("potion_use_proof")
    return (
        isinstance(proof, Mapping)
        and proof.get("stage") == "selection_completed"
        and proof.get("selection_source") == "potion"
        and proof.get("source_model_id") == "POTION.ATTACK_POTION"
        and proof.get("selected_index") == selected_index
        and proof.get("selected_reference_in_hand") is True
        and proof.get("hand_delta") == 1
        and proof.get("potion_removed") is True
        and proof.get("executor_settled") is True
        and proof.get("postcondition_verified") is True
    )


def map_runreplays_command(
    state: Mapping[str, Any],
    command: str,
    *,
    legal_actions: Sequence[HeadlessAction] | None = None,
) -> HeadlessAction:
    """Map one RunReplays command to one native action or typed replay adapter.

    The mapper intentionally implements only commands for which the native
    decision state carries enough information to prove the correspondence.
    A replay adapter is permitted only when each native stage is proven against
    the legal actions exported at that stage. Unknown or plausible mappings
    fail closed.
    """

    actions = tuple(legal_actions or legal_headless_actions(state))
    parts = command.split()
    if not parts:
        _mapping_error("invalid_command", "Replay command is empty")
    name = parts[0]

    if name == "EndTurn":
        if len(parts) != 1:
            _mapping_error("invalid_command", "EndTurn takes no arguments")
        return _unique_action(
            [action for action in actions if action.command == "end_turn"],
            command=name,
        )

    if name == "MoveToMapCoord":
        if len(parts) != 2:
            _mapping_error("invalid_command", "MoveToMapCoord requires one column")
        col = _int_argument(parts[1], command=name, name="column")
        decision = str(state.get("decision", ""))
        treasure_proceed = (
            decision == "treasure" and state.get("treasure_stage") == "proceed"
        )
        if decision == "shop" or treasure_proceed:
            leave = _unique_action(
                [action for action in actions if action.command == "leave_room"],
                command=name,
            )
            room_kind = "TREASURE" if treasure_proceed else "SHOP"
            return HeadlessAction.create(
                "replay_two_stage",
                f"RUNREPLAYS.{room_kind}_TO_MAP.{col}",
                col=col,
                first_action=leave.command,
            )
        return _unique_action(
            [
                action
                for action in actions
                if action.command == "select_map_node"
                and action.argument_dict().get("col") == col
            ],
            command=name,
        )

    if name == "OpenShop":
        if len(parts) != 1:
            _mapping_error("invalid_command", "OpenShop takes no arguments")
        if (
            str(state.get("decision", "")) != "shop"
            or state.get("shop_kind") != "merchant_room"
            or state.get("shop_stage") != "merchant_ready"
        ):
            _mapping_error(
                "wrong_decision",
                "OpenShop requires the native merchant-ready stage",
            )
        return _unique_action(
            [action for action in actions if action.command == "open_shop"],
            command=name,
        )

    if name == "OpenFakeShop":
        if len(parts) != 1:
            _mapping_error("invalid_command", "OpenFakeShop takes no arguments")
        if str(state.get("decision", "")) != "fake_shop":
            _mapping_error(
                "wrong_decision",
                "OpenFakeShop requires the native FakeMerchant ready stage",
            )
        return _unique_action(
            [action for action in actions if action.command == "open_fake_shop"],
            command=name,
        )

    if name == "OpenChest":
        if len(parts) != 1:
            _mapping_error("invalid_command", "OpenChest takes no arguments")
        if (
            str(state.get("decision", "")) != "treasure"
            or state.get("treasure_stage") != "unopened"
        ):
            _mapping_error(
                "wrong_decision",
                "OpenChest requires the native unopened treasure stage",
            )
        return _unique_action(
            [action for action in actions if action.command == "open_chest"],
            command=name,
        )

    if name == "TakeChestRelic":
        if len(parts) != 1:
            _mapping_error("invalid_command", "TakeChestRelic takes no arguments")
        if (
            str(state.get("decision", "")) != "treasure"
            or state.get("treasure_stage") != "relic_ready"
        ):
            _mapping_error(
                "wrong_decision",
                "TakeChestRelic requires the native relic-ready treasure stage",
            )
        return _unique_action(
            [
                action
                for action in actions
                if action.command == "take_chest_relic"
                and action.argument_dict().get("relic_index") == 0
            ],
            command=name,
        )

    if name == "BuyCardRemoval":
        if len(parts) != 1:
            _mapping_error("invalid_command", "BuyCardRemoval takes no arguments")
        return _unique_action(
            [action for action in actions if action.command == "remove_card"],
            command=name,
        )

    if name == "BuyCard":
        prefix = "BuyCard "
        if not command.startswith(prefix) or not command[len(prefix) :]:
            _mapping_error("invalid_command", "BuyCard requires one exact card title")
        if str(state.get("decision", "")) != "shop":
            _mapping_error(
                "wrong_decision", "BuyCard requires an exported shop decision"
            )
        card_title = command[len(prefix) :]
        matching_indices = [
            int(card.get("index", fallback))
            for fallback, card in enumerate(_objects(state.get("cards")))
            if card.get("name") == card_title
        ]
        matches = [
            action
            for action in actions
            if action.command == "buy_card"
            and action.argument_dict().get("card_index") in matching_indices
        ]
        return _unique_action(matches, command=name)

    if name == "BuyRelic":
        prefix = "BuyRelic "
        if not command.startswith(prefix) or not command[len(prefix) :]:
            _mapping_error("invalid_command", "BuyRelic requires one exact relic title")
        relic_title = command[len(prefix) :]
        matching_indices = [
            int(relic.get("index", fallback))
            for fallback, relic in enumerate(_objects(state.get("relics")))
            if relic.get("name") == relic_title
        ]
        matches = [
            action
            for action in actions
            if action.command == "buy_relic"
            and action.argument_dict().get("relic_index") in matching_indices
        ]
        return _unique_action(matches, command=name)

    if name == "ChooseEventOption":
        if len(parts) != 2:
            _mapping_error("invalid_command", "ChooseEventOption requires one index")
        try:
            option_index = int(parts[1])
        except ValueError:
            _mapping_error(
                "invalid_command", "ChooseEventOption requires an integer index"
            )
        if option_index == -1:
            if str(state.get("decision", "")) == "event_reward":
                return _unique_action(
                    [action for action in actions if action.command == "skip_rewards"],
                    command=name,
                )
            if str(state.get("decision", "")) != "event_choice":
                return HeadlessAction.create(
                    "replay_noop", "RUNREPLAYS.EVENT_PROCEED_NOOP"
                )
            return _unique_action(
                [action for action in actions if action.command == "leave_room"],
                command=name,
            )
        if option_index < 0:
            _mapping_error(
                "invalid_command", "ChooseEventOption index must be -1 or nonnegative"
            )
        return _unique_action(
            [
                action
                for action in actions
                if action.command == "choose_option"
                and action.argument_dict().get("option_index") == option_index
            ],
            command=name,
        )

    if name == "ChooseRestSiteOption":
        prefix = "ChooseRestSiteOption "
        if not command.startswith(prefix):
            _mapping_error(
                "invalid_command", "ChooseRestSiteOption requires one exact option id"
            )
        option_id = command[len(prefix) :]
        if not option_id or any(character.isspace() for character in option_id):
            _mapping_error(
                "invalid_command", "ChooseRestSiteOption requires one exact option id"
            )
        if str(state.get("decision", "")) != "rest_site":
            _mapping_error(
                "wrong_decision",
                "ChooseRestSiteOption requires an exported rest_site decision",
            )
        serialized_option_id = f"OPTION.{option_id}"
        return _unique_action(
            [
                action
                for action in actions
                if action.command == "choose_option"
                and action.option_id == serialized_option_id
            ],
            command=name,
        )

    if name == "SelectCardFromScreen":
        if len(parts) != 2:
            _mapping_error(
                "invalid_command", "SelectCardFromScreen requires one exact index"
            )
        selected_index = _int_argument(
            parts[1], command=name, name="card index"
        )
        proof = state.get("potion_use_proof")
        cards = _objects(state.get("cards"))
        exact_pending_attack_potion = (
            str(state.get("decision", "")) == "card_select"
            and state.get("selection_source") == "potion"
            and isinstance(proof, Mapping)
            and proof.get("stage") == "selection_pending"
            and proof.get("selection_source") == "potion"
            and proof.get("source_model_id") == "POTION.ATTACK_POTION"
            and proof.get("option_count") == 3
            and proof.get("can_skip") is True
            and proof.get("potion_removed") is True
            and proof.get("postcondition_verified") is True
            and int(state.get("min_select", -1)) == 0
            and int(state.get("max_select", -1)) == 1
            and len(cards) == 3
            and len({str(card.get("id")) for card in cards}) == 3
            and all(card.get("type") == "Attack" for card in cards)
        )
        if not exact_pending_attack_potion:
            _mapping_error(
                "unproven_selection",
                "SelectCardFromScreen lacks the exact pending Attack Potion proof",
            )
        return _unique_action(
            [
                action
                for action in actions
                if action.command == "select_cards"
                and action.argument_dict().get("indices") == str(selected_index)
            ],
            command=name,
        )

    if name in {"SelectGridCard", "SelectHandCards"}:
        if len(parts) < 2:
            _mapping_error("invalid_command", f"{name} requires selected indices")
        indices = [
            _int_argument(value, command=name, name="card index")
            for value in parts[1:]
        ]
        joined = ",".join(str(value) for value in indices)
        return _unique_action(
            [
                action
                for action in actions
                if action.command == "select_cards"
                and action.argument_dict().get("indices") == joined
            ],
            command=name,
        )

    if name == "ClaimReward":
        if len(parts) != 2:
            _mapping_error("invalid_command", "ClaimReward requires one index")
        reward_index = _int_argument(parts[1], command=name, name="reward index")
        if str(state.get("decision", "")) == "event_reward":
            return _unique_action(
                [
                    action
                    for action in actions
                    if action.command == "claim_reward"
                    and action.argument_dict().get("reward_index") == reward_index
                ],
                command=name,
            )
        if str(state.get("decision", "")) != "card_reward":
            _mapping_error(
                "wrong_decision",
                "ClaimReward requires an exported card_reward decision",
            )
        queue = _reward_replay_queue(state)
        if queue is None:
            _mapping_error(
                "missing_reward_provenance",
                "ClaimReward requires the native replay reward queue",
            )
        if reward_index >= len(queue):
            _mapping_error(
                "illegal_reward_index",
                "ClaimReward index is absent from the remaining reward queue",
            )
        reward_token = queue[reward_index]
        disposition = str(reward_token.get("disposition", ""))
        if disposition not in {
            "auto_collected",
            "direct_collected",
            "retained_unselected",
            "card_reward_open",
        }:
            _mapping_error(
                "unhandled_reward",
                "ClaimReward points to a reward without proven headless equivalence",
            )
        if not _reward_disposition_is_proven(reward_token):
            _mapping_error(
                "unproven_reward",
                "ClaimReward token is missing a required native selection postcondition",
            )
        return HeadlessAction.create(
            "replay_noop",
            f"RUNREPLAYS.CLAIM_REWARD.{disposition.upper()}",
            reward_index=reward_index,
            disposition=disposition,
        )

    if name == "TakeCard":
        if len(parts) != 2:
            _mapping_error("invalid_command", "TakeCard requires one selection")
        queue = _reward_replay_queue(state)
        if queue is None:
            _mapping_error(
                "missing_reward_provenance",
                "TakeCard requires the native replay reward queue",
            )
        retained = [
            item
            for item in queue
            if str(item.get("disposition", "")) == "retained_unselected"
            and _reward_disposition_is_proven(item)
        ]
        if len(retained) != len(queue):
            _mapping_error(
                "pending_reward_claims",
                "TakeCard arrived before all reward claims were reconstructed",
            )
        if parts[1].casefold() == "skip":
            return _unique_action(
                [
                    action
                    for action in actions
                    if action.command == "skip_card_reward"
                ],
                command=name,
            )
        if parts[1].casefold() == "sacrifice":
            _mapping_error(
                "unsupported_selection",
                "TakeCard sacrifice has no proven headless mapping",
            )
        card_index = _int_argument(parts[1], command=name, name="card index")
        return _unique_action(
            [
                action
                for action in actions
                if action.command == "select_card_reward"
                and action.argument_dict().get("card_index") == card_index
            ],
            command=name,
        )

    if name == "PlayCard":
        if len(parts) not in {2, 3}:
            _mapping_error(
                "invalid_command", "PlayCard requires a card id and optional target id"
            )
        combat_card_id = _int_argument(
            parts[1], command=name, name="combat card id"
        )
        cards = [
            card
            for card in _objects(state.get("hand"))
            if card.get("combat_card_id") is not None
            and int(card["combat_card_id"]) == combat_card_id
        ]
        if not cards:
            _mapping_error(
                "missing_card_id",
                "PlayCard combat card id is absent from the current hand",
            )
        if len(cards) != 1:
            _mapping_error(
                "ambiguous_card_id", "PlayCard combat card id is not unique"
            )
        card_index = int(cards[0].get("index", 0))
        target_index = None
        if len(parts) == 3:
            target_index = _matching_creature_index(
                state,
                combat_id=_int_argument(
                    parts[2], command=name, name="target combat id"
                ),
                command=name,
            )
        matches = []
        for action in actions:
            arguments = action.argument_dict()
            if action.command != "play_card":
                continue
            if arguments.get("card_index") != card_index:
                continue
            if target_index is None and "target_index" in arguments:
                continue
            if target_index is not None and arguments.get("target_index") != target_index:
                continue
            matches.append(action)
        return _unique_action(matches, command=name)

    if name == "UsePotion":
        if len(parts) not in {2, 3}:
            _mapping_error(
                "invalid_command", "UsePotion requires a slot and optional target id"
            )
        potion_index = _int_argument(parts[1], command=name, name="potion slot")
        target_index = None
        if len(parts) == 3:
            target_index = _matching_creature_index(
                state,
                combat_id=_int_argument(
                    parts[2], command=name, name="target combat id"
                ),
                command=name,
            )
        matches = []
        for action in actions:
            arguments = action.argument_dict()
            if action.command != "use_potion":
                continue
            if arguments.get("potion_index") != potion_index:
                continue
            if target_index is None and "target_index" in arguments:
                continue
            if target_index is not None and arguments.get("target_index") != target_index:
                continue
            matches.append(action)
        return _unique_action(matches, command=name)

    if name == "DiscardPotion":
        if len(parts) != 2:
            _mapping_error("invalid_command", "DiscardPotion requires one slot")
        potion_index = _int_argument(parts[1], command=name, name="potion slot")
        return _unique_action(
            [
                action
                for action in actions
                if action.command == "discard_potion"
                and action.argument_dict().get("potion_index") == potion_index
            ],
            command=name,
        )

    if name == "NetPickRelicAction":
        if command != "NetPickRelicAction for player 1 index 0":
            _mapping_error(
                "quarantined_command",
                "NetPickRelicAction does not match the exact audited singleplayer echo",
            )
        if not _treasure_relic_award_is_proven(state):
            _mapping_error(
                "quarantined_command",
                "NetPickRelicAction lacks a verified preceding chest-relic award",
            )
        return HeadlessAction.create(
            "replay_noop", "RUNREPLAYS.VERIFIED_NET_PICK_RELIC_ECHO"
        )

    if name in AUDITED_UNSUPPORTED_COMMANDS:
        _mapping_error(
            "quarantined_command",
            f"{name} is a quarantined legacy fallback command",
        )
    if name in KNOWN_COMMANDS:
        _mapping_error(
            "unsupported_command",
            f"{name} has no proven headless reconstruction mapping",
        )
    _mapping_error("unknown_command", f"Unknown replay command type {name}")


def resolve_runreplays_followup_action(
    adapter: HeadlessAction,
    state: Mapping[str, Any],
    *,
    legal_actions: Sequence[HeadlessAction] | None = None,
) -> HeadlessAction:
    """Resolve the second native stage of a typed replay-only adapter."""

    if adapter.command != "replay_two_stage":
        _mapping_error("invalid_adapter", "Action is not a two-stage replay adapter")
    arguments = adapter.argument_dict()
    if arguments.get("first_action") != "leave_room" or not isinstance(
        arguments.get("col"), int
    ):
        _mapping_error("invalid_adapter", "Room-to-map adapter arguments are invalid")
    if str(state.get("decision", "")) != "map_select":
        _mapping_error(
            "wrong_followup_decision",
            "Room-to-map adapter requires map_select after leaving the room",
        )
    actions = tuple(legal_actions or legal_headless_actions(state))
    col = int(arguments["col"])
    return _unique_action(
        [
            action
            for action in actions
            if action.command == "select_map_node"
            and action.argument_dict().get("col") == col
        ],
        command="MoveToMapCoord followup",
    )


def _read_snapshot(path: Path) -> dict[str, Any]:
    headers: dict[str, str] = {}
    commands: list[str] = []
    diagnostic_prestates: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            if text.startswith("#"):
                header = text[1:].strip()
                if ":" in header:
                    key, value = header.split(":", 1)
                    headers[key.strip().casefold()] = value.strip()
                continue
            command_text, separator, _ = text.partition(" || ")
            command = command_text.split(" # ", 1)[0].strip()
            if not command:
                continue
            if any(character in command for character in "\r\n\x00"):
                raise ValueError(f"Invalid replay command in {path} at line {line_number}")
            commands.append(command)
            if separator:
                diagnostic_prestates[_command_name(command)] += 1

    missing_headers = sorted(REQUIRED_HEADERS - headers.keys())
    character = canonical_character(headers.get("character"))
    identity = "\x1f".join(
        (headers.get("game", ""), character, headers.get("seed", ""))
    )
    command_types = Counter(_command_name(command) for command in commands)
    canonical_commands = [
        command for command in commands if _command_name(command) in KNOWN_COMMANDS
    ]
    combat_commands = [
        command for command in commands if _command_name(command) in COMBAT_COMMANDS
    ]
    snapshot_floor = _floor(path)
    return {
        "path": path,
        "file_sha256": _sha256_file(path),
        "trajectory_key": _sha256_bytes(identity.encode("utf-8")),
        "character": character,
        "build_id": headers.get("game", "unknown"),
        "mod_version": headers.get("mod", "unknown"),
        "ascension": headers.get("ascension", "unknown"),
        "floor": snapshot_floor,
        "missing_headers": missing_headers,
        "commands": commands,
        "canonical_commands": canonical_commands,
        "command_types": command_types,
        "combat_commands": combat_commands,
        "diagnostic_prestates": diagnostic_prestates,
        "unsupported_commands": sum(
            count
            for name, count in command_types.items()
            if name in AUDITED_UNSUPPORTED_COMMANDS
        ),
        "unsupported_command_types": Counter(
            {
                name: count
                for name, count in command_types.items()
                if name in AUDITED_UNSUPPORTED_COMMANDS
            }
        ),
        "unknown_commands": sum(
            count
            for name, count in command_types.items()
            if name not in KNOWN_COMMANDS
            and name not in AUDITED_UNSUPPORTED_COMMANDS
        ),
        "invalid_combat_commands": sum(
            not _valid_combat_command(command) for command in combat_commands
        ),
        "run_save": _run_save_sidecar(
            path,
            headers=headers,
            character=character,
            snapshot_floor=snapshot_floor,
        ),
    }


def _deduplicated_instances(
    snapshots: Sequence[Mapping[str, Any]],
) -> tuple[set[tuple[str, int, str]], int, int, int, int]:
    instances: set[tuple[str, int, str]] = set()
    prefix_conforming = 0
    conflicts = 0
    ambiguous_groups = 0
    effective_streams = 0
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for snapshot in snapshots:
        grouped[str(snapshot["trajectory_key"])].append(snapshot)

    for trajectory_key, group in grouped.items():
        longest = max(group, key=lambda snapshot: len(snapshot["canonical_commands"]))
        canonical = list(longest["canonical_commands"])
        conforming = [
            list(snapshot["canonical_commands"])
            == canonical[: len(snapshot["canonical_commands"])]
            for snapshot in group
        ]
        prefix_conforming += sum(conforming)
        ambiguous = not all(conforming)
        ambiguous_groups += int(ambiguous)
        effective_streams += len(group) if ambiguous else 1
        for snapshot in group:
            commands = list(snapshot["canonical_commands"])
            stream_key = (
                f"{trajectory_key}:{snapshot['file_sha256']}"
                if ambiguous
                else trajectory_key
            )
            for index, command in enumerate(commands):
                instances.add((stream_key, index, command))
                if index < len(canonical) and canonical[index] != command:
                    conflicts += 1
    return (
        instances,
        prefix_conforming,
        conflicts,
        ambiguous_groups,
        effective_streams,
    )


def _coverage(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    expected_build: str,
) -> dict[str, Any]:
    raw_commands = sum(len(snapshot["commands"]) for snapshot in snapshots)
    supported_commands = sum(
        len(snapshot["canonical_commands"]) for snapshot in snapshots
    )
    (
        instances,
        prefix_conforming,
        conflicts,
        ambiguous_groups,
        effective_streams,
    ) = _deduplicated_instances(snapshots)
    unique_commands = len(instances)
    duplicate_commands = supported_commands - unique_commands
    unique_types = Counter(
        name if name in KNOWN_COMMANDS else "<unknown>"
        for name in (_command_name(instance[2]) for instance in instances)
    )
    combat_types = Counter(
        {name: count for name, count in unique_types.items() if name in COMBAT_COMMANDS}
    )
    combat_total = sum(combat_types.values())
    unique_invalid_combat = sum(
        _command_name(instance[2]) in COMBAT_COMMANDS
        and not _valid_combat_command(instance[2])
        for instance in instances
    )
    raw_combat_total = sum(len(snapshot["combat_commands"]) for snapshot in snapshots)
    raw_prestates = Counter()
    unsupported_types = Counter()
    for snapshot in snapshots:
        raw_prestates.update(snapshot["diagnostic_prestates"])
        unsupported_types.update(snapshot["unsupported_command_types"])
    raw_combat_prestates = sum(
        count for name, count in raw_prestates.items() if name in COMBAT_COMMANDS
    )
    save_sidecars = [snapshot["run_save"] for snapshot in snapshots]
    checked_sidecars = [
        sidecar for sidecar in save_sidecars if sidecar["status"] == "checked"
    ]
    return {
        "files": len(snapshots),
        "trajectory_identity_groups": len(
            {snapshot["trajectory_key"] for snapshot in snapshots}
        ),
        "effective_replay_streams": effective_streams,
        "files_matching_expected_build": sum(
            snapshot["build_id"] == expected_build for snapshot in snapshots
        ),
        "files_with_missing_headers": sum(
            bool(snapshot["missing_headers"]) for snapshot in snapshots
        ),
        "raw_commands": raw_commands,
        "supported_commands": supported_commands,
        "unique_commands_after_snapshot_dedup": unique_commands,
        "duplicate_snapshot_commands": duplicate_commands,
        "snapshot_dedup_rate": round(
            duplicate_commands / supported_commands if supported_commands else 0.0,
            6,
        ),
        "prefix_conforming_snapshots": prefix_conforming,
        "snapshot_prefix_conflicts": conflicts,
        "ambiguous_trajectory_groups": ambiguous_groups,
        "unknown_commands": sum(snapshot["unknown_commands"] for snapshot in snapshots),
        "unsupported_commands": sum(
            snapshot["unsupported_commands"] for snapshot in snapshots
        ),
        "unsupported_command_types": dict(sorted(unsupported_types.items())),
        "invalid_combat_commands": sum(
            snapshot["invalid_combat_commands"] for snapshot in snapshots
        ),
        "unique_command_types": dict(sorted(unique_types.items())),
        "combat_action_identity": {
            "eligible": combat_total,
            "syntactically_parsed": combat_total - unique_invalid_combat,
            "by_type": dict(sorted(combat_types.items())),
        },
        "raw_diagnostic_prestate_coverage": {
            "combat_commands": raw_combat_total,
            "with_diagnostic_prestate": raw_combat_prestates,
            "rate": round(
                raw_combat_prestates / raw_combat_total
                if raw_combat_total
                else 0.0,
                6,
            ),
            "by_type": {
                name: {
                    "commands": sum(
                        snapshot["command_types"].get(name, 0)
                        for snapshot in snapshots
                    ),
                    "with_diagnostic_prestate": raw_prestates.get(name, 0),
                }
                for name in sorted(COMBAT_COMMANDS)
            },
        },
        "run_save_sidecars": {
            "present": sum(sidecar["status"] != "missing" for sidecar in save_sidecars),
            "valid": len(checked_sidecars),
            "all_crosschecks_passed": sum(
                sidecar["all_crosschecks_passed"] for sidecar in checked_sidecars
            ),
            "solo": sum(sidecar.get("player_count") == 1 for sidecar in checked_sidecars),
            "standard_game_mode": sum(
                str(sidecar.get("game_mode", "")).casefold() == "standard"
                for sidecar in checked_sidecars
            ),
            "schema_versions": dict(
                sorted(
                    Counter(
                        str(sidecar.get("schema_version", "unknown"))
                        for sidecar in checked_sidecars
                    ).items()
                )
            ),
        },
        "legal_action_reconstruction": {
            "eligible": combat_total,
            "success": 0,
            "rate": 0.0,
        },
        "trainable_combat_decisions": 0,
    }


def audit_runreplays(
    inputs: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
    *,
    expected_build: str,
    source_uri: str | None = None,
    source_revision: str | None = None,
    license_spdx: str | None = None,
    license_file: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Audit replay snapshots without producing or authorizing training data."""

    if not expected_build.strip():
        raise ValueError("expected_build is required")
    paths = _discover(inputs)
    if not paths:
        raise ValueError(f"No {REPLAY_FILENAME} files were found")
    snapshots = [_read_snapshot(path) for path in paths]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for snapshot in snapshots:
        grouped[str(snapshot["character"])].append(snapshot)

    license_path = Path(license_file) if license_file is not None else None
    license_verified = bool(
        license_spdx
        and license_path is not None
        and license_path.is_file()
    )
    overall = _coverage(snapshots, expected_build=expected_build)
    report = {
        "schema_version": 1,
        "audit": "runreplays_provenance_and_trainability",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run_only": True,
        "training_authorized": False,
        "expected_build": expected_build,
        "provenance": {
            "source_name": "RunReplays",
            "source_uri": source_uri,
            "source_revision": source_revision,
            "input_roots": len(inputs),
            "license": {
                "spdx": license_spdx,
                "file_sha256": (
                    _sha256_file(license_path) if license_verified and license_path else None
                ),
                "repository_license_verified": license_verified,
                "scope_caveat": (
                    "Repository licensing does not establish rights to unrelated "
                    "third-party replay collections."
                ),
            },
            "files": [
                {
                    "file_sha256": snapshot["file_sha256"],
                    "trajectory_key": snapshot["trajectory_key"],
                    "character": snapshot["character"],
                    "build_id": snapshot["build_id"],
                    "mod_version": snapshot["mod_version"],
                    "ascension": snapshot["ascension"],
                    "snapshot_floor": snapshot["floor"],
                    "commands": len(snapshot["commands"]),
                    "missing_headers": snapshot["missing_headers"],
                    "run_save": snapshot["run_save"],
                }
                for snapshot in snapshots
            ],
        },
        "overall": overall,
        "by_character": {
            character: _coverage(group, expected_build=expected_build)
            for character, group in sorted(grouped.items())
        },
        "exact_build_gate": {
            "passed": (
                overall["files_matching_expected_build"] == overall["files"]
                and overall["files_with_missing_headers"] == 0
                and overall["run_save_sidecars"]["all_crosschecks_passed"]
                == overall["files"]
            ),
            "expected_build": expected_build,
            "matching_files": overall["files_matching_expected_build"],
            "total_files": overall["files"],
        },
        "trainability_gate": {
            "passed": False,
            "reason": (
                "RunReplays command snapshots identify chosen commands but do not "
                "contain exact pre-action observations or complete legal-action masks."
            ),
            "missing_per_decision_fields": [
                "pre_action_observation",
                "complete_legal_actions",
                "combat_card_id_to_hand_index_mapping",
                "model_table_hash_for_modded_replay",
            ],
            "next_required_step": (
                "Replay commands on the exact game build while instrumenting every "
                "pre-action state and legal candidate set, then audit the reconstructed "
                "records before training."
            ),
        },
        "privacy": {
            "raw_seeds_persisted": False,
            "raw_commands_persisted": False,
            "inline_comments_persisted": False,
            "trajectory_identity": "sha256(game_build, character, seed)",
        },
    }
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Audit report already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return report


def _runtime_identity(config: HeadlessConfig) -> dict[str, Any]:
    runtime = config.runtime.resolve()
    identity: dict[str, Any] = {
        "headless_runtime_sha256": _sha256_file(runtime),
    }
    for parent in runtime.parents:
        game_assembly = parent / "lib" / "sts2.dll"
        if game_assembly.is_file():
            identity["game_assembly_sha256"] = _sha256_file(game_assembly)
            break
    return identity


def _decision_summary(
    state: Mapping[str, Any], legal_actions: Sequence[HeadlessAction]
) -> dict[str, Any]:
    summary = {
        "decision": state.get("decision"),
        "context": state.get("context"),
        "legal_actions": [action.as_dict() for action in legal_actions],
    }
    reward_queue = _reward_replay_queue(state)
    if reward_queue is not None:
        summary["replay_reward_queue"] = [dict(item) for item in reward_queue]
    return summary


def _write_new_json(destination: Path, value: Mapping[str, Any]) -> None:
    if destination.exists():
        raise FileExistsError(f"Report already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def probe_runreplays_reconstruction(
    base_snapshot: str | os.PathLike[str],
    target_snapshot: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    expected_build: str,
    max_commands: int = 100,
    max_overlap: int = 0,
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Replay one cumulative snapshot suffix without authorizing training.

    The paired save next to ``base_snapshot`` is loaded in the native runtime.
    Prefix alignment is exact on raw commands, or exact on recognized commands
    when all differences are audited legacy fallbacks and both streams contain
    no unknown command types. Every suffix command must then map explicitly; the
    probe stops at the first unsupported or divergent command.
    """

    if not expected_build.strip():
        raise ValueError("expected_build is required")
    if max_commands <= 0:
        raise ValueError("max_commands must be positive")
    if max_overlap < 0 or max_overlap > 20:
        raise ValueError("max_overlap must be between 0 and 20")
    base_path = Path(base_snapshot)
    target_path = Path(target_snapshot)
    if base_path.name.casefold() != REPLAY_FILENAME:
        raise ValueError(f"Base snapshot must be named {REPLAY_FILENAME}")
    if target_path.name.casefold() != REPLAY_FILENAME:
        raise ValueError(f"Target snapshot must be named {REPLAY_FILENAME}")
    base = _read_snapshot(base_path)
    target = _read_snapshot(target_path)

    alignment = _prefix_alignment(
        base["commands"], target["commands"], max_overlap=max_overlap
    )
    validation = {
        "same_trajectory": base["trajectory_key"] == target["trajectory_key"],
        "same_character": base["character"] == target["character"],
        "same_ascension": base["ascension"] == target["ascension"],
        "exact_build": (
            base["build_id"] == expected_build
            and target["build_id"] == expected_build
        ),
        "complete_headers": not base["missing_headers"]
        and not target["missing_headers"],
        "base_save_crosschecks": base["run_save"]["all_crosschecks_passed"],
        "target_save_crosschecks": target["run_save"]["all_crosschecks_passed"],
        "prefix_gate": alignment["prefix_gate"],
        "overlap_compatible": alignment["overlap_compatible"],
        "target_extends_base": len(target["canonical_commands"])
        > len(base["canonical_commands"]),
    }
    failed = [name for name, passed in validation.items() if not passed]
    if failed:
        raise ValueError(
            "Replay reconstruction validation failed: " + ", ".join(failed)
        )

    target_boundary = alignment["target_boundary"]
    if target_boundary is None:
        raise ValueError("Replay reconstruction could not derive the target boundary")
    save_path = base_path.with_name("run.save")
    suffix = list(target["commands"][target_boundary:])
    runtime_config = config or HeadlessConfig()
    character = str(base["character"])
    try:
        ascension = int(str(base["ascension"]))
    except ValueError as exc:
        raise ValueError("Replay ascension is not an integer") from exc

    records: list[dict[str, Any]] = []
    command_types: Counter[str] = Counter()
    reconstructed_types: Counter[str] = Counter()
    stop: dict[str, Any] = {"status": "not_started"}
    attempted = 0
    reconstructed = 0
    reconstructed_noops = 0
    reconstructed_two_stage_adapters = 0
    replay_commands = suffix
    replay_start_index = target_boundary
    initial_alignment: dict[str, Any] | None = None

    with Sts2HeadlessClient(runtime_config) as client:
        state = client.load_save(save_path)
        if state.get("type") == "error":
            stop = {
                "status": "runtime_error",
                "stage": "load_save",
                "message": str(state.get("message", "Unknown load_save error")),
            }
        else:
            initial_legal_actions = legal_headless_actions(state)
            initial_alignment = {
                "max_overlap": max_overlap,
                "loaded_state": {
                    **_decision_summary(state, initial_legal_actions),
                },
                "tail_candidates": [],
                "matched_command_index": None,
                "overlap_commands": 0,
            }
            if max_overlap:
                candidate_start = max(0, len(base["commands"]) - max_overlap)
                matches: list[tuple[int, HeadlessAction]] = []
                for index in range(candidate_start, len(base["commands"])):
                    candidate = base["commands"][index]
                    candidate_evidence = {
                        "command_index": index,
                        "command_type": _command_name(candidate),
                        "command_sha256": _sha256_bytes(candidate.encode("utf-8")),
                    }
                    try:
                        action = map_runreplays_command(
                            state,
                            candidate,
                            legal_actions=initial_legal_actions,
                        )
                    except ReplayMappingError as exc:
                        candidate_evidence["match"] = False
                        candidate_evidence["code"] = exc.code
                    else:
                        candidate_evidence["match"] = True
                        candidate_evidence["action"] = action.as_dict()
                        matches.append((index, action))
                    initial_alignment["tail_candidates"].append(candidate_evidence)
                if len(matches) != 1:
                    stop = {
                        "status": "alignment_failure",
                        "code": (
                            "no_tail_match" if not matches else "ambiguous_tail_match"
                        ),
                        "matching_tail_commands": len(matches),
                    }
                else:
                    replay_start_index = matches[0][0]
                    replay_commands = (
                        list(base["commands"][replay_start_index:]) + suffix
                    )
                    initial_alignment["matched_command_index"] = replay_start_index
                    initial_alignment["overlap_commands"] = (
                        len(base["commands"]) - replay_start_index
                    )

            if stop["status"] == "not_started":
                for offset, command in enumerate(replay_commands[:max_commands]):
                    attempted += 1
                    command_name = _command_name(command)
                    command_types[command_name] += 1
                    command_hash = _sha256_bytes(command.encode("utf-8"))
                    absolute_index = replay_start_index + offset
                    try:
                        legal_actions = legal_headless_actions(state)
                        chosen = map_runreplays_command(
                            state, command, legal_actions=legal_actions
                        )
                    except ReplayMappingError as exc:
                        stop = {
                            "status": "mapping_failure",
                            "command_index": absolute_index,
                            "command_type": command_name,
                            "command_sha256": command_hash,
                            "code": exc.code,
                            "message": str(exc),
                            "state": _decision_summary(state, legal_actions),
                        }
                        break
                    except ValueError as exc:
                        stop = {
                            "status": "state_mapping_failure",
                            "command_index": absolute_index,
                            "command_type": command_name,
                            "command_sha256": command_hash,
                            "message": str(exc),
                            "state": _decision_summary(state, legal_actions),
                        }
                        break

                    pending_record: dict[str, Any] | None = None
                    if command_name in COMBAT_COMMANDS:
                        pending_record = {
                            "replay_command_index": absolute_index,
                            "replay_command_type": command_name,
                            "replay_command_sha256": command_hash,
                            "pre_action_observation": normalize_headless_observation(
                                state,
                                character=character,
                                ascension=ascension,
                                seed=None,
                            ),
                            "legal_actions": [
                                action.as_dict() for action in legal_actions
                            ],
                            "legal_action_mask": [True] * len(legal_actions),
                            "chosen_legal_index": legal_actions.index(chosen),
                            "chosen_action": chosen.as_dict(),
                        }

                    if chosen.command == "replay_noop":
                        next_state = dict(state)
                        if command_name == "ClaimReward":
                            reward_queue = _reward_replay_queue(state)
                            if reward_queue is None:
                                raise RuntimeError(
                                    "Mapped ClaimReward is missing its reward queue"
                                )
                            remaining = [dict(item) for item in reward_queue]
                            reward_index = int(
                                chosen.argument_dict()["reward_index"]
                            )
                            if chosen.argument_dict()["disposition"] != "retained_unselected":
                                remaining.pop(reward_index)
                            next_state["_runreplays_reward_queue"] = remaining
                        reconstructed_noops += 1
                    elif chosen.command == "replay_two_stage":
                        adapter_arguments = chosen.argument_dict()
                        intermediate_state = client.action(
                            str(adapter_arguments["first_action"])
                        )
                        if intermediate_state.get("type") == "error":
                            stop = {
                                "status": "runtime_error",
                                "stage": "adapter_leave_room",
                                "command_index": absolute_index,
                                "command_type": command_name,
                                "command_sha256": command_hash,
                                "message": str(
                                    intermediate_state.get(
                                        "message", "Unknown leave_room error"
                                    )
                                ),
                                "state": _decision_summary(state, legal_actions),
                            }
                            break
                        followup_legal_actions: Sequence[HeadlessAction] = ()
                        try:
                            followup_legal_actions = legal_headless_actions(
                                intermediate_state
                            )
                            followup = resolve_runreplays_followup_action(
                                chosen,
                                intermediate_state,
                                legal_actions=followup_legal_actions,
                            )
                        except ReplayMappingError as exc:
                            stop = {
                                "status": "mapping_failure",
                                "stage": "adapter_select_map_node",
                                "command_index": absolute_index,
                                "command_type": command_name,
                                "command_sha256": command_hash,
                                "code": exc.code,
                                "message": str(exc),
                                "state": _decision_summary(
                                    intermediate_state,
                                    followup_legal_actions,
                                ),
                            }
                            break
                        except ValueError as exc:
                            stop = {
                                "status": "state_mapping_failure",
                                "stage": "adapter_select_map_node",
                                "command_index": absolute_index,
                                "command_type": command_name,
                                "command_sha256": command_hash,
                                "message": str(exc),
                            }
                            break
                        next_state = client.action(
                            followup.command, **followup.argument_dict()
                        )
                        if next_state.get("type") == "error":
                            stop = {
                                "status": "runtime_error",
                                "stage": "adapter_select_map_node",
                                "command_index": absolute_index,
                                "command_type": command_name,
                                "command_sha256": command_hash,
                                "message": str(
                                    next_state.get(
                                        "message", "Unknown select_map_node error"
                                    )
                                ),
                                "state": _decision_summary(
                                    intermediate_state, followup_legal_actions
                                ),
                            }
                            break
                        reconstructed_two_stage_adapters += 1
                    else:
                        next_state = client.action(
                            chosen.command, **chosen.argument_dict()
                        )
                    if next_state.get("type") == "error":
                        stop = {
                            "status": "runtime_error",
                            "stage": "action",
                            "command_index": absolute_index,
                            "command_type": command_name,
                            "command_sha256": command_hash,
                            "message": str(
                                next_state.get("message", "Unknown action error")
                            ),
                            "state": _decision_summary(state, legal_actions),
                        }
                        break
                    if command_name == "ClaimReward" and chosen.command == "claim_reward":
                        reward_index = int(chosen.argument_dict()["reward_index"])
                        if not _event_reward_claim_is_proven(
                            next_state, reward_index=reward_index
                        ):
                            stop = {
                                "status": "runtime_proof_failure",
                                "stage": "event_reward_claim",
                                "command_index": absolute_index,
                                "command_type": command_name,
                                "command_sha256": command_hash,
                                "message": (
                                    "Native event reward claim did not return a verified "
                                    "selection postcondition"
                                ),
                                "state": _decision_summary(next_state),
                            }
                            break
                    if command_name == "UsePotion" and chosen.command == "use_potion":
                        potion_index = int(chosen.argument_dict()["potion_index"])
                        if not _potion_use_is_proven(
                            next_state, potion_index=potion_index
                        ):
                            stop = {
                                "status": "runtime_proof_failure",
                                "stage": "potion_use",
                                "command_index": absolute_index,
                                "command_type": command_name,
                                "command_sha256": command_hash,
                                "message": (
                                    "Native potion use did not return a verified removal "
                                    "or pending-selection postcondition"
                                ),
                                "state": _decision_summary(next_state),
                            }
                            break
                    if (
                        command_name == "SelectCardFromScreen"
                        and chosen.command == "select_cards"
                    ):
                        selected_index = int(
                            chosen.argument_dict()["indices"]
                        )
                        if not _potion_screen_selection_is_proven(
                            next_state, selected_index=selected_index
                        ):
                            stop = {
                                "status": "runtime_proof_failure",
                                "stage": "potion_card_selection",
                                "command_index": absolute_index,
                                "command_type": command_name,
                                "command_sha256": command_hash,
                                "message": (
                                    "Attack Potion selection did not return an exact "
                                    "selected-reference-in-hand postcondition"
                                ),
                                "state": _decision_summary(next_state),
                            }
                            break
                    reconstructed += 1
                    reconstructed_types[command_name] += 1
                    if pending_record is not None:
                        records.append(pending_record)
                    state = next_state
                else:
                    if len(replay_commands) <= max_commands:
                        stop = {"status": "target_exhausted"}
                    else:
                        stop = {
                            "status": "max_commands_reached",
                            "next_command_index": replay_start_index + max_commands,
                        }

        runtime_protocol = client.version

    report = {
        "schema_version": 1,
        "probe": "runreplays_headless_reconstruction",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run_only": True,
        "training_authorized": False,
        "expected_build": expected_build,
        "validation": validation,
        "alignment": alignment,
        "provenance": {
            "source_name": "RunReplays",
            "trajectory_key": base["trajectory_key"],
            "character": character,
            "ascension": ascension,
            "base_snapshot_floor": base["floor"],
            "target_snapshot_floor": target["floor"],
            "base_snapshot_sha256": base["file_sha256"],
            "target_snapshot_sha256": target["file_sha256"],
            "base_run_save_sha256": base["run_save"].get("file_sha256"),
        },
        "runtime": {
            "protocol_version": runtime_protocol,
            **_runtime_identity(runtime_config),
        },
        "coverage": {
            "suffix_commands": len(suffix),
            "commands_scheduled_with_overlap": len(replay_commands),
            "commands_attempted": attempted,
            "commands_reconstructed": reconstructed,
            "reconstructed_noops": reconstructed_noops,
            "reconstructed_two_stage_adapters": reconstructed_two_stage_adapters,
            "reconstruction_rate_of_attempted": round(
                reconstructed / attempted if attempted else 0.0, 6
            ),
            "attempted_by_type": dict(sorted(command_types.items())),
            "reconstructed_by_type": dict(sorted(reconstructed_types.items())),
            "combat_decisions_reconstructed": len(records),
            "combat_decisions_with_complete_legal_mask": len(records),
        },
        "initial_alignment": initial_alignment,
        "stop": stop,
        "combat_decisions": records,
        "privacy": {
            "raw_seed_persisted": False,
            "raw_replay_commands_persisted": False,
            "inline_comments_persisted": False,
        },
    }
    _write_new_json(Path(output), report)
    return report
