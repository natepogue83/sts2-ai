"""Gym-like training environment for the pinned native STS2 runtime.

The module intentionally has no Gymnasium dependency.  It exposes the same
reset/step concepts while preserving dynamic, candidate-based action spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence
import uuid

from stsai.headless import (
    HeadlessConfig,
    HeadlessRuntimeError,
    HeadlessSnapshot,
    HeadlessSnapshotError,
    Sts2HeadlessClient,
)
from stsai.ids import canonical_character, canonical_entity


HEADLESS_CHARACTERS = ("Ironclad", "Silent", "Defect", "Necrobinder", "Regent")


class UnsupportedHeadlessDecision(ValueError):
    """Raised rather than guessing an action for an unknown runtime decision."""


@dataclass(frozen=True)
class HeadlessAction:
    """One explicit legal candidate for the runtime's dynamic action space."""

    command: str
    option_id: str
    arguments: tuple[tuple[str, int | str], ...] = ()

    @classmethod
    def create(
        cls, command: str, option_id: str, **arguments: int | str
    ) -> HeadlessAction:
        return cls(command, option_id, tuple(sorted(arguments.items())))

    def argument_dict(self) -> dict[str, int | str]:
        return dict(self.arguments)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "action": self.command,
            "option_id": self.option_id,
        }
        if self.arguments:
            result["args"] = self.argument_dict()
        return result


@dataclass(frozen=True)
class RewardConfig:
    """Sparse objective-aligned rewards; shaping can be added explicitly later."""

    victory: float = 1.0
    defeat: float = -1.0
    nonterminal: float = 0.0


@dataclass(frozen=True)
class StepResult:
    observation: Mapping[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]


@dataclass(frozen=True)
class Sts2EnvSnapshot:
    """Character-aware wrapper around one immutable native snapshot."""

    schema_version: int
    native: HeadlessSnapshot
    character: str
    ascension: int
    seed: str
    last_combat_context_json: str | None
    observation_sha256: str
    legal_actions: tuple[HeadlessAction, ...]


def _objects(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _index(item: Mapping[str, Any], fallback: int) -> int:
    value = item.get("index", fallback)
    return int(value)


def _entity_option(item: Mapping[str, Any], prefix: str, fallback: str) -> str:
    value = item.get("id") or item.get("name") or fallback
    return canonical_entity(value, prefix)


def _targeted_action(
    *,
    command: str,
    option_id: str,
    index_name: str,
    index: int,
    target_type: Any,
    enemies: Sequence[Mapping[str, Any]],
) -> list[HeadlessAction]:
    if str(target_type or "").lower() == "anyenemy":
        return [
            HeadlessAction.create(
                command,
                f"{option_id}|TARGET.{_index(enemy, target_index)}",
                **{index_name: index, "target_index": _index(enemy, target_index)},
            )
            for target_index, enemy in enumerate(enemies)
        ]
    return [HeadlessAction.create(command, option_id, **{index_name: index})]


def legal_headless_actions(
    state: Mapping[str, Any], *, max_card_selection_actions: int = 20_000
) -> tuple[HeadlessAction, ...]:
    """Enumerate supported actions that the current decision advertises as legal."""

    decision = str(state.get("decision", ""))
    actions: list[HeadlessAction] = []

    if decision == "game_over":
        return ()

    if decision == "map_select":
        for fallback, choice in enumerate(_objects(state.get("choices"))):
            col = int(choice["col"])
            row = int(choice["row"])
            room = str(choice.get("type", "UNKNOWN")).upper()
            actions.append(
                HeadlessAction.create(
                    "select_map_node", f"MAP.{room}.{col}.{row}", col=col, row=row
                )
            )
        return tuple(actions)

    if decision == "combat_play":
        enemies = _objects(state.get("enemies"))
        for fallback, card in enumerate(_objects(state.get("hand"))):
            if card.get("can_play") is not True:
                continue
            index = _index(card, fallback)
            option_id = _entity_option(card, "CARD", f"HAND_{index}")
            actions.extend(
                _targeted_action(
                    command="play_card",
                    option_id=option_id,
                    index_name="card_index",
                    index=index,
                    target_type=card.get("target_type"),
                    enemies=enemies,
                )
            )
        player = state.get("player")
        player = player if isinstance(player, Mapping) else {}
        for fallback, potion in enumerate(_objects(player.get("potions"))):
            if potion.get("can_use_now") is False:
                continue
            index = _index(potion, fallback)
            option_id = _entity_option(potion, "POTION", f"SLOT_{index}")
            actions.extend(
                _targeted_action(
                    command="use_potion",
                    option_id=option_id,
                    index_name="potion_index",
                    index=index,
                    target_type=potion.get("target_type"),
                    enemies=enemies,
                )
            )
        actions.append(HeadlessAction.create("end_turn", "END_TURN"))
        return tuple(actions)

    if decision in {"event_choice", "rest_site"}:
        for fallback, option in enumerate(_objects(state.get("options"))):
            enabled = option.get("is_enabled", True) is not False
            unlocked = option.get("is_locked", False) is not True
            if not enabled or not unlocked:
                continue
            index = _index(option, fallback)
            stable_id = (
                option.get("option_id")
                or option.get("text_key")
                or option.get("title")
                or f"OPTION_{index}"
            )
            actions.append(
                HeadlessAction.create(
                    "choose_option", f"OPTION.{stable_id}", option_index=index
                )
            )
        has_room_options = bool(actions)
        if decision == "event_choice":
            player = state.get("player")
            player = player if isinstance(player, Mapping) else {}
            for fallback, potion in enumerate(_objects(player.get("potions"))):
                index = _index(potion, fallback)
                option_id = _entity_option(potion, "POTION", f"SLOT_{index}")
                actions.append(
                    HeadlessAction.create(
                        "discard_potion",
                        f"DISCARD.{option_id}",
                        potion_index=index,
                    )
                )
        if not has_room_options:
            actions.append(HeadlessAction.create("leave_room", "LEAVE_ROOM"))
        return tuple(actions)

    if decision == "event_reward":
        for fallback, reward in enumerate(_objects(state.get("rewards"))):
            index = _index(reward, fallback)
            reward_type = str(reward.get("reward_type", "REWARD")).upper()
            entity_id = str(reward.get("entity_id") or reward_type)
            actions.append(
                HeadlessAction.create(
                    "claim_reward",
                    f"EVENT_REWARD.{index}.{entity_id}",
                    reward_index=index,
                )
            )
        if state.get("can_skip") is True:
            actions.append(HeadlessAction.create("skip_rewards", "SKIP_REWARDS"))
        return tuple(actions)

    if decision == "fake_shop":
        if (
            state.get("event_stage") == "fake_shop_ready"
            and state.get("inventory_present") is True
            and state.get("started_fight") is False
        ):
            return (HeadlessAction.create("open_fake_shop", "OPEN_FAKE_SHOP"),)
        return ()

    if decision == "treasure":
        stage = str(state.get("treasure_stage", ""))
        if stage == "unopened":
            return (HeadlessAction.create("open_chest", "OPEN_CHEST"),)
        if stage == "relic_ready":
            relics = _objects(state.get("relics"))
            if len(relics) != 1 or _index(relics[0], 0) != 0:
                return ()
            option_id = _entity_option(relics[0], "RELIC", "CHEST_0")
            return (
                HeadlessAction.create(
                    "take_chest_relic", option_id, relic_index=0
                ),
            )
        if stage == "proceed":
            return (HeadlessAction.create("leave_room", "LEAVE_ROOM"),)
        return ()

    if decision == "card_reward":
        for fallback, card in enumerate(_objects(state.get("cards"))):
            index = _index(card, fallback)
            option_id = _entity_option(card, "CARD", f"REWARD_{index}")
            actions.append(
                HeadlessAction.create(
                    "select_card_reward", option_id, card_index=index
                )
            )
        if state.get("can_skip") is True:
            actions.append(HeadlessAction.create("skip_card_reward", "SKIP"))
        return tuple(actions)

    if decision == "bundle_select":
        for fallback, bundle in enumerate(_objects(state.get("bundles"))):
            index = _index(bundle, fallback)
            actions.append(
                HeadlessAction.create(
                    "select_bundle", f"BUNDLE.{index}", bundle_index=index
                )
            )
        return tuple(actions)

    if decision == "card_select":
        cards = _objects(state.get("cards"))
        minimum = max(0, int(state.get("min_select", 0)))
        maximum = min(len(cards), max(minimum, int(state.get("max_select", minimum))))
        if minimum == 0:
            actions.append(HeadlessAction.create("skip_select", "SKIP"))
        for count in range(max(1, minimum), maximum + 1):
            for selected in combinations(range(len(cards)), count):
                indices = tuple(_index(cards[position], position) for position in selected)
                joined = ",".join(str(value) for value in indices)
                actions.append(
                    HeadlessAction.create(
                        "select_cards", f"CARDS.{joined}", indices=joined
                    )
                )
                if len(actions) > max_card_selection_actions:
                    raise UnsupportedHeadlessDecision(
                        "Card-selection action space exceeds "
                        f"{max_card_selection_actions} candidates"
                    )
        return tuple(actions)

    if decision == "shop":
        player = state.get("player")
        player = player if isinstance(player, Mapping) else {}
        stage = str(state.get("shop_stage", "inventory_open"))
        if stage == "merchant_ready":
            for fallback, potion in enumerate(_objects(player.get("potions"))):
                if potion.get("can_use_now") is not True:
                    continue
                index = _index(potion, fallback)
                option_id = _entity_option(potion, "POTION", f"SLOT_{index}")
                actions.extend(
                    _targeted_action(
                        command="use_potion",
                        option_id=option_id,
                        index_name="potion_index",
                        index=index,
                        target_type=potion.get("target_type"),
                        enemies=(),
                    )
                )
            actions.append(HeadlessAction.create("open_shop", "OPEN_SHOP"))
            actions.append(HeadlessAction.create("leave_room", "LEAVE_ROOM"))
            return tuple(actions)
        if stage != "inventory_open":
            return ()
        gold = int(player.get("gold", 0))
        for collection, command, argument, prefix in (
            ("cards", "buy_card", "card_index", "CARD"),
            ("relics", "buy_relic", "relic_index", "RELIC"),
            ("potions", "buy_potion", "potion_index", "POTION"),
        ):
            for fallback, item in enumerate(_objects(state.get(collection))):
                if item.get("is_stocked", True) is False:
                    continue
                if int(item.get("cost", gold + 1)) > gold:
                    continue
                index = _index(item, fallback)
                option_id = _entity_option(item, prefix, f"SHOP_{index}")
                actions.append(
                    HeadlessAction.create(command, option_id, **{argument: index})
                )
        removal_cost = state.get("card_removal_cost")
        if removal_cost is not None and int(removal_cost) <= gold:
            actions.append(HeadlessAction.create("remove_card", "REMOVE_CARD"))
        actions.append(HeadlessAction.create("leave_room", "LEAVE_ROOM"))
        return tuple(actions)

    if decision == "unknown":
        return (HeadlessAction.create("proceed", "PROCEED"),)

    if decision in {"event_blocked", "shop_blocked", "treasure_blocked"}:
        return ()

    raise UnsupportedHeadlessDecision(
        f"No legal-action mapping for runtime decision {decision or '<missing>'}"
    )


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _compact_training_value(value: Any) -> Any:
    """Copy JSON data while dropping localized prose that is not a game feature."""

    if isinstance(value, Mapping):
        return {
            str(key): _compact_training_value(item)
            for key, item in value.items()
            if key != "description"
        }
    if isinstance(value, list):
        return [_compact_training_value(item) for item in value]
    return value


def normalize_headless_observation(
    state: Mapping[str, Any], *, character: str, ascension: int, seed: str | None
) -> dict[str, Any]:
    """Wrap runtime state in a stable, character-explicit training schema."""

    context = state.get("context")
    player = state.get("player")
    normalized_player = _compact_training_value(
        player if isinstance(player, Mapping) else {}
    )
    normalized_player["character"] = canonical_character(character)
    for card in _objects(normalized_player.get("deck")):
        if card.get("id"):
            card["id"] = canonical_entity(card["id"], "CARD")
    for relic in _objects(normalized_player.get("relics")):
        relic["id"] = canonical_entity(relic.get("id") or relic.get("name"), "RELIC")
    for potion in _objects(normalized_player.get("potions")):
        potion["id"] = canonical_entity(
            potion.get("id") or potion.get("name"), "POTION"
        )

    reserved = {"type", "decision", "context", "player", "description"}
    decision_state = {
        key: _compact_training_value(value)
        for key, value in state.items()
        if key not in reserved
    }
    return {
        "schema_version": 1,
        "source": "sts2-cli",
        "character": canonical_character(character),
        "ascension": ascension,
        "seed": seed,
        "decision_type": str(state.get("decision", "unknown")),
        "context": _compact_training_value(
            context if isinstance(context, Mapping) else {}
        ),
        "player": normalized_player,
        "decision_state": decision_state,
    }


class Sts2Env:
    """One character-aware STS2 episode backed by one native process."""

    def __init__(
        self,
        config: HeadlessConfig | None = None,
        *,
        client: Any | None = None,
        rewards: RewardConfig | None = None,
        max_card_selection_actions: int = 20_000,
    ):
        if client is not None and config is not None:
            raise ValueError("Pass either config or client, not both")
        self._owns_client = client is None
        self.client = client or Sts2HeadlessClient(config)
        self.rewards = rewards or RewardConfig()
        self.max_card_selection_actions = max_card_selection_actions
        self._raw_state: dict[str, Any] | None = None
        self._observation: dict[str, Any] | None = None
        self._legal_actions: tuple[HeadlessAction, ...] = ()
        self._character = ""
        self._ascension = 0
        self._seed: str | None = None
        self._last_combat_context: dict[str, Any] | None = None
        self._closed = False

    @property
    def protocol_version(self) -> str:
        return str(getattr(self.client, "version", "unknown"))

    @property
    def observation(self) -> Mapping[str, Any]:
        if self._observation is None:
            raise RuntimeError("Call reset before reading the observation")
        return self._observation

    @property
    def raw_state(self) -> Mapping[str, Any]:
        if self._raw_state is None:
            raise RuntimeError("Call reset before reading the raw state")
        return self._raw_state

    @property
    def legal_actions(self) -> tuple[HeadlessAction, ...]:
        return self._legal_actions

    @property
    def action_mask(self) -> tuple[bool, ...]:
        return (True,) * len(self._legal_actions)

    @property
    def terminated(self) -> bool:
        return bool(
            self._raw_state is not None
            and self._raw_state.get("decision") == "game_over"
        )

    def reset(
        self, *, character: str, ascension: int, seed: str | None = None
    ) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if character not in HEADLESS_CHARACTERS:
            raise ValueError(f"Unsupported character: {character}")
        if not 0 <= ascension <= 10:
            raise ValueError("ascension must be between 0 and 10")
        self._character = character
        self._ascension = ascension
        self._seed = seed
        self._last_combat_context = None
        state = self.client.start_run(
            character=character, ascension=ascension, seed=seed
        )
        self._accept_state(state)
        return self.observation

    def step(self, action: int | HeadlessAction) -> StepResult:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._raw_state is None:
            raise RuntimeError("Call reset before step")
        if self.terminated:
            raise RuntimeError("Episode is already terminated")

        if isinstance(action, int):
            if action < 0 or action >= len(self._legal_actions):
                raise IndexError(f"Illegal action index: {action}")
            action_index = action
            selected = self._legal_actions[action]
        else:
            try:
                action_index = self._legal_actions.index(action)
            except ValueError as exc:
                raise ValueError("Action is not legal in the current state") from exc
            selected = action

        previous_decision = str(self._raw_state.get("decision", "unknown"))
        response = self.client.action(selected.command, **selected.argument_dict())
        self._accept_state(response)
        terminated = self.terminated
        if terminated:
            reward = (
                self.rewards.victory
                if self._raw_state and self._raw_state.get("victory") is True
                else self.rewards.defeat
            )
        else:
            reward = self.rewards.nonterminal
        info: dict[str, Any] = {
            "action_index": action_index,
            "action": selected.as_dict(),
            "previous_decision_type": previous_decision,
            "decision_type": self._raw_state.get("decision"),
        }
        if terminated:
            info["victory"] = self._raw_state.get("victory") is True
        return StepResult(self.observation, reward, terminated, False, info)

    def snapshot(self) -> Sts2EnvSnapshot:
        """Freeze an exact seeded decision point for process-isolated search."""

        if self._closed:
            raise HeadlessSnapshotError("Cannot snapshot a closed environment")
        if self._raw_state is None:
            raise HeadlessSnapshotError("Call reset before creating a snapshot")
        if self._seed is None:
            raise HeadlessSnapshotError(
                "Environment reset must use an explicit seed before it can be forked"
            )
        snapshot_method = getattr(self.client, "snapshot", None)
        if not callable(snapshot_method):
            raise HeadlessSnapshotError("Client does not support native snapshots")
        native = snapshot_method()
        native_state = native.state()
        if _json_copy(native_state) != _json_copy(self._raw_state):
            raise HeadlessSnapshotError(
                "Environment state does not match the native reconstructive state"
            )
        context_json = (
            json.dumps(
                self._last_combat_context,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            if self._last_combat_context is not None
            else None
        )
        observation_json = json.dumps(
            self._observation,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return Sts2EnvSnapshot(
            schema_version=1,
            native=native,
            character=self._character,
            ascension=self._ascension,
            seed=self._seed,
            last_combat_context_json=context_json,
            observation_sha256=hashlib.sha256(
                observation_json.encode("utf-8")
            ).hexdigest(),
            legal_actions=self._legal_actions,
        )

    def fork(self, snapshot: Sts2EnvSnapshot | None = None) -> Sts2Env:
        """Return an independent environment reconstructed at the frozen decision."""

        if self._closed:
            raise HeadlessSnapshotError("Cannot fork a closed environment")
        frozen = snapshot or self.snapshot()
        if frozen.schema_version != 1:
            raise HeadlessSnapshotError(
                f"Unsupported environment snapshot schema: {frozen.schema_version}"
            )
        fork_method = getattr(self.client, "fork", None)
        if not callable(fork_method):
            raise HeadlessSnapshotError("Client does not support native forks")
        child_client = fork_method(frozen.native)
        child = Sts2Env(
            client=child_client,
            rewards=self.rewards,
            max_card_selection_actions=self.max_card_selection_actions,
        )
        # This process was created specifically for the fork, so the child owns it
        # even though dependency injection was used to construct the environment.
        child._owns_client = True
        try:
            child._character = frozen.character
            child._ascension = frozen.ascension
            child._seed = frozen.seed
            child._last_combat_context = (
                json.loads(frozen.last_combat_context_json)
                if frozen.last_combat_context_json is not None
                else None
            )
            child._accept_state(child_client.reconstructed_state)
            observation_json = json.dumps(
                child._observation,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            actual_observation_sha = hashlib.sha256(
                observation_json.encode("utf-8")
            ).hexdigest()
            if actual_observation_sha != frozen.observation_sha256:
                raise HeadlessSnapshotError(
                    "Forked environment observation does not match its snapshot"
                )
            if child._legal_actions != frozen.legal_actions:
                raise HeadlessSnapshotError(
                    "Forked environment legal actions do not match their snapshot"
                )
            return child
        except BaseException:
            child.close()
            raise

    def _accept_state(self, state: Mapping[str, Any]) -> None:
        if state.get("type") == "error":
            message = state.get("message", "unknown runtime error")
            logs = tuple(getattr(self.client, "recent_logs", ()))
            if logs:
                message = f"{message}\nRecent runtime logs:\n" + "\n".join(logs[-20:])
            raise HeadlessRuntimeError(str(message))
        if state.get("type") != "decision":
            raise HeadlessRuntimeError(f"Expected decision response, received: {state}")
        copied = _json_copy(dict(state))
        actions = legal_headless_actions(
            copied,
            max_card_selection_actions=self.max_card_selection_actions,
        )
        if copied.get("decision") != "game_over" and not actions:
            raise UnsupportedHeadlessDecision(
                f"Runtime returned no legal actions for {copied.get('decision')}"
            )
        observation = normalize_headless_observation(
            copied,
            character=self._character,
            ascension=self._ascension,
            seed=self._seed,
        )
        if copied.get("decision") == "combat_play":
            self._last_combat_context = {
                "encounter_id": copied.get("encounter_id"),
                "monsters": [
                    canonical_entity(enemy.get("id") or enemy.get("name"), "MONSTER")
                    for enemy in _objects(copied.get("enemies"))
                    if enemy.get("id") or enemy.get("name")
                ],
                "turns_taken": copied.get("round"),
            }
        elif copied.get("decision") == "card_reward" and self._last_combat_context:
            observation["previous_combat"] = _json_copy(self._last_combat_context)
        self._raw_state = copied
        self._legal_actions = actions
        self._observation = observation

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> Sts2Env:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class AtomicTrajectoryRecorder:
    """Write an episode incrementally, publishing it only after a clean commit."""

    def __init__(self, output: str | Path, metadata: Mapping[str, Any]):
        self.output = Path(output)
        self.temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.temporary.exists():
            raise FileExistsError(
                f"Incomplete trajectory already exists: {self.temporary}"
            )
        self.trajectory_id = uuid.uuid4().hex
        self._handle = self.temporary.open("x", encoding="utf-8", newline="\n")
        self._committed = False
        self._write(
            {
                "schema_version": 1,
                "event_type": "episode_start",
                "trajectory_id": self.trajectory_id,
                "recorded_at": _utc_now(),
                **dict(metadata),
            }
        )

    def record_reset(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[HeadlessAction],
    ) -> None:
        self._write(
            {
                "schema_version": 1,
                "event_type": "reset",
                "trajectory_id": self.trajectory_id,
                "observation": observation,
                "legal_actions": [action.as_dict() for action in legal_actions],
                "action_mask": [True] * len(legal_actions),
            }
        )

    def record_transition(
        self,
        *,
        step: int,
        observation: Mapping[str, Any],
        legal_actions: Sequence[HeadlessAction],
        action_index: int,
        action: HeadlessAction,
        result: StepResult,
        next_legal_actions: Sequence[HeadlessAction],
    ) -> None:
        self._write(
            {
                "schema_version": 1,
                "event_type": "transition",
                "trajectory_id": self.trajectory_id,
                "step": step,
                "observation": observation,
                "legal_actions": [item.as_dict() for item in legal_actions],
                "action_mask": [True] * len(legal_actions),
                "action_index": action_index,
                "action": action.as_dict(),
                "reward": result.reward,
                "next_observation": result.observation,
                "next_legal_actions": [
                    item.as_dict() for item in next_legal_actions
                ],
                "next_action_mask": [True] * len(next_legal_actions),
                "terminated": result.terminated,
                "truncated": result.truncated,
                "info": result.info,
            }
        )

    def commit(self, summary: Mapping[str, Any]) -> Path:
        if self._committed:
            return self.output
        self._write(
            {
                "schema_version": 1,
                "event_type": "episode_end",
                "trajectory_id": self.trajectory_id,
                "recorded_at": _utc_now(),
                **dict(summary),
            }
        )
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self.temporary.replace(self.output)
        self._committed = True
        return self.output

    def abort(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def _write(self, event: Mapping[str, Any]) -> None:
        if self._handle.closed:
            raise RuntimeError("Trajectory recorder is closed")
        encoded = json.dumps(event, separators=(",", ":"), ensure_ascii=False)
        self._handle.write(encoded + "\n")
        self._handle.flush()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_headless_rollout(
    *,
    output: str | Path,
    character: str,
    ascension: int,
    seed: str,
    policy_name: str,
    select_action: Callable[[Sts2Env, int], tuple[int, Mapping[str, Any]]],
    policy_metadata: Mapping[str, Any] | None = None,
    max_steps: int = 1_000,
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Run and atomically record one dynamic-candidate policy episode."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    recorder: AtomicTrajectoryRecorder | None = None
    with Sts2Env(config) as env:
        observation = env.reset(
            character=character, ascension=ascension, seed=seed
        )
        recorder = AtomicTrajectoryRecorder(
            output,
            {
                "source": "sts2-cli",
                "protocol_version": env.protocol_version,
                "character": canonical_character(character),
                "ascension": ascension,
                "seed": seed,
                "policy": policy_name,
                "policy_metadata": dict(policy_metadata or {}),
            },
        )
        recorder.record_reset(observation, env.legal_actions)
        total_reward = 0.0
        steps = 0
        terminated = env.terminated
        policy_sources: dict[str, int] = {}
        try:
            while not terminated and steps < max_steps:
                before = env.observation
                candidates = env.legal_actions
                if not candidates:
                    raise UnsupportedHeadlessDecision(
                        "Nonterminal state has no legal actions"
                    )
                action_index, policy_info = select_action(env, steps)
                if action_index < 0 or action_index >= len(candidates):
                    raise IndexError(
                        f"Policy selected illegal action index {action_index} "
                        f"from {len(candidates)} candidates"
                    )
                source = str(policy_info.get("source", policy_name))
                policy_sources[source] = policy_sources.get(source, 0) + 1
                selected = candidates[action_index]
                result = env.step(action_index)
                result = StepResult(
                    observation=result.observation,
                    reward=result.reward,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    info={**dict(result.info), "policy": dict(policy_info)},
                )
                steps += 1
                total_reward += result.reward
                recorder.record_transition(
                    step=steps,
                    observation=before,
                    legal_actions=candidates,
                    action_index=action_index,
                    action=selected,
                    result=result,
                    next_legal_actions=env.legal_actions,
                )
                terminated = result.terminated

            truncated = not terminated
            raw = env.raw_state
            summary = {
                "steps": steps,
                "total_reward": total_reward,
                "terminated": terminated,
                "truncated": truncated,
                "victory": raw.get("victory") is True if terminated else None,
                "act": raw.get("act") or (raw.get("context") or {}).get("act"),
                "floor": raw.get("floor") or (raw.get("context") or {}).get("floor"),
                "policy_sources": policy_sources,
            }
            recorder.commit(summary)
            return {
                **summary,
                "trajectory": str(Path(output).resolve()),
                "trajectory_id": recorder.trajectory_id,
            }
        except BaseException:
            recorder.abort()
            raise


def run_random_rollout(
    *,
    output: str | Path,
    character: str,
    ascension: int,
    seed: str,
    policy_seed: int = 7,
    max_steps: int = 1_000,
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Run a legal-random episode with guards for known runtime-broken actions."""

    rng = random.Random(policy_seed)

    def select_action(env: Sts2Env, _: int) -> tuple[int, Mapping[str, Any]]:
        available = list(range(len(env.legal_actions)))
        guard = None
        if env.raw_state.get("decision") == "shop":
            safe = [
                index
                for index, action in enumerate(env.legal_actions)
                if action.command == "leave_room"
            ]
            if safe:
                available = safe
                guard = "shop_leave_only_v0.107.1"
        chosen = available[rng.randrange(len(available))]
        info: dict[str, Any] = {"source": "legal_random"}
        if guard:
            info["runtime_guard"] = guard
        return chosen, info

    return run_headless_rollout(
        output=output,
        character=character,
        ascension=ascension,
        seed=seed,
        policy_name="legal_random",
        select_action=select_action,
        policy_metadata={
            "policy_seed": policy_seed,
            "runtime_guards": ["shop_leave_only_v0.107.1"],
        },
        max_steps=max_steps,
        config=config,
    )
