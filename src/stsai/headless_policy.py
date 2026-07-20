"""Character-conditioned policy adapters for the native headless environment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from stsai.combat import (
    headless_action_to_combat,
    normalize_headless_combat_observation,
    safe_potion_indexes,
)
from stsai.combat_policy import CombatFeatureHasher
from stsai.dataset import SKIP
from stsai.headless import HeadlessConfig, HeadlessRuntimeError
from stsai.headless_env import (
    HeadlessAction,
    Sts2Env,
    run_headless_rollout,
)
from stsai.ids import canonical_entity
from stsai.policy import FeatureHasher, MaskedMLPPolicy


def _objects(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _by_index(items: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(item.get("index", position)): item for position, item in enumerate(items)}


class HeadlessHybridPolicy:
    """Use the macro BC model where supervised and labeled rules everywhere else."""

    def __init__(
        self,
        macro_model: MaskedMLPPolicy,
        *,
        build_id: str = "v0.107.1",
        use_macro: bool = True,
        combat_model: MaskedMLPPolicy | None = None,
        heuristic_version: str = "v1",
        safe_potion_ids: frozenset[str] | None = None,
    ):
        if heuristic_version not in {"v1", "v2_chip_aware"}:
            raise ValueError(f"Unsupported heuristic version: {heuristic_version}")
        self.macro_model = macro_model
        self.hasher = FeatureHasher(macro_model.dimension)
        self.build_id = build_id
        self.use_macro = use_macro
        self.combat_model = combat_model
        self.heuristic_version = heuristic_version
        self.safe_potion_ids = safe_potion_ids
        self.combat_hasher = (
            CombatFeatureHasher(combat_model.dimension) if combat_model is not None else None
        )
        self._last_fallback_event_state: str | None = None
        self._tried_fallback_event_actions: set[HeadlessAction] = set()

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        build_id: str = "v0.107.1",
        combat_model_path: str | Path | None = None,
    ) -> HeadlessHybridPolicy:
        combat_model = (
            MaskedMLPPolicy.load(combat_model_path) if combat_model_path is not None else None
        )
        return cls(
            MaskedMLPPolicy.load(path),
            build_id=build_id,
            combat_model=combat_model,
        )

    def select(self, env: Sts2Env, _: int = 0) -> tuple[int, Mapping[str, Any]]:
        combat = self._combat_selection(env)
        if combat is not None:
            return combat
        macro = self._macro_selection(env) if self.use_macro else None
        if macro is not None:
            return macro
        scores = [
            self._heuristic_score(env.raw_state, action, env.legal_actions)
            for action in env.legal_actions
        ]
        if self.safe_potion_ids is not None:
            allowed = safe_potion_indexes(
                _mapping(env.raw_state.get("player")), self.safe_potion_ids
            )
            for index, action in enumerate(env.legal_actions):
                if action.command != "use_potion":
                    continue
                potion_index = int(action.argument_dict().get("potion_index", -1))
                if potion_index not in allowed:
                    scores[index] = -1_000_000.0
        best = max(range(len(scores)), key=lambda index: (scores[index], -index))
        guard = self._guard_repeated_event_action(env, best)
        if guard is not None:
            best = guard
        info: dict[str, Any] = {
            "source": "heuristic",
            "policy": f"greedy_rules_{self.heuristic_version}",
            "ranked_actions": _ranked_actions(env.legal_actions, scores),
        }
        if guard is not None:
            info["runtime_guard"] = "repeated_noop_event_alternate_v0.107.1"
        rest_guard = _rest_runtime_guard(env.raw_state, env.legal_actions[best])
        if rest_guard is not None:
            info["runtime_guard"] = rest_guard
        return best, info

    def _combat_selection(self, env: Sts2Env) -> tuple[int, Mapping[str, Any]] | None:
        if (
            self.combat_model is None
            or self.combat_hasher is None
            or str(env.raw_state.get("decision", "")) != "combat_play"
        ):
            return None
        actions = [headless_action_to_combat(action.as_dict()) for action in env.legal_actions]
        record = {
            "decision_type": "combat_action",
            "patch": {"build_id": self.build_id},
            "observation": normalize_headless_combat_observation(env.observation),
        }
        features = np.stack([self.combat_hasher.encode(record, action) for action in actions])
        scores = self.combat_model.scores(features)
        probabilities = np.exp(scores - np.max(scores))
        probabilities /= probabilities.sum()
        order = np.argsort(-scores)
        best = int(order[0])
        return best, {
            "source": "combat_bc",
            "label_source": "deterministic_heuristic_distillation",
            "character": record["observation"]["player"]["character"],
            "ranked_actions": [
                {
                    "action_index": int(index),
                    "action": env.legal_actions[int(index)].as_dict(),
                    "score": round(float(scores[int(index)]), 6),
                    "probability": round(float(probabilities[int(index)]), 6),
                }
                for index in order[:5]
            ],
        }

    def _guard_repeated_event_action(self, env: Sts2Env, proposed_index: int) -> int | None:
        """Avoid replaying one event action after it provably left state unchanged."""

        state = env.raw_state
        if str(state.get("decision", "")) != "event_choice":
            self._last_fallback_event_state = None
            self._tried_fallback_event_actions.clear()
            return None

        fingerprint = json.dumps(
            env.observation, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        proposed = env.legal_actions[proposed_index]
        if fingerprint != self._last_fallback_event_state:
            self._last_fallback_event_state = fingerprint
            self._tried_fallback_event_actions.clear()
        selected_index = proposed_index
        if proposed in self._tried_fallback_event_actions:
            alternative = next(
                (
                    index
                    for index, action in enumerate(env.legal_actions)
                    if action not in self._tried_fallback_event_actions
                ),
                None,
            )
            if alternative is None:
                raise HeadlessRuntimeError(
                    "Every legal event action left the normalized state unchanged"
                )
            selected_index = alternative
        self._tried_fallback_event_actions.add(env.legal_actions[selected_index])
        return selected_index if selected_index != proposed_index else None

    def _macro_selection(self, env: Sts2Env) -> tuple[int, Mapping[str, Any]] | None:
        mapped = self._macro_options(env.raw_state, env.legal_actions)
        if mapped is None:
            return None
        decision_type, option_ids = mapped
        record = {
            "decision_type": decision_type,
            "patch": {"build_id": self.build_id},
            "observation": self._macro_observation(env),
            "legal_options": option_ids,
        }
        features = np.stack([self.hasher.encode(record, option_id) for option_id in option_ids])
        scores = self.macro_model.scores(features)
        probabilities = np.exp(scores - np.max(scores))
        probabilities /= probabilities.sum()
        best = int(np.argmax(scores))
        ranked = sorted(
            (
                {
                    "action_index": index,
                    "option": option_ids[index],
                    "score": round(float(scores[index]), 6),
                    "probability": round(float(probabilities[index]), 6),
                }
                for index in range(len(option_ids))
            ),
            key=lambda item: (-item["score"], item["action_index"]),
        )
        return best, {
            "source": "macro_bc",
            "decision_type": decision_type,
            "chosen_option": option_ids[best],
            "ranked_options": ranked,
        }

    def _macro_options(
        self,
        state: Mapping[str, Any],
        actions: Sequence[HeadlessAction],
    ) -> tuple[str, list[str]] | None:
        decision = str(state.get("decision", ""))
        if decision == "card_reward":
            cards = _by_index(_objects(state.get("cards")))
            options: list[str] = []
            for action in actions:
                if action.command == "skip_card_reward":
                    options.append(SKIP)
                    continue
                if action.command != "select_card_reward":
                    return None
                index = int(action.argument_dict()["card_index"])
                card = cards.get(index)
                if card is None or not card.get("id"):
                    return None
                options.append(canonical_entity(card["id"], "CARD"))
            return ("card_reward", options) if options else None

        event_name = str(state.get("event_name", ""))
        options = _by_index(_objects(state.get("options")))
        is_neow = event_name.lower() == "neow" or any(
            str(option.get("text_key", "")).upper().startswith("NEOW.")
            for option in options.values()
        )
        if decision != "event_choice" or not is_neow:
            return None
        identifiers: list[str] = []
        for action in actions:
            if action.command != "choose_option":
                return None
            index = int(action.argument_dict()["option_index"])
            option = options.get(index)
            if option is None:
                return None
            text_key = str(option.get("text_key", ""))
            stable = text_key.rsplit(".", 1)[-1] if text_key else option.get("title")
            identifiers.append(canonical_entity(stable, "ANCIENT"))
        return ("ancient", identifiers) if identifiers else None

    def _macro_observation(self, env: Sts2Env) -> dict[str, Any]:
        observation = env.observation
        player = _mapping(observation.get("player"))
        context = _mapping(observation.get("context"))
        deck = [
            canonical_entity(card.get("id"), "CARD")
            for card in _objects(player.get("deck"))
            if card.get("id")
        ]
        relics = [
            canonical_entity(relic.get("id") or relic.get("name"), "RELIC")
            for relic in _objects(player.get("relics"))
            if relic.get("id") or relic.get("name")
        ]
        state = env.raw_state
        previous_combat = _mapping(observation.get("previous_combat"))
        monsters = [str(value) for value in previous_combat.get("monsters") or []]
        if not monsters:
            monsters = [
                canonical_entity(enemy.get("id") or enemy.get("name"), "MONSTER")
                for enemy in _objects(state.get("enemies"))
                if enemy.get("id") or enemy.get("name")
            ]
        event_name = state.get("event_name")
        room_id = state.get("encounter_id") or previous_combat.get("encounter_id")
        if not room_id and event_name:
            room_id = canonical_entity(event_name, "EVENT")
        return {
            "character": observation["character"],
            "ascension": int(observation.get("ascension", 0)),
            "act": int(context.get("act", 0)),
            "floor": int(context.get("floor", 0)),
            "current_hp": int(player.get("hp", 0)),
            "max_hp": int(player.get("max_hp", 0)),
            "current_gold": int(player.get("gold", 0)),
            "deck": deck,
            "relics": sorted(relics),
            "map_point_type": str(context.get("room_type", "unknown")).lower(),
            "room_id": room_id,
            "monsters": monsters,
            "turns_taken": (
                state.get("turns_taken") or state.get("round") or previous_combat.get("turns_taken")
            ),
        }

    def _heuristic_score(
        self,
        state: Mapping[str, Any],
        action: HeadlessAction,
        candidates: Sequence[HeadlessAction],
    ) -> float:
        decision = str(state.get("decision", ""))
        if decision == "combat_play":
            return _combat_score(
                state,
                action,
                candidates,
                chip_aware=self.heuristic_version == "v2_chip_aware",
            )
        if decision == "map_select":
            return _map_score(state, action)
        if decision == "rest_site":
            return _rest_score(state, action)
        if decision == "shop":
            return {
                "buy_relic": -100.0,
                "remove_card": -100.0,
                "buy_card": -100.0,
                "buy_potion": -100.0,
                "leave_room": 100.0,
            }.get(action.command, -100.0)
        if decision == "card_reward":
            return 10.0 if action.command == "select_card_reward" else 0.0
        if decision == "card_select":
            if action.command == "skip_select":
                return -1.0
            indices = str(action.argument_dict().get("indices", ""))
            count = len([value for value in indices.split(",") if value])
            return 10.0 - count * 0.01
        if decision == "bundle_select":
            return -_number(action.argument_dict().get("bundle_index"))
        if decision == "event_choice":
            return _event_score(state, action)
        if decision == "unknown":
            index = action.argument_dict().get("option_index", 0)
            return -_number(index)
        return 0.0


def _incoming_damage(state: Mapping[str, Any]) -> float:
    total = 0.0
    for enemy in _objects(state.get("enemies")):
        for intent in _objects(enemy.get("intents")):
            if intent.get("total_damage") is not None:
                total += _number(intent.get("total_damage"))
            else:
                total += _number(intent.get("damage")) * max(_number(intent.get("hits"), 1.0), 1.0)
    return total


def _combat_score(
    state: Mapping[str, Any],
    action: HeadlessAction,
    candidates: Sequence[HeadlessAction],
    *,
    chip_aware: bool = False,
) -> float:
    arguments = action.argument_dict()
    playable_exists = any(candidate.command == "play_card" for candidate in candidates)
    if action.command == "end_turn":
        return -1_000.0 if playable_exists else 0.0

    player = _mapping(state.get("player"))
    hp = _number(player.get("hp"))
    max_hp = max(_number(player.get("max_hp")), 1.0)
    incoming = max(0.0, _incoming_damage(state) - _number(player.get("block")))
    if action.command == "use_potion":
        if hp / max_hp < 0.45 or incoming >= hp:
            return 800.0
        return -10.0

    if action.command != "play_card":
        return -100.0
    cards = _by_index(_objects(state.get("hand")))
    card = cards.get(int(arguments["card_index"]), {})
    card_type = str(card.get("type", "")).lower()
    stats = _mapping(card.get("stats"))
    cost = _number(card.get("cost"))
    score = 100.0 - cost

    if card_type == "attack":
        target_index = int(arguments.get("target_index", 0))
        damage = 0.0
        repeat = 1.0
        for row in _objects(card.get("damage_by_target")):
            if int(row.get("target_index", -1)) == target_index:
                damage = _number(row.get("damage"))
                repeat = max(_number(row.get("repeat"), 1.0), 1.0)
                if row.get("total_damage") is not None:
                    damage = _number(row.get("total_damage"))
                    repeat = 1.0
                break
        if damage == 0:
            damage = _number(stats.get("calculateddamage") or stats.get("damage"))
            repeat = max(_number(stats.get("repeat"), 1.0), 1.0)
        total_damage = damage * repeat
        enemies = _by_index(_objects(state.get("enemies")))
        target = enemies.get(target_index, {})
        effective_hp = _number(target.get("hp")) + _number(target.get("block"))
        score += 100.0 + total_damage * 4.0
        if total_damage >= effective_hp > 0:
            score += 1_000.0
        return score

    if card_type == "skill":
        block = _number(stats.get("block"))
        prevented = min(block, incoming)
        if chip_aware:
            hp_ratio = hp / max_hp
            prevention_weight = 22.0 + 16.0 * (1.0 - hp_ratio)
            score += 45.0 + prevented * prevention_weight + block
            if incoming >= hp and max(incoming - block, 0.0) < hp:
                score += 1_000.0
        else:
            score += 80.0 + prevented * 5.0 + block
        return score
    if card_type == "power":
        return score + 140.0
    return score + 40.0


def _map_score(state: Mapping[str, Any], action: HeadlessAction) -> float:
    arguments = action.argument_dict()
    choices = _objects(state.get("choices"))
    chosen = next(
        (
            choice
            for choice in choices
            if int(choice.get("col", -1)) == int(arguments.get("col", -2))
            and int(choice.get("row", -1)) == int(arguments.get("row", -2))
        ),
        {},
    )
    room = str(chosen.get("type", "unknown")).lower().replace("_", "")
    player = _mapping(state.get("player"))
    hp_ratio = _number(player.get("hp")) / max(_number(player.get("max_hp")), 1.0)
    if room in {"rest", "restsite"}:
        return 120.0 if hp_ratio < 0.7 else 25.0
    if room == "treasure":
        return 90.0
    if room in {"shop", "merchant"}:
        return 75.0 if _number(player.get("gold")) >= 100 else 20.0
    if room in {"unknown", "event"}:
        return 60.0
    if room in {"monster", "combat"}:
        return 45.0
    if room == "elite":
        return 35.0 if hp_ratio >= 0.85 else -30.0
    if room == "boss":
        return 0.0
    return 10.0


def _rest_score(state: Mapping[str, Any], action: HeadlessAction) -> float:
    index = int(action.argument_dict().get("option_index", -1))
    options = _by_index(_objects(state.get("options")))
    option_id = str(options.get(index, {}).get("option_id", "")).upper()
    player = _mapping(state.get("player"))
    hp_ratio = _number(player.get("hp")) / max(_number(player.get("max_hp")), 1.0)
    if option_id == "HEAL":
        if _tiny_mailbox_rewards_would_overflow(player):
            return -1_000.0
        return 120.0 if hp_ratio < 0.75 else 10.0
    if option_id in {"SMITH", "UPGRADE"}:
        return 90.0
    return 50.0


def _tiny_mailbox_rewards_would_overflow(player: Mapping[str, Any]) -> bool:
    relic_ids = {
        canonical_entity(relic.get("id") or relic.get("name"), "RELIC")
        for relic in _objects(player.get("relics"))
    }
    if "RELIC.TINY_MAILBOX" not in relic_ids:
        return False
    slots = int(_number(player.get("potion_slot_count"), 3.0))
    occupied = len(_objects(player.get("potions")))
    return slots - occupied < 2


def _rest_runtime_guard(state: Mapping[str, Any], action: HeadlessAction) -> str | None:
    if str(state.get("decision", "")) != "rest_site":
        return None
    index = int(action.argument_dict().get("option_index", -1))
    option = _by_index(_objects(state.get("options"))).get(index, {})
    if str(option.get("option_id", "")).upper() not in {"SMITH", "UPGRADE"}:
        return None
    if _tiny_mailbox_rewards_would_overflow(_mapping(state.get("player"))):
        return "tiny_mailbox_overflow_avoid_heal_v0.107.1"
    return None


def _event_score(state: Mapping[str, Any], action: HeadlessAction) -> float:
    if action.command == "leave_room":
        return 200.0
    index = int(action.argument_dict().get("option_index", -1))
    option = _by_index(_objects(state.get("options"))).get(index, {})
    text = " ".join(str(option.get(field, "")) for field in ("title", "text_key")).lower()
    safe_exit_markers = (
        "give up",
        "leave",
        "escape",
        "walk away",
        "depart",
        "stop",
        "back away",
    )
    if any(marker in text for marker in safe_exit_markers):
        return 150.0
    escalating_markers = ("lose everything", "keep ", "continue ")
    if any(marker in text for marker in escalating_markers):
        return -50.0
    return -float(index)


def _ranked_actions(
    actions: Sequence[HeadlessAction], scores: Sequence[float]
) -> list[dict[str, Any]]:
    order = sorted(range(len(actions)), key=lambda index: (-scores[index], index))
    return [
        {
            "action_index": index,
            "action": actions[index].as_dict(),
            "score": round(float(scores[index]), 6),
        }
        for index in order[:5]
    ]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_hybrid_rollout(
    *,
    model_path: str | Path,
    output: str | Path,
    character: str,
    ascension: int,
    seed: str,
    max_steps: int = 1_000,
    build_id: str = "v0.107.1",
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Run the macro-BC plus explicitly heuristic full-game policy."""

    model_path = Path(model_path)
    policy = HeadlessHybridPolicy.from_path(model_path, build_id=build_id)
    return run_headless_rollout(
        output=output,
        character=character,
        ascension=ascension,
        seed=seed,
        policy_name="hybrid_macro_bc_greedy_v1",
        select_action=policy.select,
        policy_metadata={
            "macro_model": str(model_path.resolve()),
            "macro_model_sha256": _sha256(model_path),
            "build_id": build_id,
            "learned_decisions": ["ancient", "card_reward"],
            "fallback": "greedy_rules_v1",
            "runtime_guards": [
                "shop_leave_only_v0.107.1",
                "repeated_noop_event_alternate_v0.107.1",
            ],
        },
        max_steps=max_steps,
        config=config,
    )


def run_heuristic_rollout(
    *,
    model_path: str | Path,
    output: str | Path,
    character: str,
    ascension: int,
    seed: str,
    max_steps: int = 1_000,
    build_id: str = "v0.107.1",
    heuristic_version: str = "v1",
    safe_potion_ids: Sequence[str] | None = None,
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Run the exact hybrid fallback with all learned scoring disabled."""

    model_path = Path(model_path)
    policy = HeadlessHybridPolicy(
        MaskedMLPPolicy.load(model_path),
        build_id=build_id,
        use_macro=False,
        heuristic_version=heuristic_version,
        safe_potion_ids=(
            None if safe_potion_ids is None else frozenset(safe_potion_ids)
        ),
    )
    policy_name = f"greedy_rules_{heuristic_version}"
    return run_headless_rollout(
        output=output,
        character=character,
        ascension=ascension,
        seed=seed,
        policy_name=policy_name,
        select_action=policy.select,
        policy_metadata={
            "ablation": "macro_bc_disabled",
            "build_id": build_id,
            "heuristic_version": heuristic_version,
            "safe_potion_ids": (
                "all" if safe_potion_ids is None else sorted(safe_potion_ids)
            ),
            "runtime_guards": [
                "shop_leave_only_v0.107.1",
                "repeated_noop_event_alternate_v0.107.1",
            ],
        },
        max_steps=max_steps,
        config=config,
    )


def run_combat_distilled_rollout(
    *,
    macro_model_path: str | Path,
    combat_model_path: str | Path,
    output: str | Path,
    character: str,
    ascension: int,
    seed: str,
    max_steps: int = 1_000,
    build_id: str = "v0.107.1",
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Use combat BC only in combat and the exact heuristic elsewhere."""

    macro_model_path = Path(macro_model_path)
    combat_model_path = Path(combat_model_path)
    policy = HeadlessHybridPolicy(
        MaskedMLPPolicy.load(macro_model_path),
        build_id=build_id,
        use_macro=False,
        combat_model=MaskedMLPPolicy.load(combat_model_path),
    )
    return run_headless_rollout(
        output=output,
        character=character,
        ascension=ascension,
        seed=seed,
        policy_name="combat_bc_greedy_noncombat_v1",
        select_action=policy.select,
        policy_metadata={
            "combat_model": str(combat_model_path.resolve()),
            "combat_model_sha256": _sha256(combat_model_path),
            "macro_model_loaded_for_interface_only": str(macro_model_path.resolve()),
            "macro_model_sha256": _sha256(macro_model_path),
            "build_id": build_id,
            "learned_decisions": ["combat_play"],
            "label_source": "deterministic_heuristic_distillation",
            "noncombat_policy": "greedy_rules_v1",
        },
        max_steps=max_steps,
        config=config,
    )
