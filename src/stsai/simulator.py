"""Small, strict headless combat kernel for patch-versioned STS2 mechanics.

This module intentionally implements only mechanics represented by a loaded
catalog. Unknown cards and unsupported phase transitions fail loudly so search
cannot optimize against an accidental approximation of the real game.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from stsai.combat import action_key
from stsai.ids import canonical_character, canonical_entity


SUPPORTED_EFFECTS = frozenset(
    {"damage", "random_damage", "block", "apply_power", "discard_from_hand"}
)
SUPPORTED_TARGETS = frozenset({"none", "self", "any_enemy", "random_enemy"})
_U32_MASK = (1 << 32) - 1
_U64_MASK = (1 << 64) - 1


class UnsupportedMechanic(NotImplementedError):
    """Raised instead of silently approximating an unimplemented STS2 mechanic."""


@dataclass(frozen=True)
class EffectSpec:
    op: str
    amount: int
    power: str | None = None
    hits: int = 1


@dataclass(frozen=True)
class CardSpec:
    card_id: str
    cost: int
    target: str
    effects: tuple[EffectSpec, ...]
    upgraded_effects: tuple[EffectSpec, ...]
    playable: bool = True
    keywords: tuple[str, ...] = ()

    def effects_for(self, upgraded: bool) -> tuple[EffectSpec, ...]:
        return self.upgraded_effects if upgraded else self.effects


@dataclass(frozen=True)
class CardCatalog:
    patch: str
    character: str
    cards: Mapping[str, CardSpec]

    @classmethod
    def from_path(cls, path: str | Path) -> CardCatalog:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported card catalog schema: {source}")
        patch = str(payload.get("patch", "")).strip()
        if not patch:
            raise ValueError("Card catalog requires a patch")
        character = canonical_character(payload.get("character"))
        raw_cards = payload.get("cards")
        if not isinstance(raw_cards, Mapping) or not raw_cards:
            raise ValueError("Card catalog requires at least one card")
        cards: dict[str, CardSpec] = {}
        for raw_id, raw_spec in raw_cards.items():
            if not isinstance(raw_spec, Mapping):
                raise ValueError(f"Invalid card spec for {raw_id}")
            card_id = canonical_entity(raw_id, "CARD")
            target = str(raw_spec.get("target", "")).lower()
            if target not in SUPPORTED_TARGETS:
                raise UnsupportedMechanic(f"Unsupported target type for {card_id}: {target}")
            playable = raw_spec.get("playable") is not False
            effects = _read_effects(
                card_id, raw_spec.get("effects"), allow_empty=not playable
            )
            upgraded = _read_effects(
                card_id, raw_spec.get("upgraded_effects"), allow_empty=not playable
            )
            cards[card_id] = CardSpec(
                card_id=card_id,
                cost=int(raw_spec["cost"]),
                target=target,
                effects=effects,
                upgraded_effects=upgraded,
                playable=playable,
                keywords=tuple(str(value).upper() for value in raw_spec.get("keywords", [])),
            )
        return cls(patch=patch, character=character, cards=cards)

    def require(self, card_id: str) -> CardSpec:
        canonical = canonical_entity(card_id, "CARD")
        try:
            return self.cards[canonical]
        except KeyError as exc:
            raise UnsupportedMechanic(
                f"{canonical} is not implemented for {self.character} on {self.patch}"
            ) from exc


def _read_effects(
    card_id: str, value: Any, *, allow_empty: bool = False
) -> tuple[EffectSpec, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{card_id} requires a non-empty effect list")
    result: list[EffectSpec] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError(f"Invalid effect for {card_id}")
        op = str(raw.get("op", ""))
        if op not in SUPPORTED_EFFECTS:
            raise UnsupportedMechanic(f"Unsupported effect for {card_id}: {op}")
        power = str(raw["power"]).upper() if raw.get("power") is not None else None
        result.append(
            EffectSpec(
                op=op,
                amount=int(raw["amount"]),
                power=power,
                hits=int(raw.get("hits", 1)),
            )
        )
    return tuple(result)


def _int32(value: int) -> int:
    value &= _U32_MASK
    return value if value < (1 << 31) else value - (1 << 32)


def deterministic_string_hash(value: str) -> int:
    """Match STS2 StringHelper.GetDeterministicHashCode with int overflow."""

    first = 352654597
    second = first
    for index in range(0, len(value), 2):
        first = _int32(_int32((first << 5) + first) ^ ord(value[index]))
        if index == len(value) - 1:
            break
        second = _int32(_int32((second << 5) + second) ^ ord(value[index + 1]))
    return _int32(first + _int32(second * 1566083941))


def _rotate_left(value: int, bits: int) -> int:
    return ((value << bits) | (value >> (64 - bits))) & _U64_MASK


def _splitmix64(value: int) -> tuple[int, int]:
    state = (value + 11400714819323198485) & _U64_MASK
    mixed = state
    mixed = ((mixed ^ (mixed >> 30)) * 13787848793156543929) & _U64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * 10723151780598845931) & _U64_MASK
    return state, (mixed ^ (mixed >> 31)) & _U64_MASK


@dataclass(frozen=True)
class StsRngState:
    """Serializable STS2 Xoshiro256** RNG state and public-call counter."""

    s0: int
    s1: int
    s2: int
    s3: int
    counter: int = 0

    @classmethod
    def from_seed(cls, seed: int, counter: int = 0) -> StsRngState:
        split_state = seed & _U64_MASK
        words: list[int] = []
        for _ in range(4):
            split_state, word = _splitmix64(split_state)
            words.append(word)
        result = cls(*words)
        for _ in range(counter):
            _, result = result.next_ulong()
        return replace(result, counter=counter)

    @classmethod
    def for_run_stream(
        cls, run_seed: str, stream_name: str, counter: int = 0
    ) -> StsRngState:
        run_hash = deterministic_string_hash(run_seed) & _U32_MASK
        stream_hash = deterministic_string_hash(stream_name) & _U32_MASK
        return cls.from_seed((run_hash + stream_hash) & _U32_MASK, counter)

    @classmethod
    def combat_targets(cls, run_seed: str, counter: int = 0) -> StsRngState:
        return cls.for_run_stream(run_seed, "combat_targets", counter)

    @classmethod
    def shuffle(cls, run_seed: str, counter: int = 0) -> StsRngState:
        return cls.for_run_stream(run_seed, "shuffle", counter)

    def next_ulong(self) -> tuple[int, StsRngState]:
        result = (_rotate_left((self.s1 * 5) & _U64_MASK, 7) * 9) & _U64_MASK
        shifted = (self.s1 << 17) & _U64_MASK
        s2 = self.s2 ^ self.s0
        s3 = self.s3 ^ self.s1
        s1 = self.s1 ^ s2
        s0 = self.s0 ^ s3
        s2 ^= shifted
        s3 = _rotate_left(s3, 45)
        return result, StsRngState(s0, s1, s2, s3, self.counter + 1)

    def next_int(self, maximum: int) -> tuple[int, StsRngState]:
        if maximum < 1:
            raise ValueError("maximum must be positive")
        raw, state = self.next_ulong()
        value = int(((raw >> 11) * (2.0 ** -53)) * maximum)
        return value, state

    def shuffle_list(self, values: Sequence[Any]) -> tuple[list[Any], StsRngState]:
        result = list(values)
        state = self
        remaining = len(result)
        while remaining > 1:
            remaining -= 1
            selected, state = state.next_int(remaining + 1)
            result[selected], result[remaining] = result[remaining], result[selected]
        return result, state


@dataclass(frozen=True)
class PowerState:
    power_id: str
    amount: int


@dataclass(frozen=True)
class CreatureState:
    entity_id: str
    hp: int
    max_hp: int
    block: int = 0
    powers: tuple[PowerState, ...] = ()

    @property
    def alive(self) -> bool:
        return self.hp > 0


@dataclass(frozen=True)
class CardInstance:
    uid: int
    card_id: str
    upgraded: bool = False


@dataclass(frozen=True)
class PendingSelection:
    kind: str
    remaining: int


@dataclass(frozen=True)
class CombatState:
    patch: str
    character: str
    turn: int
    energy: int
    max_energy: int
    player: CreatureState
    enemies: tuple[CreatureState, ...]
    hand: tuple[CardInstance, ...]
    draw_pile: tuple[CardInstance, ...] = ()
    discard_pile: tuple[CardInstance, ...] = ()
    exhaust_pile: tuple[CardInstance, ...] = ()
    pending_selection: PendingSelection | None = None
    combat_targets_rng: StsRngState | None = None
    terminal: bool = False


@dataclass(frozen=True)
class SimulationResult:
    state: CombatState
    events: tuple[Mapping[str, Any], ...]


class PlayerPhaseSimulator:
    """Exact-within-catalog simulator for player actions in one combat turn."""

    def __init__(self, catalog: CardCatalog):
        self.catalog = catalog

    def legal_actions(self, state: CombatState) -> list[dict[str, Any]]:
        self._validate_state(state)
        if state.terminal:
            return []
        if state.pending_selection is not None:
            if state.pending_selection.kind != "discard_from_hand":
                raise UnsupportedMechanic(
                    f"Unsupported selection: {state.pending_selection.kind}"
                )
            return [
                {"action": "combat_select_card", "card_index": index}
                for index in range(len(state.hand))
            ]

        actions: list[dict[str, Any]] = []
        living_targets = [enemy.entity_id for enemy in state.enemies if enemy.alive]
        for index, card in enumerate(state.hand):
            spec = self.catalog.require(card.card_id)
            if not spec.playable or spec.cost > state.energy:
                continue
            base = {"action": "play_card", "card_index": index}
            if spec.target == "any_enemy":
                actions.extend({**base, "target": target} for target in living_targets)
            else:
                actions.append(base)
        actions.append({"action": "end_turn"})
        return actions

    def step(self, state: CombatState, action: Mapping[str, Any]) -> SimulationResult:
        legal = self.legal_actions(state)
        if action_key(action) not in {action_key(candidate) for candidate in legal}:
            raise ValueError(f"Illegal action: {dict(action)}")
        name = str(action["action"])
        if name == "end_turn":
            raise UnsupportedMechanic(
                "Enemy turns and draw/shuffle hooks are not implemented in the v0 kernel"
            )
        if name == "combat_select_card":
            return self._select_card(state, int(action["card_index"]))
        return self._play_card(state, int(action["card_index"]), action.get("target"))

    def _validate_state(self, state: CombatState) -> None:
        if state.patch != self.catalog.patch:
            raise ValueError(
                f"State patch {state.patch} does not match catalog {self.catalog.patch}"
            )
        if canonical_character(state.character) != self.catalog.character:
            raise ValueError(
                f"State character {state.character} does not match catalog "
                f"{self.catalog.character}"
            )
        if state.energy < 0 or state.max_energy < 0:
            raise ValueError("Energy cannot be negative")

    def _select_card(self, state: CombatState, index: int) -> SimulationResult:
        selection = state.pending_selection
        if selection is None or selection.kind != "discard_from_hand":
            raise ValueError("No discard selection is active")
        selected = state.hand[index]
        hand = state.hand[:index] + state.hand[index + 1 :]
        remaining = selection.remaining - 1
        next_selection = replace(selection, remaining=remaining) if remaining > 0 else None
        next_state = replace(
            state,
            hand=hand,
            discard_pile=state.discard_pile + (selected,),
            pending_selection=next_selection,
        )
        return SimulationResult(
            next_state,
            ({"event": "discard", "card_id": selected.card_id, "uid": selected.uid},),
        )

    def _play_card(
        self, state: CombatState, index: int, raw_target: Any
    ) -> SimulationResult:
        card = state.hand[index]
        spec = self.catalog.require(card.card_id)
        hand = state.hand[:index] + state.hand[index + 1 :]
        next_state = replace(
            state,
            energy=state.energy - spec.cost,
            hand=hand,
            discard_pile=state.discard_pile + (card,),
        )
        target = str(raw_target) if raw_target is not None else None
        events: list[Mapping[str, Any]] = [
            {"event": "play_card", "card_id": card.card_id, "uid": card.uid}
        ]
        for effect in spec.effects_for(card.upgraded):
            if effect.op == "damage":
                next_state, event = _damage_enemy(next_state, target, effect.amount)
                events.append(event)
            elif effect.op == "random_damage":
                next_state, random_events = _random_damage(
                    next_state, effect.amount, effect.hits
                )
                events.extend(random_events)
            elif effect.op == "block":
                player = replace(
                    next_state.player, block=next_state.player.block + effect.amount
                )
                next_state = replace(next_state, player=player)
                events.append({"event": "gain_block", "amount": effect.amount})
            elif effect.op == "apply_power":
                next_state, event = _apply_enemy_power(
                    next_state, target, effect.power or "UNKNOWN", effect.amount
                )
                events.append(event)
            elif effect.op == "discard_from_hand":
                if len(next_state.hand) < effect.amount:
                    raise UnsupportedMechanic(
                        "Forced discard with fewer cards than requested needs game-hook semantics"
                    )
                next_state = replace(
                    next_state,
                    pending_selection=PendingSelection(effect.op, effect.amount),
                )
                events.append({"event": "selection_started", "kind": effect.op})
            else:  # protected by catalog validation
                raise UnsupportedMechanic(effect.op)
        if next_state.enemies and not any(enemy.alive for enemy in next_state.enemies):
            next_state = replace(next_state, terminal=True)
        return SimulationResult(next_state, tuple(events))


def _enemy_index(state: CombatState, target: str | None) -> int:
    if target is None:
        raise ValueError("Target is required")
    for index, enemy in enumerate(state.enemies):
        if enemy.entity_id == target and enemy.alive:
            return index
    raise ValueError(f"Unknown or defeated target: {target}")


def _damage_enemy(
    state: CombatState, target: str | None, amount: int
) -> tuple[CombatState, Mapping[str, Any]]:
    index = _enemy_index(state, target)
    enemy = state.enemies[index]
    absorbed = min(enemy.block, amount)
    hp_loss = min(enemy.hp, amount - absorbed)
    updated = replace(
        enemy,
        block=enemy.block - absorbed,
        hp=enemy.hp - hp_loss,
    )
    enemies = state.enemies[:index] + (updated,) + state.enemies[index + 1 :]
    return replace(state, enemies=enemies), {
        "event": "damage",
        "target": target,
        "amount": amount,
        "blocked": absorbed,
        "hp_loss": hp_loss,
    }


def _apply_enemy_power(
    state: CombatState, target: str | None, power_id: str, amount: int
) -> tuple[CombatState, Mapping[str, Any]]:
    index = _enemy_index(state, target)
    enemy = state.enemies[index]
    powers = list(enemy.powers)
    for power_index, power in enumerate(powers):
        if power.power_id == power_id:
            powers[power_index] = replace(power, amount=power.amount + amount)
            break
    else:
        powers.append(PowerState(power_id, amount))
    updated = replace(enemy, powers=tuple(powers))
    enemies = state.enemies[:index] + (updated,) + state.enemies[index + 1 :]
    return replace(state, enemies=enemies), {
        "event": "apply_power",
        "target": target,
        "power": power_id,
        "amount": amount,
    }


def _random_damage(
    state: CombatState, amount: int, hits: int
) -> tuple[CombatState, tuple[Mapping[str, Any], ...]]:
    rng = state.combat_targets_rng
    if rng is None:
        raise UnsupportedMechanic("Random targeting requires the combat-target RNG state")
    next_state = state
    events: list[Mapping[str, Any]] = []
    for _ in range(hits):
        targets = [enemy.entity_id for enemy in next_state.enemies if enemy.alive]
        if not targets:
            break
        target_index, rng = rng.next_int(len(targets))
        next_state, event = _damage_enemy(next_state, targets[target_index], amount)
        events.append(event)
    return replace(next_state, combat_targets_rng=rng), tuple(events)


def audit_replay_coverage(
    replay_path: str | Path, catalog: CardCatalog
) -> dict[str, Any]:
    """Report which native replay card plays the current catalog can simulate."""

    payload = json.loads(Path(replay_path).read_text(encoding="utf-8-sig"))
    if payload.get("format") != "sts2_combat_replay_summary":
        raise ValueError("Expected an sts2_combat_replay_summary")
    players = payload.get("replay", {}).get("players", [])
    replay_character = canonical_character(players[0].get("character")) if players else None
    counts: Counter[str] = Counter()
    for action in payload.get("actions", []):
        if action.get("action_type") != "NetPlayCardAction":
            continue
        counts[canonical_entity(action.get("card_id"), "CARD")] += 1
    supported = sum(count for card_id, count in counts.items() if card_id in catalog.cards)
    total = sum(counts.values())
    missing = {
        card_id: count for card_id, count in sorted(counts.items()) if card_id not in catalog.cards
    }
    return {
        "schema_version": 1,
        "patch": catalog.patch,
        "catalog_character": catalog.character,
        "replay_character": replay_character,
        "character_match": replay_character == catalog.character,
        "card_plays": total,
        "supported_card_plays": supported,
        "coverage": supported / total if total else 1.0,
        "missing_card_plays": missing,
    }


def audit_replay_hands(
    replay_path: str | Path, catalog: CardCatalog
) -> dict[str, Any]:
    """Reconstruct hand membership from the native deck, RNG, actions, and choices."""

    payload = json.loads(Path(replay_path).read_text(encoding="utf-8-sig"))
    if payload.get("format") != "sts2_combat_replay_summary":
        raise ValueError("Expected an sts2_combat_replay_summary")
    replay = payload.get("replay", {})
    players = replay.get("players", [])
    if not players:
        raise ValueError("Replay has no players")
    player = players[0]
    if canonical_character(player.get("character")) != catalog.character:
        raise ValueError("Replay character does not match the simulator catalog")
    deck_payload = player.get("deck")
    if not isinstance(deck_payload, list) or not deck_payload:
        raise ValueError("Replay export does not include the initial deck")
    run_rng = replay.get("run_rng", {})
    run_seed = str(run_rng.get("seed", ""))
    counters = run_rng.get("counters", {})
    if not run_seed or not isinstance(counters, Mapping):
        raise ValueError("Replay export does not include run RNG state")

    cards = [
        {
            "combat_card_index": None,
            "card_id": canonical_entity(card.get("card_id"), "CARD"),
            "upgrade_level": int(card.get("upgrade_level", 0)),
            "sort_id": card.get("sort_id"),
        }
        for card in deck_payload
    ]
    for card in cards:
        catalog.require(card["card_id"])
    shuffle_rng = StsRngState.shuffle(run_seed, int(counters.get("Shuffle", 0)))
    draw_pile, shuffle_rng = shuffle_rng.shuffle_list(cards)
    for combat_index, card in enumerate(draw_pile):
        card["combat_card_index"] = combat_index

    hand: list[dict[str, Any]] = []
    discard: list[dict[str, Any]] = []
    exhaust: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    relics = {str(relic) for relic in player.get("relics", [])}
    initial_draw = 5 + (2 if "RING_OF_THE_SNAKE" in relics else 0)

    def stable_key(card: Mapping[str, Any]) -> tuple[Any, ...]:
        # STS2's stable reshuffle sorts by ModelId net entry then upgrade level.
        # New exports can provide sort_id; card_id is deterministic fallback for
        # earlier summaries.
        sort_id = card.get("sort_id")
        return (
            0 if sort_id is not None else 1,
            int(sort_id) if sort_id is not None else str(card["card_id"]),
            int(card["upgrade_level"]),
        )

    def draw_cards(count: int) -> None:
        nonlocal draw_pile, discard, shuffle_rng
        for _ in range(count):
            if not draw_pile and discard:
                ordered = sorted(discard, key=stable_key)
                discard = []
                draw_pile, shuffle_rng = shuffle_rng.shuffle_list(ordered)
            if not draw_pile:
                break
            hand.append(draw_pile.pop(0))

    def snapshot(turn: int) -> None:
        snapshots.append(
            {
                "turn": turn,
                "hand": [
                    {
                        "combat_card_index": card["combat_card_index"],
                        "card_id": card["card_id"],
                    }
                    for card in hand
                ],
            }
        )

    draw_cards(initial_draw)
    snapshot(1)
    timeline = sorted(
        list(payload.get("actions", [])) + list(payload.get("choices", [])),
        key=lambda event: int(event.get("replay_event_index", -1)),
    )
    for event in timeline:
        event_index = int(event.get("replay_event_index", -1))
        if event.get("action_type") == "NetPlayCardAction":
            combat_index = int(event["combat_card_index"])
            expected_id = canonical_entity(event.get("card_id"), "CARD")
            matches = [
                card for card in hand if card["combat_card_index"] == combat_index
            ]
            if not matches or matches[0]["card_id"] != expected_id:
                mismatches.append(
                    {
                        "replay_event_index": event_index,
                        "kind": "play_card_not_in_hand",
                        "combat_card_index": combat_index,
                        "card_id": expected_id,
                    }
                )
                continue
            hand.remove(matches[0])
            discard.append(matches[0])
        elif event.get("choice_type") == "CombatCard":
            for raw_index in event.get("combat_card_indices", []):
                combat_index = int(raw_index)
                matches = [
                    card for card in hand if card["combat_card_index"] == combat_index
                ]
                if not matches:
                    mismatches.append(
                        {
                            "replay_event_index": event_index,
                            "kind": "selected_card_not_in_hand",
                            "combat_card_index": combat_index,
                        }
                    )
                    continue
                hand.remove(matches[0])
                discard.append(matches[0])
        elif event.get("action_type") == "NetEndPlayerTurnAction":
            for card in hand:
                spec = catalog.require(card["card_id"])
                (exhaust if "ETHEREAL" in spec.keywords else discard).append(card)
            hand = []
            draw_cards(5)
            snapshot(int(event.get("turn_number", 0)) + 1)

    return {
        "schema_version": 1,
        "patch": catalog.patch,
        "character": catalog.character,
        "valid": not mismatches,
        "mismatches": mismatches,
        "turn_hands": snapshots,
        "final_shuffle_counter": shuffle_rng.counter,
        "discard_size": len(discard),
        "exhaust_size": len(exhaust),
    }
