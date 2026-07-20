"""Build strict, privacy-safe combat decisions from Workshop bridge transitions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from stsai.ids import canonical_character, canonical_entity


COMBAT_STATE_TYPES = frozenset({"monster", "elite", "boss"})
SUPPORTED_COMBAT_ACTIONS = frozenset(
    {
        "play_card",
        "use_potion",
        "end_turn",
        "combat_select_card",
        "combat_confirm_selection",
    }
)


@dataclass
class CombatDatasetSummary:
    files_seen: int = 0
    events_seen: int = 0
    decisions_written: int = 0
    skipped_unsuccessful: int = 0
    skipped_noncombat: int = 0
    skipped_illegal: int = 0
    skipped_wrong_source: int = 0
    splits: Counter[str] = field(default_factory=Counter)
    action_types: Counter[str] = field(default_factory=Counter)
    characters: Counter[str] = field(default_factory=Counter)
    sources: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "files_seen": self.files_seen,
            "events_seen": self.events_seen,
            "decisions_written": self.decisions_written,
            "skipped_unsuccessful": self.skipped_unsuccessful,
            "skipped_noncombat": self.skipped_noncombat,
            "skipped_illegal": self.skipped_illegal,
            "skipped_wrong_source": self.skipped_wrong_source,
            "splits": dict(self.splits),
            "action_types": dict(self.action_types),
            "characters": dict(self.characters),
            "sources": dict(self.sources),
        }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _objects(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def canonical_action(action: Mapping[str, Any]) -> dict[str, Any]:
    """Remove response-only fields and normalize the bridge action payload."""

    name = str(action.get("action", "")).strip()
    if name not in SUPPORTED_COMBAT_ACTIONS:
        raise ValueError(f"Unsupported combat action: {name or '<missing>'}")
    result: dict[str, Any] = {"action": name}
    if name in {"play_card", "combat_select_card"}:
        result["card_index"] = int(action["card_index"])
    elif name == "use_potion":
        result["slot"] = int(action["slot"])
    if name in {"play_card", "use_potion"} and action.get("target") is not None:
        result["target"] = str(action["target"])
    return result


def action_key(action: Mapping[str, Any]) -> str:
    return json.dumps(canonical_action(action), sort_keys=True, separators=(",", ":"))


def _enemy_targets(state: Mapping[str, Any]) -> list[str]:
    battle = _mapping(state.get("battle"))
    targets: list[str] = []
    for enemy in _objects(battle.get("enemies")):
        entity_id = enemy.get("entity_id")
        if entity_id and int(enemy.get("hp", 1) or 0) > 0:
            targets.append(str(entity_id))
    return targets


def _living_combat_id(creature: Mapping[str, Any]) -> str | None:
    if int(creature.get("hp", 1) or 0) <= 0 or creature.get("alive") is False:
        return None
    combat_id = creature.get("combat_id")
    return str(combat_id) if combat_id is not None else None


def _ally_targets(state: Mapping[str, Any], *, players_only: bool = False) -> list[str]:
    player = _mapping(state.get("player"))
    candidates: list[Mapping[str, Any]] = [player]
    if not players_only:
        candidates.extend(_objects(player.get("pets")))
    for teammate in _objects(state.get("players")):
        candidates.append(teammate)
        if not players_only:
            candidates.extend(_objects(teammate.get("pets")))
    result: list[str] = []
    for candidate in candidates:
        target = _living_combat_id(candidate)
        if target is not None and target not in result:
            result.append(target)
    return result


def _targeted_actions(
    base: dict[str, Any],
    target_type: Any,
    enemy_targets: Sequence[str],
    ally_targets: Sequence[str],
    player_targets: Sequence[str],
) -> list[dict[str, Any]]:
    normalized = str(target_type or "").lower()
    targets: Sequence[str] | None = None
    if normalized == "anyenemy":
        targets = enemy_targets
    elif normalized == "anyally":
        targets = ally_targets
    elif normalized == "anyplayer":
        targets = player_targets
    if targets is not None:
        return [{**base, "target": target} for target in targets]
    return [base]


def legal_combat_actions(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Enumerate exactly the supported actions advertised by an STS2MCP state."""

    state_type = str(state.get("state_type", ""))
    if state_type == "hand_select":
        selection = _mapping(state.get("hand_select"))
        actions = [
            {"action": "combat_select_card", "card_index": int(card["index"])}
            for card in _objects(selection.get("cards"))
            if card.get("index") is not None
        ]
        if selection.get("can_confirm") is True:
            actions.append({"action": "combat_confirm_selection"})
        return actions

    if state_type not in COMBAT_STATE_TYPES:
        return []
    battle = _mapping(state.get("battle"))
    if battle.get("is_play_phase") is not True:
        return []
    if battle.get("player_actions_disabled") is True:
        return []
    player = _mapping(state.get("player"))
    if int(player.get("hp", 1) or 0) <= 0:
        return []

    enemy_targets = _enemy_targets(state)
    ally_targets = _ally_targets(state)
    player_targets = _ally_targets(state, players_only=True)
    actions: list[dict[str, Any]] = []
    for card in _objects(player.get("hand")):
        if card.get("can_play") is not True or card.get("index") is None:
            continue
        base = {"action": "play_card", "card_index": int(card["index"])}
        actions.extend(
            _targeted_actions(
                base,
                card.get("target_type"),
                enemy_targets,
                ally_targets,
                player_targets,
            )
        )

    for potion in _objects(player.get("potions")):
        if potion.get("can_use_in_combat") is not True or potion.get("slot") is None:
            continue
        base = {"action": "use_potion", "slot": int(potion["slot"])}
        actions.extend(
            _targeted_actions(
                base,
                potion.get("target_type"),
                enemy_targets,
                ally_targets,
                player_targets,
            )
        )

    if battle.get("can_end_turn") is not False:
        actions.append({"action": "end_turn"})
    return actions


def _copy_fields(source: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {field: source[field] for field in fields if field in source}


def _normalize_statuses(value: Any) -> list[dict[str, Any]]:
    return [
        _copy_fields(status, ("id", "amount", "type"))
        for status in _objects(value)
    ]


def _normalize_card(card: Mapping[str, Any], *, in_hand: bool) -> dict[str, Any]:
    fields = ["index"] if in_hand else []
    fields.extend(
        (
            "id",
            "name",
            "type",
            "cost",
            "star_cost",
            "target_type",
            "can_play",
            "unplayable_reason",
            "is_upgraded",
        )
    )
    result = _copy_fields(card, fields)
    if result.get("id"):
        result["id"] = canonical_entity(result["id"], "CARD")
    return result


def _normalize_creature(creature: Mapping[str, Any]) -> dict[str, Any]:
    result = _copy_fields(
        creature,
        ("entity_id", "combat_id", "id", "name", "alive", "hp", "max_hp", "block"),
    )
    result["status"] = _normalize_statuses(creature.get("status"))
    if "intents" in creature:
        result["intents"] = [
            _copy_fields(intent, ("type", "damage", "hits", "total_damage", "amount"))
            for intent in _objects(creature.get("intents"))
        ]
    return result


def normalize_combat_observation(state: Mapping[str, Any]) -> dict[str, Any]:
    """Keep model-relevant state while excluding names/IDs of human players."""

    state_type = str(state.get("state_type", "unknown"))
    run = _copy_fields(_mapping(state.get("run")), ("act", "floor", "ascension"))
    source_player = _mapping(state.get("player"))
    player = _copy_fields(
        source_player,
        (
            "hp",
            "max_hp",
            "block",
            "gold",
            "energy",
            "max_energy",
            "stars",
            "draw_pile_count",
            "discard_pile_count",
            "exhaust_pile_count",
            "orb_slots",
            "orb_empty_slots",
            "entity_id",
            "combat_id",
        ),
    )
    player["character"] = canonical_character(
        source_player.get("character_id") or source_player.get("character")
    )
    player["hand"] = [
        _normalize_card(card, in_hand=True) for card in _objects(source_player.get("hand"))
    ]
    for pile_name in ("draw_pile", "discard_pile", "exhaust_pile"):
        player[pile_name] = [
            _normalize_card(card, in_hand=False)
            for card in _objects(source_player.get(pile_name))
        ]
    player["status"] = _normalize_statuses(source_player.get("status"))
    player["relics"] = [
        {
            **_copy_fields(relic, ("counter",)),
            "id": canonical_entity(relic.get("id"), "RELIC"),
        }
        for relic in _objects(source_player.get("relics"))
        if relic.get("id")
    ]
    player["potions"] = [
        {
            **_copy_fields(potion, ("slot", "can_use_in_combat", "target_type")),
            "id": canonical_entity(potion.get("id"), "POTION"),
        }
        for potion in _objects(source_player.get("potions"))
        if potion.get("id")
    ]
    player["orbs"] = [
        {
            **_copy_fields(orb, ("passive_val", "evoke_val")),
            "id": canonical_entity(orb.get("id"), "ORB"),
        }
        for orb in _objects(source_player.get("orbs"))
        if orb.get("id")
    ]
    player["pets"] = [
        _normalize_creature(pet) for pet in _objects(source_player.get("pets"))
    ]

    battle_source = _mapping(state.get("battle"))
    battle = _copy_fields(
        battle_source,
        (
            "round",
            "turn",
            "is_play_phase",
            "player_actions_disabled",
            "can_end_turn",
            "encounter_id",
        ),
    )
    battle["enemies"] = [
        _normalize_creature(enemy) for enemy in _objects(battle_source.get("enemies"))
    ]

    result: dict[str, Any] = {
        "state_type": state_type,
        "run": run,
        "player": player,
        "battle": battle,
    }
    result["players"] = []
    for source in _objects(state.get("players")):
        teammate = _copy_fields(
            source,
            (
                "entity_id",
                "combat_id",
                "is_local",
                "hp",
                "max_hp",
                "block",
                "is_alive",
            ),
        )
        teammate["character"] = canonical_character(source.get("character"))
        teammate["pets"] = [
            _normalize_creature(pet) for pet in _objects(source.get("pets"))
        ]
        result["players"].append(teammate)
    if state_type == "hand_select":
        selection = _mapping(state.get("hand_select"))
        result["hand_select"] = {
            "mode": selection.get("mode"),
            "cards": [
                _normalize_card(card, in_hand=True)
                for card in _objects(selection.get("cards"))
            ],
            "selected_indices": [
                int(card["index"])
                for card in _objects(selection.get("selected_cards"))
                if card.get("index") is not None
            ],
            "can_confirm": selection.get("can_confirm") is True,
        }
    return result


def headless_combat_trajectory_key(
    *, build_id: str, character: Any, seed: Any
) -> str:
    """Return the stable split key for one native headless character/seed run."""

    payload = f"{build_id}\x1f{canonical_character(character)}\x1f{seed}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def headless_combat_split(*, build_id: str, character: Any, seed: Any) -> str:
    """Assign one deterministic character/seed run to a dataset split."""

    return _split(
        headless_combat_trajectory_key(
            build_id=build_id, character=character, seed=seed
        )
    )


def _headless_statuses(value: Any) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for status in _objects(value):
        identifier = status.get("id") or status.get("name")
        normalized = _copy_fields(status, ("amount", "type"))
        if identifier:
            normalized["id"] = canonical_entity(identifier, "POWER")
        statuses.append(normalized)
    return statuses


def safe_potion_indexes(
    player: Mapping[str, Any], safe_potion_ids: Sequence[str] | frozenset[str]
) -> set[int]:
    """Raw potion indexes whose canonical id is in the safe allowlist.

    Raw headless potions carry "index" (normalization is what derives "slot"
    from it), and legal use_potion actions reference that same raw index.
    Both sides are canonicalized so an uncanonical raw id or allowlist entry
    cannot silently exclude a safe potion or admit an unsafe one.
    """

    safe = {canonical_entity(identifier, "POTION") for identifier in safe_potion_ids}
    allowed: set[int] = set()
    for fallback, potion in enumerate(_objects(player.get("potions"))):
        identifier = canonical_entity(potion.get("id") or potion.get("name"), "POTION")
        if identifier in safe:
            allowed.add(int(potion.get("index", fallback)))
    return allowed


def normalize_headless_combat_observation(
    observation: Mapping[str, Any]
) -> dict[str, Any]:
    """Map a native headless combat decision into the shared combat schema."""

    if str(observation.get("decision_type", "")) != "combat_play":
        raise ValueError("Headless observation is not a combat_play decision")
    state = _mapping(observation.get("decision_state"))
    context = _mapping(observation.get("context"))
    source_player = _mapping(observation.get("player"))
    hand = []
    for source in _objects(state.get("hand")):
        card = _copy_fields(
            source,
            (
                "index",
                "id",
                "name",
                "type",
                "cost",
                "star_cost",
                "target_type",
                "can_play",
                "upgraded",
            ),
        )
        if card.get("id"):
            card["id"] = canonical_entity(card["id"], "CARD")
        if "upgraded" in card:
            card["is_upgraded"] = bool(card.pop("upgraded"))
        hand.append(card)

    potions = []
    for slot, source in enumerate(_objects(source_player.get("potions"))):
        potion = _copy_fields(source, ("id", "name", "target_type"))
        identifier = potion.get("id") or potion.get("name")
        if identifier:
            potion["id"] = canonical_entity(identifier, "POTION")
        potion["slot"] = int(source.get("index", slot))
        potion["can_use_in_combat"] = True
        potions.append(potion)

    player = {
        **_copy_fields(source_player, ("hp", "max_hp", "block", "gold")),
        **_copy_fields(
            state,
            (
                "energy",
                "max_energy",
                "stars",
                "draw_pile_count",
                "discard_pile_count",
                "exhaust_pile_count",
            ),
        ),
        "character": canonical_character(observation.get("character")),
        "hand": hand,
        "draw_pile": [],
        "discard_pile": [],
        "exhaust_pile": [],
        "status": _headless_statuses(state.get("player_powers")),
        "relics": [
            {
                "id": canonical_entity(relic.get("id") or relic.get("name"), "RELIC"),
                **_copy_fields(relic, ("counter",)),
            }
            for relic in _objects(source_player.get("relics"))
            if relic.get("id") or relic.get("name")
        ],
        "potions": potions,
        "orbs": [
            _copy_fields(orb, ("id", "passive_val", "evoke_val"))
            for orb in _objects(state.get("orbs"))
        ],
        "pets": [
            _normalize_creature(pet) for pet in _objects(state.get("pets"))
        ],
    }
    enemies = []
    for position, source in enumerate(_objects(state.get("enemies"))):
        index = int(source.get("index", position))
        enemy = {
            "entity_id": f"enemy:{index}",
            "combat_id": str(index),
            **_copy_fields(source, ("id", "name", "hp", "max_hp", "block")),
            "status": _headless_statuses(source.get("powers")),
            "intents": [
                _copy_fields(intent, ("type", "damage", "hits", "amount"))
                for intent in _objects(source.get("intents"))
            ],
        }
        if enemy.get("id"):
            enemy["id"] = canonical_entity(enemy["id"], "MONSTER")
        enemies.append(enemy)

    room_type = str(context.get("room_type", "monster")).lower()
    if room_type not in COMBAT_STATE_TYPES:
        room_type = "monster"
    return {
        "state_type": room_type,
        "run": {
            "act": context.get("act"),
            "floor": context.get("floor"),
            "ascension": observation.get("ascension"),
        },
        "player": player,
        "battle": {
            "round": state.get("round"),
            "turn": "player",
            "is_play_phase": True,
            "enemies": enemies,
        },
        "players": [],
    }


def headless_action_to_combat(action: Mapping[str, Any]) -> dict[str, Any]:
    """Map one recorded native candidate to the shared combat action schema."""

    command = str(action.get("action", ""))
    arguments = _mapping(action.get("args"))
    if command == "end_turn":
        return {"action": "end_turn"}
    if command == "play_card":
        result: dict[str, Any] = {
            "action": "play_card",
            "card_index": int(arguments["card_index"]),
        }
    elif command == "use_potion":
        result = {
            "action": "use_potion",
            "slot": int(arguments["potion_index"]),
        }
    else:
        raise ValueError(f"Unsupported headless combat action: {command or '<missing>'}")
    if arguments.get("target_index") is not None:
        result["target"] = f"enemy:{int(arguments['target_index'])}"
    return result


def _iter_json_values(path: Path) -> Iterator[Mapping[str, Any]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if isinstance(value, Mapping):
                yield value
            elif isinstance(value, list):
                yield from (item for item in value if isinstance(item, Mapping))


def _file_key(path: Path, build_id: str) -> str:
    digest = hashlib.sha256()
    digest.update(build_id.encode("utf-8"))
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _split(key: str) -> str:
    bucket = int(key[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _trajectory_key(event: Mapping[str, Any], fallback: str, build_id: str) -> str:
    value = event.get("trajectory_id")
    if not isinstance(value, str) or not value.strip():
        return fallback
    return hashlib.sha256(f"{build_id}\x1f{value}".encode("utf-8")).hexdigest()


def build_combat_dataset(
    sources: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
    *,
    build_id: str,
) -> CombatDatasetSummary:
    """Convert accepted bridge transitions to atomic, action-masked NDJSON."""

    if not build_id.strip():
        raise ValueError("build_id is required")
    paths = [Path(source) for source in sources]
    if not paths:
        raise ValueError("At least one combat trajectory is required")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    summary = CombatDatasetSummary()
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for path in paths:
                summary.files_seen += 1
                fallback_key = _file_key(path, build_id)
                for event in _iter_json_values(path):
                    summary.events_seen += 1
                    trajectory_key = _trajectory_key(event, fallback_key, build_id)
                    split = _split(trajectory_key)
                    is_human_ui = (
                        event.get("event_type") == "human_combat_action"
                        and event.get("source") == "game_ui"
                    )
                    response = _mapping(event.get("bridge_response"))
                    if not is_human_ui and (
                        response.get("status") != "ok"
                        or event.get("state_changed") is not True
                    ):
                        summary.skipped_unsuccessful += 1
                        continue
                    observation = _mapping(event.get("observation"))
                    if str(observation.get("state_type", "")) not in (
                        COMBAT_STATE_TYPES | {"hand_select"}
                    ):
                        summary.skipped_noncombat += 1
                        continue
                    legal_actions = legal_combat_actions(observation)
                    try:
                        chosen = canonical_action(_mapping(event.get("action")))
                    except (KeyError, TypeError, ValueError):
                        summary.skipped_illegal += 1
                        continue
                    legal_keys = {action_key(action) for action in legal_actions}
                    if action_key(chosen) not in legal_keys:
                        summary.skipped_illegal += 1
                        continue

                    normalized = normalize_combat_observation(observation)
                    next_state = _mapping(event.get("next_observation"))
                    record = {
                        "schema_version": 1,
                        "decision_type": "combat_action",
                        "split": split,
                        "trajectory_key": trajectory_key,
                        "patch": {"build_id": build_id, "bridge": "STS2MCP"},
                        "observation": normalized,
                        "legal_actions": legal_actions,
                        "chosen_action": chosen,
                        "next_observation": (
                            normalize_combat_observation(next_state) if next_state else None
                        ),
                        "terminal": event.get("terminal") is True,
                    }
                    handle.write(
                        json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
                    )
                    summary.decisions_written += 1
                    summary.splits[split] += 1
                    summary.action_types[chosen["action"]] += 1
                    summary.characters[normalized["player"]["character"]] += 1
                    summary.sources["human_ui" if is_human_ui else "bridge"] += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return summary


def _discover_headless_trajectories(
    sources: Sequence[str | os.PathLike[str]],
) -> list[Path]:
    paths: list[Path] = []
    for raw in sources:
        source = Path(raw)
        if source.is_file():
            paths.append(source)
        elif source.is_dir():
            paths.extend(
                path
                for path in source.rglob("*")
                if path.is_file() and path.suffix.lower() in {".jsonl", ".ndjson"}
            )
        else:
            raise FileNotFoundError(source)
    return sorted(set(paths))


def build_headless_combat_dataset(
    sources: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
    *,
    build_id: str,
    teacher_source: str = "heuristic",
) -> CombatDatasetSummary:
    """Convert native headless teacher trajectories to the shared combat schema."""

    if not build_id.strip():
        raise ValueError("build_id is required")
    paths = _discover_headless_trajectories(sources)
    if not paths:
        raise ValueError("No native headless trajectory files were found")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    summary = CombatDatasetSummary()
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for path in paths:
                summary.files_seen += 1
                episode: Mapping[str, Any] = {}
                for event in _iter_json_values(path):
                    summary.events_seen += 1
                    if event.get("event_type") == "episode_start":
                        episode = event
                        continue
                    if event.get("event_type") != "transition":
                        continue
                    observation = _mapping(event.get("observation"))
                    if observation.get("decision_type") != "combat_play":
                        summary.skipped_noncombat += 1
                        continue
                    policy = _mapping(_mapping(event.get("info")).get("policy"))
                    source = str(policy.get("source", "unknown"))
                    if teacher_source and source != teacher_source:
                        summary.skipped_wrong_source += 1
                        continue
                    try:
                        legal_actions = [
                            headless_action_to_combat(action)
                            for action in _objects(event.get("legal_actions"))
                        ]
                        chosen = headless_action_to_combat(_mapping(event.get("action")))
                    except (KeyError, TypeError, ValueError):
                        summary.skipped_illegal += 1
                        continue
                    legal_keys = {action_key(action) for action in legal_actions}
                    if not legal_actions or action_key(chosen) not in legal_keys:
                        summary.skipped_illegal += 1
                        continue

                    character = episode.get("character") or observation.get("character")
                    seed = episode.get("seed") or observation.get("seed")
                    trajectory_key = headless_combat_trajectory_key(
                        build_id=build_id, character=character, seed=seed
                    )
                    normalized = normalize_headless_combat_observation(observation)
                    next_state = _mapping(event.get("next_observation"))
                    record = {
                        "schema_version": 1,
                        "decision_type": "combat_action",
                        "split": _split(trajectory_key),
                        "trajectory_key": trajectory_key,
                        "patch": {
                            "build_id": build_id,
                            "runtime": "sts2-cli",
                            "label_source": f"{teacher_source}_distillation",
                        },
                        "observation": normalized,
                        "legal_actions": legal_actions,
                        "chosen_action": chosen,
                        "next_observation": (
                            normalize_headless_combat_observation(next_state)
                            if next_state.get("decision_type") == "combat_play"
                            else None
                        ),
                        "terminal": event.get("terminated") is True,
                    }
                    handle.write(
                        json.dumps(record, separators=(",", ":"), ensure_ascii=False)
                        + "\n"
                    )
                    summary.decisions_written += 1
                    summary.splits[record["split"]] += 1
                    summary.action_types[chosen["action"]] += 1
                    summary.characters[normalized["player"]["character"]] += 1
                    summary.sources[f"headless_{source}"] += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return summary
