"""Leakage-resistant behavior-cloning datasets derived from run histories."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from stsai.runs import RunFilter, iter_map_points, player_stats, read_runs, run_key


SKIP = "SKIP"


@dataclass
class DatasetBuildSummary:
    runs_seen: int = 0
    runs_used: int = 0
    duplicate_runs: int = 0
    decisions_written: int = 0
    skipped_ambiguous: int = 0
    chosen_skip: int = 0
    deck_reversal_failures: int = 0
    runs_with_reversal_failures: int = 0
    splits: Counter[str] = field(default_factory=Counter)
    builds: Counter[str] = field(default_factory=Counter)
    decision_types: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "runs_seen": self.runs_seen,
            "runs_used": self.runs_used,
            "duplicate_runs": self.duplicate_runs,
            "decisions_written": self.decisions_written,
            "skipped_ambiguous": self.skipped_ambiguous,
            "chosen_skip": self.chosen_skip,
            "deck_reversal_failures": self.deck_reversal_failures,
            "runs_with_reversal_failures": self.runs_with_reversal_failures,
            "splits": dict(self.splits),
            "builds": dict(self.builds),
            "decision_types": dict(self.decision_types),
        }


def _card_id(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        nested = value.get("card")
        if nested is not None:
            return _card_id(nested)
        identifier = value.get("id")
        return str(identifier) if identifier else None
    return None


def _remove_one(cards: list[str], identifier: str | None) -> bool:
    """Remove one instance; report failure instead of hiding it.

    A failed removal means a future card could not be reversed out of an
    earlier deck snapshot — that snapshot now leaks post-decision information.
    """

    if not identifier:
        return False
    try:
        cards.remove(identifier)
        return True
    except ValueError:
        # Schema/patch variations can omit a final-deck instance (e.g. the
        # final deck holds the upgraded id while the delta recorded the base
        # id). A partial deck is still useful and failing closed would discard
        # the whole run, but callers must count these so leakage is measurable.
        return False


def _reverse_deck_snapshots(
    run: Mapping[str, Any], player: Mapping[str, Any]
) -> tuple[dict[int, list[str]], int]:
    """Reconstruct the deck before each floor from the final deck and deltas.

    Returns the snapshots plus the count of delta entries that failed to
    reverse; every failure leaves a future card in earlier floors' snapshots.
    """

    deck = [identifier for card in (player.get("deck") or []) if (identifier := _card_id(card))]
    points = list(iter_map_points(run))
    before: dict[int, list[str]] = {}
    failures = 0
    for _, floor, point in reversed(points):
        stats = player_stats(point, player)

        # Reverse additions first, then transformations, then removals. We only
        # track IDs here; upgrade/enchantment state gets its own encoder later.
        for gained in stats.get("cards_gained") or []:
            failures += not _remove_one(deck, _card_id(gained))
        for transformed in reversed(stats.get("cards_transformed") or []):
            if not isinstance(transformed, dict):
                continue
            failures += not _remove_one(deck, _card_id(transformed.get("final_card")))
            original = _card_id(transformed.get("original_card"))
            if original:
                deck.append(original)
        for removed in stats.get("cards_removed") or []:
            identifier = _card_id(removed)
            if identifier:
                deck.append(identifier)
        before[floor] = sorted(deck)
    return before, failures


def _relics_before_floor(player: Mapping[str, Any], floor: int) -> list[str]:
    relics: list[str] = []
    for relic in player.get("relics") or []:
        if isinstance(relic, str):
            relics.append(relic)
            continue
        if not isinstance(relic, dict) or not relic.get("id"):
            continue
        added = relic.get("floor_added_to_deck")
        if added is None or int(added) < floor:
            relics.append(str(relic["id"]))
    return sorted(relics)


def split_for_run(run: Mapping[str, Any]) -> str:
    """Assign a split by seed, keeping repeated deterministic seeds together."""

    seed = str(run.get("seed") or run_key(run))
    bucket = int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _room_context(point: Mapping[str, Any]) -> tuple[str | None, list[str], int | None]:
    rooms = [room for room in (point.get("rooms") or []) if isinstance(room, dict)]
    if not rooms:
        return None, [], None
    room = rooms[0]
    model_id = str(room["model_id"]) if room.get("model_id") else None
    monsters = [str(value) for value in (room.get("monster_ids") or [])]
    turns = int(room["turns_taken"]) if room.get("turns_taken") is not None else None
    return model_id, monsters, turns


def _base_observation(
    run: Mapping[str, Any],
    player: Mapping[str, Any],
    stats: Mapping[str, Any],
    point: Mapping[str, Any],
    snapshots: Mapping[int, list[str]],
    act: int,
    floor: int,
) -> dict[str, Any]:
    room_id, monsters, turns = _room_context(point)
    return {
        "character": str(player.get("character", "unknown")),
        "ascension": int(run.get("ascension") or 0),
        "act": act,
        "floor": floor,
        "current_hp": int(stats.get("current_hp") or 0),
        "max_hp": int(stats.get("max_hp") or 0),
        "current_gold": int(stats.get("current_gold") or 0),
        "deck": snapshots.get(floor, []),
        "relics": _relics_before_floor(player, floor),
        "map_point_type": point.get("map_point_type"),
        "room_id": room_id,
        "monsters": monsters,
        "turns_taken": turns,
    }


def _record(
    run: Mapping[str, Any],
    decision_type: str,
    observation: Mapping[str, Any],
    options: Sequence[str],
    chosen: str,
    floors_reached: int,
    *,
    can_skip: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "decision_type": decision_type,
        "split": split_for_run(run),
        "run_key": run_key(run),
        "patch": {
            "build_id": str(run.get("build_id", "unknown")),
            "run_schema_version": run.get("schema_version"),
        },
        "observation": dict(observation),
        "legal_options": list(options) + ([SKIP] if can_skip else []),
        "chosen": chosen,
        "label_metadata": {
            "run_won": bool(run.get("win", False)),
            "floors_reached": floors_reached,
        },
    }


def _offered_choices(
    stats: Mapping[str, Any], decision_type: str
) -> tuple[list[str], str] | None:
    if decision_type == "card_reward":
        raw = stats.get("card_choices") or []
        identifier_key = "card"
        picked_key = "was_picked"
        minimum_options = 1
    elif decision_type == "relic_reward":
        raw = stats.get("relic_choices") or []
        identifier_key = "choice"
        picked_key = "was_picked"
        minimum_options = 2
    elif decision_type == "potion_reward":
        raw = stats.get("potion_choices") or []
        identifier_key = "choice"
        picked_key = "was_picked"
        minimum_options = 2
    elif decision_type == "ancient":
        raw = stats.get("ancient_choice") or []
        identifier_key = "TextKey"
        picked_key = "was_chosen"
        minimum_options = 2
    else:
        raise ValueError(f"Unsupported decision type: {decision_type}")

    if not isinstance(raw, list) or len(raw) < minimum_options:
        return None
    choices: list[tuple[str, bool]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        value = item.get(identifier_key)
        identifier = _card_id(value)
        if decision_type == "ancient" and identifier:
            identifier = f"ANCIENT.{identifier}"
        if identifier:
            choices.append((identifier, bool(item.get(picked_key, False))))
    if len(choices) < minimum_options:
        return None
    picked = [identifier for identifier, was_picked in choices if was_picked]
    if len(picked) > 1:
        return None
    if decision_type == "ancient" and not picked:
        return None
    return [identifier for identifier, _ in choices], picked[0] if picked else SKIP


SUPPORTED_DECISION_TYPES = frozenset(
    {"card_reward", "relic_reward", "potion_reward", "ancient"}
)


def decision_records(
    run: Mapping[str, Any],
    decision_types: frozenset[str] = SUPPORTED_DECISION_TYPES,
    *,
    summary: DatasetBuildSummary | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield decisions whose chosen and rejected alternatives are both recorded."""

    unknown = decision_types - SUPPORTED_DECISION_TYPES
    if unknown:
        raise ValueError(f"Unsupported decision types: {', '.join(sorted(unknown))}")

    players = run.get("players") or []
    if not isinstance(players, list) or len(players) != 1 or not isinstance(players[0], dict):
        return
    player = players[0]
    snapshots, reversal_failures = _reverse_deck_snapshots(run, player)
    if summary is not None and reversal_failures:
        summary.deck_reversal_failures += reversal_failures
        summary.runs_with_reversal_failures += 1
    floors_reached = sum(1 for _ in iter_map_points(run))

    for act, floor, point in iter_map_points(run):
        stats = player_stats(point, player)
        observation = _base_observation(run, player, stats, point, snapshots, act, floor)
        for decision_type in sorted(decision_types):
            offered = _offered_choices(stats, decision_type)
            if offered is None:
                continue
            options, chosen = offered
            yield _record(
                run,
                decision_type,
                observation,
                options,
                chosen,
                floors_reached,
                can_skip=decision_type != "ancient",
            )


def card_reward_records(run: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield anonymized card-reward decisions for a single-player run."""

    yield from decision_records(run, frozenset({"card_reward"}))


def build_card_dataset(
    sources: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
    run_filter: RunFilter | None = None,
) -> DatasetBuildSummary:
    """Write card-reward records as NDJSON using an atomic final rename."""

    return build_decision_dataset(
        sources, output, run_filter, decision_types=frozenset({"card_reward"})
    )


def build_decision_dataset(
    sources: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
    run_filter: RunFilter | None = None,
    *,
    decision_types: frozenset[str] = SUPPORTED_DECISION_TYPES,
) -> DatasetBuildSummary:
    """Write supported macro decisions as NDJSON using an atomic final rename."""

    run_filter = run_filter or RunFilter()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    summary = DatasetBuildSummary()
    seen_runs: set[str] = set()
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for source in sources:
                for run in read_runs(source):
                    summary.runs_seen += 1
                    if not run_filter.accepts(run):
                        continue
                    fingerprint = run_key(run)
                    if fingerprint in seen_runs:
                        summary.duplicate_runs += 1
                        continue
                    seen_runs.add(fingerprint)
                    records = list(
                        decision_records(run, decision_types, summary=summary)
                    )
                    if not records:
                        continue
                    summary.runs_used += 1
                    build_id = str(run.get("build_id", "unknown"))
                    summary.builds[build_id] += 1
                    for record in records:
                        json.dump(record, handle, separators=(",", ":"), ensure_ascii=False)
                        handle.write("\n")
                        summary.decisions_written += 1
                        summary.splits[record["split"]] += 1
                        summary.decision_types[record["decision_type"]] += 1
                        summary.chosen_skip += int(record["chosen"] == SKIP)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return summary
