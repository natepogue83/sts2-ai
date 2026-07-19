"""Character-conditioned live macro decisions for the STS2MCP bridge."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from stsai.dataset import SKIP
from stsai.ids import canonical_character, canonical_entity
from stsai.policy import FeatureHasher, MaskedMLPPolicy
from stsai.runs import default_run_history_root


class UnsupportedLiveState(ValueError):
    """Raised when the trained macro policy cannot safely act on a screen."""


def detect_game_build(game_dir: str | os.PathLike[str] | None = None) -> str:
    candidates: list[Path] = []
    if game_dir:
        candidates.append(Path(game_dir))
    if os.environ.get("STS2_GAME_DIR"):
        candidates.append(Path(os.environ["STS2_GAME_DIR"]))
    if os.name == "nt":
        candidates.append(
            Path(r"C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2")
        )
    for candidate in candidates:
        release = candidate / "release_info.json"
        try:
            value = json.loads(release.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if value.get("version"):
            return str(value["version"])
    return "unknown"


def _id_from_saved(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and value.get("id"):
        return str(value["id"])
    return None


def load_current_deck(
    character: str, save_root: str | os.PathLike[str] | None = None
) -> list[str]:
    """Read the newest matching current-run save without returning account paths/ids."""

    root = Path(save_root) if save_root else default_run_history_root()
    if not root.exists():
        return []
    files = sorted(
        root.rglob("current_run.save"),
        key=lambda value: value.stat().st_mtime,
        reverse=True,
    )
    expected = canonical_character(character)
    for path in files:
        try:
            run = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        for player in run.get("players") or []:
            if not isinstance(player, dict):
                continue
            if canonical_character(player.get("character_id")) != expected:
                continue
            cards = [
                identifier
                for card in (player.get("deck") or [])
                if (identifier := _id_from_saved(card))
            ]
            return sorted(canonical_entity(card, "CARD") for card in cards)
    return []


class MacroPolicyAgent:
    """Recommend only action types supported by valid run-history supervision."""

    def __init__(
        self,
        model: MaskedMLPPolicy,
        *,
        build_id: str | None = None,
        save_root: str | os.PathLike[str] | None = None,
    ):
        self.model = model
        self.hasher = FeatureHasher(model.dimension)
        self.build_id = build_id or detect_game_build()
        self.save_root = save_root

    @classmethod
    def from_path(
        cls,
        model_path: str | os.PathLike[str],
        *,
        build_id: str | None = None,
        save_root: str | os.PathLike[str] | None = None,
    ) -> "MacroPolicyAgent":
        return cls(
            MaskedMLPPolicy.load(model_path), build_id=build_id, save_root=save_root
        )

    def _observation(self, state: Mapping[str, Any]) -> dict[str, Any]:
        player = state.get("player") or {}
        run = state.get("run") or {}
        character = canonical_character(player.get("character"))
        relics = [
            canonical_entity(item.get("id"), "RELIC")
            for item in (player.get("relics") or [])
            if isinstance(item, dict) and item.get("id")
        ]
        room_id = None
        if isinstance(state.get("event"), dict):
            room_id = state["event"].get("event_id")
        return {
            "character": character,
            "ascension": int(run.get("ascension") or 0),
            "act": int(run.get("act") or 0),
            "floor": int(run.get("floor") or 0),
            "current_hp": int(player.get("hp") or 0),
            "max_hp": int(player.get("max_hp") or 0),
            "current_gold": int(player.get("gold") or 0),
            "deck": load_current_deck(character, self.save_root),
            "relics": sorted(relics),
            "map_point_type": state.get("state_type"),
            "room_id": room_id,
            "monsters": [],
            "turns_taken": None,
        }

    @staticmethod
    def _ancient_id(option: Mapping[str, Any]) -> str:
        # Ancient options grant relics whose display names map to the TextKey
        # stored in run history. Fall back to option title for unusual options.
        value = option.get("relic_name") or option.get("title")
        return canonical_entity(value, "ANCIENT")

    def _options(
        self, state: Mapping[str, Any]
    ) -> tuple[str, list[tuple[str, int | None]], bool]:
        state_type = state.get("state_type")
        if state_type == "card_reward":
            payload = state.get("card_reward") or {}
            options = [
                (canonical_entity(item.get("id"), "CARD"), int(item["index"]))
                for item in (payload.get("cards") or [])
                if isinstance(item, dict) and item.get("id") is not None
            ]
            return "card_reward", options, bool(payload.get("can_skip", False))
        if state_type == "relic_select":
            payload = state.get("relic_select") or {}
            options = [
                (canonical_entity(item.get("id"), "RELIC"), int(item["index"]))
                for item in (payload.get("relics") or [])
                if isinstance(item, dict) and item.get("id") is not None
            ]
            return "relic_reward", options, bool(payload.get("can_skip", False))
        if state_type == "event" and bool((state.get("event") or {}).get("is_ancient")):
            event = state.get("event") or {}
            if bool(event.get("in_dialogue", False)):
                raise UnsupportedLiveState("Ancient dialogue must be advanced before choosing")
            available = [
                item
                for item in (event.get("options") or [])
                if isinstance(item, dict)
                and not bool(item.get("is_locked", False))
                and not bool(item.get("is_proceed", False))
            ]
            options = [(self._ancient_id(item), int(item["index"])) for item in available]
            return "ancient", options, False
        raise UnsupportedLiveState(
            f"The learned macro policy does not support state_type={state_type!r}"
        )

    @staticmethod
    def _action(decision_type: str, chosen: str, index: int | None) -> dict[str, Any]:
        if decision_type == "card_reward":
            return (
                {"action": "skip_card_reward"}
                if chosen == SKIP
                else {"action": "select_card_reward", "card_index": index}
            )
        if decision_type == "relic_reward":
            return (
                {"action": "skip_relic_selection"}
                if chosen == SKIP
                else {"action": "select_relic", "index": index}
            )
        if decision_type == "ancient":
            return {"action": "choose_event_option", "index": index}
        raise UnsupportedLiveState(f"No bridge action mapping for {decision_type}")

    def recommend(self, state: Mapping[str, Any]) -> dict[str, Any]:
        decision_type, indexed_options, can_skip = self._options(state)
        if not indexed_options:
            raise UnsupportedLiveState("No legal learned-policy options are visible")
        option_ids = [identifier for identifier, _ in indexed_options]
        if can_skip:
            option_ids.append(SKIP)
        record = {
            "decision_type": decision_type,
            "patch": {"build_id": self.build_id},
            "observation": self._observation(state),
            "legal_options": option_ids,
        }
        features = np.stack([self.hasher.encode(record, option) for option in option_ids])
        scores = self.model.scores(features)
        shifted = scores - np.max(scores)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        chosen_index = int(np.argmax(scores))
        chosen = option_ids[chosen_index]
        bridge_index = None
        if chosen != SKIP:
            bridge_index = dict(indexed_options)[chosen]
        action = self._action(decision_type, chosen, bridge_index)
        ranked = sorted(
            (
                {
                    "option": option,
                    "score": round(float(score), 6),
                    "probability": round(float(probability), 6),
                }
                for option, score, probability in zip(
                    option_ids, scores, probabilities, strict=True
                )
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        return {
            "decision_type": decision_type,
            "character": record["observation"]["character"],
            "build_id": self.build_id,
            "chosen": chosen,
            "action": action,
            "ranked_options": ranked,
            "deck_cards_observed": len(record["observation"]["deck"]),
        }
