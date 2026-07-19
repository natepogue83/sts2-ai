"""Character-conditioned masked policy for Workshop combat decisions."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from stsai.combat import action_key, legal_combat_actions, normalize_combat_observation
from stsai.policy import FeatureHasher, MaskedMLPPolicy, PreparedDecision, read_dataset


def _atomic_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class CombatFeatureHasher(FeatureHasher):
    """Encode combat state and one candidate action into a fixed-width vector."""

    NUMERIC_FEATURES = 16

    @staticmethod
    def _items(value: Any) -> list[Mapping[str, Any]]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, Mapping)]

    @staticmethod
    def _by_int(items: Iterable[Mapping[str, Any]], field: str, value: Any) -> Mapping[str, Any]:
        try:
            target = int(value)
        except (TypeError, ValueError):
            return {}
        return next(
            (item for item in items if item.get(field) is not None and int(item[field]) == target),
            {},
        )

    def encode(self, record: Mapping[str, Any], action: Mapping[str, Any]) -> np.ndarray:
        observation = record.get("observation") or {}
        run = observation.get("run") or {}
        player = observation.get("player") or {}
        battle = observation.get("battle") or {}
        enemies = self._items(battle.get("enemies"))
        allies = [player]
        allies.extend(self._items(player.get("pets")))
        for teammate in self._items(observation.get("players")):
            allies.append(teammate)
            allies.extend(self._items(teammate.get("pets")))
        hand = self._items(player.get("hand"))
        max_hp = max(float(player.get("max_hp") or 0), 1.0)
        max_energy = max(float(player.get("max_energy") or 0), 1.0)
        alive_enemies = [enemy for enemy in enemies if float(enemy.get("hp") or 0) > 0]
        possible_targets = alive_enemies + allies
        target = next(
            (
                creature
                for creature in possible_targets
                if str(creature.get("entity_id")) == str(action.get("target"))
                or str(creature.get("combat_id")) == str(action.get("target"))
            ),
            {},
        )
        target_max_hp = max(float(target.get("max_hp") or 0), 1.0)
        vector = np.zeros(self.dimension, dtype=np.float32)
        vector[: self.NUMERIC_FEATURES] = (
            float(player.get("hp") or 0) / max_hp,
            min(float(player.get("block") or 0) / max_hp, 2.0),
            float(player.get("energy") or 0) / max_energy,
            min(float(player.get("stars") or 0) / 10.0, 2.0),
            min(float(run.get("act") or 0) / 4.0, 1.5),
            min(float(run.get("floor") or 0) / 60.0, 1.5),
            min(float(run.get("ascension") or 0) / 10.0, 2.0),
            min(float(battle.get("round") or 0) / 20.0, 2.0),
            min(len(alive_enemies) / 5.0, 2.0),
            min(sum(float(enemy.get("hp") or 0) for enemy in alive_enemies) / 300.0, 3.0),
            min(sum(float(enemy.get("block") or 0) for enemy in alive_enemies) / 100.0, 3.0),
            min(len(hand) / 10.0, 2.0),
            min(float(player.get("draw_pile_count") or 0) / 30.0, 2.0),
            min(float(player.get("discard_pile_count") or 0) / 30.0, 2.0),
            min(float(player.get("exhaust_pile_count") or 0) / 30.0, 2.0),
            float(target.get("hp") or 0) / target_max_hp if target else 0.0,
        )

        character = str(player.get("character", "unknown"))
        build_id = str((record.get("patch") or {}).get("build_id", "unknown"))
        action_name = str(action.get("action", "unknown"))
        self._add(vector, f"character={character}", 2.0)
        self._add(vector, f"build={build_id}")
        self._add(vector, f"state={observation.get('state_type')}")
        self._add(vector, f"action={action_name}", 2.0)
        self._add(vector, f"character={character}|action={action_name}", 2.0)

        candidate: Mapping[str, Any] = {}
        if action_name == "play_card":
            candidate = self._by_int(hand, "index", action.get("card_index"))
        elif action_name == "combat_select_card":
            candidate = self._by_int(
                self._items((observation.get("hand_select") or {}).get("cards")),
                "index",
                action.get("card_index"),
            )
        elif action_name == "use_potion":
            candidate = self._by_int(
                self._items(player.get("potions")), "slot", action.get("slot")
            )
        candidate_id = str(candidate.get("id", "none"))
        self._add(vector, f"candidate={candidate_id}", 2.0)
        self._add(vector, f"character={character}|candidate={candidate_id}", 2.0)
        self._add(vector, f"candidate_type={candidate.get('type')}")
        self._add(vector, f"candidate_cost={candidate.get('cost')}")
        self._add(vector, f"candidate_star_cost={candidate.get('star_cost')}")
        self._add(vector, f"candidate_upgraded={candidate.get('is_upgraded')}")

        if target:
            target_id = str(target.get("entity_id", target.get("id", "unknown")))
            self._add(vector, f"target={target_id}", 2.0)
            self._add(vector, f"target_side={'enemy' if target in enemies else 'ally'}")
            self._add(vector, f"candidate={candidate_id}|target={target_id}")
            for status in self._items(target.get("status")):
                self._add(
                    vector,
                    f"target_status={status.get('id')}",
                    math.sqrt(max(float(status.get("amount") or 1), 1.0)),
                )

        card_counts: dict[str, int] = {}
        for card in hand:
            identifier = str(card.get("id", "unknown"))
            card_counts[identifier] = card_counts.get(identifier, 0) + 1
        for identifier, count in card_counts.items():
            self._add(vector, f"hand={identifier}", math.sqrt(count))
        for relic in self._items(player.get("relics")):
            self._add(vector, f"relic={relic.get('id')}")
        for status in self._items(player.get("status")):
            self._add(vector, f"player_status={status.get('id')}")
        for orb in self._items(player.get("orbs")):
            self._add(vector, f"orb={orb.get('id')}")
        for pet in self._items(player.get("pets")):
            self._add(vector, f"pet={pet.get('id')}")
        for enemy in alive_enemies:
            self._add(vector, f"enemy={enemy.get('entity_id', enemy.get('id'))}")
            for intent in self._items(enemy.get("intents")):
                self._add(vector, f"intent={intent.get('type')}")
        return vector


def prepare_combat_records(
    records: Iterable[Mapping[str, Any]], hasher: CombatFeatureHasher
) -> list[PreparedDecision]:
    prepared: list[PreparedDecision] = []
    for record in records:
        actions = [
            action
            for action in (record.get("legal_actions") or [])
            if isinstance(action, Mapping)
        ]
        if not actions or not isinstance(record.get("chosen_action"), Mapping):
            continue
        keys = [action_key(action) for action in actions]
        chosen_key = action_key(record["chosen_action"])
        if chosen_key not in keys:
            continue
        features = np.stack([hasher.encode(record, action) for action in actions])
        observation = record.get("observation") or {}
        player = observation.get("player") or {}
        prepared.append(
            PreparedDecision(
                features=features,
                chosen_index=keys.index(chosen_key),
                patch=str((record.get("patch") or {}).get("build_id", "unknown")),
                decision_type=str(record["chosen_action"].get("action", "unknown")),
                character=str(player.get("character", "unknown")),
            )
        )
    return prepared


def train_combat_policy(
    dataset: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    dimension: int = 1024,
    hidden: int = 128,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 7,
    metrics_output: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    hasher = CombatFeatureHasher(dimension)
    train = prepare_combat_records(read_dataset(dataset, "train"), hasher)
    validation = prepare_combat_records(read_dataset(dataset, "validation"), hasher)
    model = MaskedMLPPolicy(dimension, hidden, seed)
    history = model.fit(
        train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )
    model.save(output)
    result = {
        "model": str(Path(output)),
        "configuration": {
            "dimension": dimension,
            "hidden": hidden,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
        },
        "train_final": history[-1].as_dict(),
        "validation": model.evaluate(validation).as_dict(),
    }
    if metrics_output is not None:
        _atomic_json(metrics_output, result)
    return result


def evaluate_combat_policy(
    dataset: str | os.PathLike[str],
    model_path: str | os.PathLike[str],
    split: str = "test",
    output: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    model = MaskedMLPPolicy.load(model_path)
    decisions = prepare_combat_records(
        read_dataset(dataset, split), CombatFeatureHasher(model.dimension)
    )
    by_character = {
        character: model.evaluate(
            [decision for decision in decisions if decision.character == character]
        ).as_dict()
        for character in sorted({decision.character for decision in decisions})
    }
    by_action = {
        action: model.evaluate(
            [decision for decision in decisions if decision.decision_type == action]
        ).as_dict()
        for action in sorted({decision.decision_type for decision in decisions})
    }
    result = {
        "split": split,
        "overall": model.evaluate(decisions).as_dict(),
        "by_character": by_character,
        "by_action": by_action,
    }
    if output is not None:
        _atomic_json(output, result)
    return result


class CombatPolicyAgent:
    """Score only legal bridge actions and return one non-mutating recommendation."""

    def __init__(self, model: MaskedMLPPolicy, *, build_id: str):
        self.model = model
        self.build_id = build_id
        self.hasher = CombatFeatureHasher(model.dimension)

    @classmethod
    def from_path(
        cls, path: str | os.PathLike[str], *, build_id: str
    ) -> "CombatPolicyAgent":
        return cls(MaskedMLPPolicy.load(path), build_id=build_id)

    def recommend(self, state: Mapping[str, Any]) -> dict[str, Any]:
        actions = legal_combat_actions(state)
        if not actions:
            raise ValueError("The current bridge state has no supported legal combat actions")
        observation = normalize_combat_observation(state)
        record = {
            "decision_type": "combat_action",
            "patch": {"build_id": self.build_id},
            "observation": observation,
        }
        features = np.stack([self.hasher.encode(record, action) for action in actions])
        scores = self.model.scores(features)
        probabilities = np.exp(scores - np.max(scores))
        probabilities /= probabilities.sum()
        order = np.argsort(-scores)
        best = int(order[0])
        return {
            "character": observation["player"]["character"],
            "action": actions[best],
            "confidence": float(probabilities[best]),
            "ranked_actions": [
                {"action": actions[int(index)], "probability": float(probabilities[int(index)])}
                for index in order[:5]
            ],
        }
