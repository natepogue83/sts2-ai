"""Calibrated, explainable public-state values from first-party headless runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from stsai.search import PolicyValueEstimate


VALUE_CHARACTERS = (
    "CHARACTER.IRONCLAD",
    "CHARACTER.SILENT",
    "CHARACTER.DEFECT",
    "CHARACTER.NECROBINDER",
    "CHARACTER.REGENT",
)

VALUE_FEATURE_NAMES = tuple(
    [f"character:{character.rsplit('.', 1)[-1].lower()}" for character in VALUE_CHARACTERS]
    + [
        "ascension_fraction",
        "act_fraction",
        "floor_fraction",
        "player_hp_fraction",
        "player_block_fraction",
        "gold_fraction",
        "deck_size_fraction",
        "relic_count_fraction",
        "potion_count_fraction",
        "combat_round_fraction",
        "energy_fraction",
        "hand_size_fraction",
        "playable_card_fraction",
        "enemy_count_fraction",
        "enemy_hp_fraction",
        "enemy_block_fraction",
        "incoming_damage_fraction",
        "player_power_count_fraction",
        "orb_count_fraction",
        "stars_fraction",
        "pet_hp_fraction",
    ]
)

_INTERACTION_BASE_FEATURES = (
    "floor_fraction",
    "player_hp_fraction",
    "player_block_fraction",
    "deck_size_fraction",
    "relic_count_fraction",
    "potion_count_fraction",
    "combat_round_fraction",
    "energy_fraction",
    "enemy_hp_fraction",
    "incoming_damage_fraction",
    "orb_count_fraction",
    "stars_fraction",
    "pet_hp_fraction",
)

VALUE_INTERACTION_FEATURE_NAMES = VALUE_FEATURE_NAMES + tuple(
    f"interaction:{character.rsplit('.', 1)[-1].lower()}:{feature}"
    for character in VALUE_CHARACTERS
    for feature in _INTERACTION_BASE_FEATURES
)

_NONLINEAR_HINGES = (
    ("floor_fraction", 8.0 / 60.0),
    ("floor_fraction", 16.0 / 60.0),
    ("player_hp_fraction", 0.25),
    ("player_hp_fraction", 0.5),
    ("player_hp_fraction", 0.75),
    ("player_block_fraction", 0.1),
    ("player_block_fraction", 0.25),
    ("deck_size_fraction", 0.2),
    ("deck_size_fraction", 0.35),
    ("relic_count_fraction", 0.1),
    ("relic_count_fraction", 0.2),
    ("potion_count_fraction", 0.2),
    ("potion_count_fraction", 0.4),
    ("combat_round_fraction", 0.1),
    ("combat_round_fraction", 0.25),
    ("enemy_hp_fraction", 0.25),
    ("enemy_hp_fraction", 0.5),
    ("enemy_hp_fraction", 0.75),
    ("incoming_damage_fraction", 0.05),
    ("incoming_damage_fraction", 0.25),
)
_NONLINEAR_INDICATORS = (
    "incoming_damage_zero",
    "player_hp_above_75pct",
    "floor_above_8",
    "combat_round_at_most_2",
)
_NONLINEAR_PRODUCTS = (
    "floor_x_missing_hp",
    "unblocked_incoming_damage",
    "combat_round_x_missing_hp",
    "enemy_hp_x_incoming_damage",
)
_NONLINEAR_GLOBAL_FEATURE_NAMES = tuple(
    f"hinge:{feature}>{threshold:.6f}" for feature, threshold in _NONLINEAR_HINGES
) + tuple(
    [f"indicator:{name}" for name in _NONLINEAR_INDICATORS]
    + [f"product:{name}" for name in _NONLINEAR_PRODUCTS]
)
VALUE_NONLINEAR_FEATURE_NAMES = (
    VALUE_INTERACTION_FEATURE_NAMES
    + (_NONLINEAR_GLOBAL_FEATURE_NAMES)
    + tuple(
        f"interaction:{character.rsplit('.', 1)[-1].lower()}:{feature}"
        for character in VALUE_CHARACTERS
        for feature in _NONLINEAR_GLOBAL_FEATURE_NAMES
    )
)

_FEATURE_LABELS = {
    "ascension_fraction": "ascension",
    "act_fraction": "act progress",
    "floor_fraction": "floor progress",
    "player_hp_fraction": "player health",
    "player_block_fraction": "player block",
    "gold_fraction": "gold",
    "deck_size_fraction": "deck size",
    "relic_count_fraction": "relic count",
    "potion_count_fraction": "potion count",
    "combat_round_fraction": "combat round",
    "energy_fraction": "current energy",
    "hand_size_fraction": "hand size",
    "playable_card_fraction": "playable cards",
    "enemy_count_fraction": "living enemies",
    "enemy_hp_fraction": "enemy health",
    "enemy_block_fraction": "enemy block",
    "incoming_damage_fraction": "incoming damage",
    "player_power_count_fraction": "player powers",
    "orb_count_fraction": "Defect orbs",
    "stars_fraction": "Regent stars",
    "pet_hp_fraction": "Necrobinder pet health",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _objects(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _canonical_character(value: Any) -> str:
    text = str(value or "").upper()
    if not text.startswith("CHARACTER."):
        text = "CHARACTER." + text
    if text not in VALUE_CHARACTERS:
        raise ValueError(f"Unsupported value-model character: {value}")
    return text


def value_feature_names(feature_schema: str) -> tuple[str, ...]:
    if feature_schema == "linear_v1":
        return VALUE_FEATURE_NAMES
    if feature_schema == "character_interactions_v2":
        return VALUE_INTERACTION_FEATURE_NAMES
    if feature_schema == "character_nonlinear_v3":
        return VALUE_NONLINEAR_FEATURE_NAMES
    raise ValueError(f"Unsupported value feature schema: {feature_schema}")


def public_value_features(
    observation: Mapping[str, Any], *, feature_schema: str = "linear_v1"
) -> np.ndarray:
    """Convert only the exported observation into a compact explicit feature vector."""

    character = _canonical_character(observation.get("character"))
    context = _mapping(observation.get("context"))
    player = _mapping(observation.get("player"))
    decision = _mapping(observation.get("decision_state"))
    max_hp = max(_number(player.get("max_hp"), 1.0), 1.0)
    hand = _objects(decision.get("hand"))
    enemies = [enemy for enemy in _objects(decision.get("enemies")) if _number(enemy.get("hp")) > 0]
    total_enemy_hp = sum(_number(enemy.get("hp")) for enemy in enemies)
    total_enemy_max = max(
        sum(max(_number(enemy.get("max_hp")), _number(enemy.get("hp"))) for enemy in enemies),
        1.0,
    )
    total_enemy_block = sum(max(_number(enemy.get("block")), 0.0) for enemy in enemies)
    incoming = 0.0
    for enemy in enemies:
        for intent in _objects(enemy.get("intents")):
            if intent.get("total_damage") is not None:
                incoming += _number(intent.get("total_damage"))
            else:
                incoming += _number(intent.get("damage")) * max(
                    _number(intent.get("hits"), 1.0), 1.0
                )
    pet = _mapping(decision.get("osty") or decision.get("pet"))
    pet_max_hp = max(_number(pet.get("max_hp")), 1.0)

    values = [float(character == candidate) for candidate in VALUE_CHARACTERS]
    values.extend(
        [
            _number(observation.get("ascension")) / 10.0,
            _number(context.get("act")) / 4.0,
            _number(context.get("floor")) / 60.0,
            _number(player.get("hp")) / max_hp,
            max(_number(player.get("block")), 0.0) / max_hp,
            min(max(_number(player.get("gold")), 0.0) / 500.0, 2.0),
            _number(player.get("deck_size"), len(_objects(player.get("deck")))) / 60.0,
            len(_objects(player.get("relics"))) / 30.0,
            len(_objects(player.get("potions"))) / 5.0,
            _number(decision.get("round")) / 20.0,
            _number(decision.get("energy")) / 5.0,
            len(hand) / 10.0,
            sum(bool(card.get("can_play")) for card in hand) / 10.0,
            len(enemies) / 5.0,
            total_enemy_hp / total_enemy_max,
            total_enemy_block / total_enemy_max,
            incoming / max_hp,
            len(_objects(decision.get("player_powers"))) / 10.0,
            len(_objects(decision.get("orbs"))) / 10.0,
            _number(decision.get("stars")) / 10.0,
            (_number(pet.get("hp")) / pet_max_hp) if pet else 0.0,
        ]
    )
    if feature_schema in {"character_interactions_v2", "character_nonlinear_v3"}:
        base_by_name = dict(zip(VALUE_FEATURE_NAMES, values))
        for candidate in VALUE_CHARACTERS:
            active = float(character == candidate)
            values.extend(
                active * float(base_by_name[feature]) for feature in _INTERACTION_BASE_FEATURES
            )
    if feature_schema == "character_nonlinear_v3":
        nonlinear_values = [
            max(float(base_by_name[feature]) - threshold, 0.0)
            for feature, threshold in _NONLINEAR_HINGES
        ]
        nonlinear_values.extend(
            [
                float(base_by_name["incoming_damage_fraction"] == 0.0),
                float(base_by_name["player_hp_fraction"] > 0.75),
                float(base_by_name["floor_fraction"] > 8.0 / 60.0),
                float(base_by_name["combat_round_fraction"] <= 0.1),
            ]
        )
        missing_hp = max(1.0 - float(base_by_name["player_hp_fraction"]), 0.0)
        nonlinear_values.extend(
            [
                float(base_by_name["floor_fraction"]) * missing_hp,
                max(
                    float(base_by_name["incoming_damage_fraction"])
                    - float(base_by_name["player_block_fraction"]),
                    0.0,
                ),
                float(base_by_name["combat_round_fraction"]) * missing_hp,
                float(base_by_name["enemy_hp_fraction"])
                * float(base_by_name["incoming_damage_fraction"]),
            ]
        )
        values.extend(nonlinear_values)
        for candidate in VALUE_CHARACTERS:
            active = float(character == candidate)
            values.extend(active * value for value in nonlinear_values)
    names = value_feature_names(feature_schema)
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(names),):
        raise AssertionError("Value feature schema length mismatch")
    return result


@dataclass(frozen=True)
class ValueExample:
    features: np.ndarray
    label: float
    split: str
    character: str
    group_sha256: str


@dataclass(frozen=True)
class ValueCorpus:
    examples: tuple[ValueExample, ...]
    trajectories: int
    excluded_trajectories: int
    duplicate_states: int
    horizon_floors: int
    feature_schema: str


def _trajectory_files(inputs: Sequence[str | os.PathLike[str]]) -> list[Path]:
    files: set[Path] = set()
    for raw in inputs:
        path = Path(raw)
        if path.is_file() and path.suffix.lower() == ".jsonl":
            files.add(path.resolve())
        elif path.is_dir():
            files.update(candidate.resolve() for candidate in path.rglob("*.jsonl"))
    return sorted(files)


def build_value_corpus(
    inputs: Sequence[str | os.PathLike[str]],
    *,
    horizon_floors: int = 3,
    feature_schema: str = "linear_v1",
) -> ValueCorpus:
    """Build seed-grouped combat values from completed first-party trajectories."""

    if horizon_floors <= 0:
        raise ValueError("horizon_floors must be positive")
    examples: list[ValueExample] = []
    seen: set[str] = set()
    trajectories = 0
    excluded = 0
    duplicates = 0
    for path in _trajectory_files(inputs):
        records: list[Mapping[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                value = json.loads(line)
                if isinstance(value, Mapping):
                    records.append(value)
        except (OSError, UnicodeError, json.JSONDecodeError):
            excluded += 1
            continue
        if len(records) < 3:
            excluded += 1
            continue
        start, end = records[0], records[-1]
        if (
            start.get("event_type") != "episode_start"
            or end.get("event_type") != "episode_end"
            or end.get("terminated") is not True
            or end.get("truncated") is True
        ):
            excluded += 1
            continue
        try:
            character = _canonical_character(start.get("character"))
        except ValueError:
            excluded += 1
            continue
        seed = str(start.get("seed") or "")
        if not seed:
            excluded += 1
            continue
        trajectories += 1
        final_floor = int(end.get("floor") or 0)
        victory = end.get("victory") is True
        group_sha = hashlib.sha256(f"{character}\x1f{seed}".encode()).hexdigest()
        seen_rounds: set[tuple[int, int]] = set()
        for record in records:
            if record.get("event_type") != "transition":
                continue
            observation = _mapping(record.get("observation"))
            if observation.get("decision_type") != "combat_play":
                continue
            context = _mapping(observation.get("context"))
            decision = _mapping(observation.get("decision_state"))
            floor = int(context.get("floor") or 0)
            round_number = int(decision.get("round") or 0)
            round_key = (floor, round_number)
            if round_key in seen_rounds:
                continue
            seen_rounds.add(round_key)
            canonical = json.dumps(
                observation, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            state_key = hashlib.sha256(f"{group_sha}\x1f{canonical}".encode("utf-8")).hexdigest()
            if state_key in seen:
                duplicates += 1
                continue
            seen.add(state_key)
            label = float(victory or final_floor >= floor + horizon_floors)
            examples.append(
                ValueExample(
                    features=public_value_features(observation, feature_schema=feature_schema),
                    label=label,
                    split="pending",
                    character=character,
                    group_sha256=group_sha,
                )
            )
    if not examples:
        raise ValueError("No completed combat trajectory states were found")
    groups_by_character: dict[str, set[str]] = {}
    for example in examples:
        groups_by_character.setdefault(example.character, set()).add(example.group_sha256)
    group_splits: dict[str, str] = {}
    for character, groups in groups_by_character.items():
        ranked = sorted(
            groups,
            key=lambda group: hashlib.sha256(
                f"value-split-v1\x1f{character}\x1f{group}".encode()
            ).digest(),
        )
        validation_count = max(1, round(len(ranked) * 0.1))
        test_count = max(1, round(len(ranked) * 0.1))
        if validation_count + test_count >= len(ranked):
            raise ValueError(f"Too few seed groups for character: {character}")
        for index, group in enumerate(ranked):
            if index < validation_count:
                group_splits[group] = "validation"
            elif index < validation_count + test_count:
                group_splits[group] = "test"
            else:
                group_splits[group] = "train"
    examples = [
        ValueExample(
            features=example.features,
            label=example.label,
            split=group_splits[example.group_sha256],
            character=example.character,
            group_sha256=example.group_sha256,
        )
        for example in examples
    ]
    return ValueCorpus(
        examples=tuple(examples),
        trajectories=trajectories,
        excluded_trajectories=excluded,
        duplicate_states=duplicates,
        horizon_floors=horizon_floors,
        feature_schema=feature_schema,
    )


def _sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    positive = array >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    negative_exp = np.exp(array[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return float(result) if result.ndim == 0 else result


def calibration_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-9, 1 - 1e-9)
    if labels.size == 0:
        return {"examples": 0}
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper >= 1.0 else probabilities < upper
        )
        count = int(mask.sum())
        if not count:
            continue
        predicted = float(probabilities[mask].mean())
        observed = float(labels[mask].mean())
        ece += count / labels.size * abs(predicted - observed)
        bins.append(
            {
                "lower": round(float(lower), 1),
                "upper": round(float(upper), 1),
                "examples": count,
                "mean_probability": round(predicted, 6),
                "observed_rate": round(observed, 6),
            }
        )
    return {
        "examples": int(labels.size),
        "positive_rate": round(float(labels.mean()), 6),
        "brier": round(float(np.mean((probabilities - labels) ** 2)), 6),
        "log_loss": round(
            float(
                -np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))
            ),
            6,
        ),
        "accuracy_at_0_5": round(float(np.mean((probabilities >= 0.5) == labels)), 6),
        "ece_10": round(float(ece), 6),
        "bins": bins,
    }


class CalibratedLinearValueEvaluator:
    """Character-aware probability of surviving a fixed number of future floors."""

    def __init__(
        self,
        weights: np.ndarray,
        bias: float | np.ndarray,
        calibration_slope: float | np.ndarray,
        calibration_intercept: float | np.ndarray,
        *,
        horizon_floors: int,
        build_id: str = "v0.107.1",
        calibration_status: str = "unverified",
        search_eligible: bool = False,
        feature_schema: str = "linear_v1",
        calibration_kind: str = "global_platt",
    ):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.feature_schema = feature_schema
        self.feature_names = value_feature_names(feature_schema)
        if self.weights.ndim not in {1, 2} or self.weights.shape[-1] != len(self.feature_names):
            raise ValueError("Value checkpoint feature shape does not match schema")
        self.ensemble_members = 1 if self.weights.ndim == 1 else self.weights.shape[0]
        self.bias = np.asarray(bias, dtype=np.float64)
        if self.bias.ndim == 0:
            if self.ensemble_members != 1:
                raise ValueError("Ensemble value checkpoint needs one bias per member")
        elif self.bias.shape != (self.ensemble_members,):
            raise ValueError("Value checkpoint bias shape does not match its members")
        self.calibration_kind = calibration_kind
        if calibration_kind not in {"identity", "global_platt", "per_character_platt"}:
            raise ValueError(f"Unsupported calibration kind: {calibration_kind}")
        self.calibration_slope = np.asarray(calibration_slope, dtype=np.float64)
        self.calibration_intercept = np.asarray(calibration_intercept, dtype=np.float64)
        expected_calibration_shape = (
            (len(VALUE_CHARACTERS),) if calibration_kind == "per_character_platt" else ()
        )
        if (
            self.calibration_slope.shape != expected_calibration_shape
            or self.calibration_intercept.shape != expected_calibration_shape
        ):
            raise ValueError("Value checkpoint calibration shape does not match its kind")
        self.horizon_floors = int(horizon_floors)
        self.build_id = build_id
        self.calibration_status = calibration_status
        self.search_eligible = bool(search_eligible)

    def _member_logits(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        member_weights = self.weights[None, :] if self.weights.ndim == 1 else self.weights
        member_bias = self.bias.reshape(1) if self.bias.ndim == 0 else self.bias
        return matrix @ member_weights.T + member_bias

    def raw_probabilities(self, features: np.ndarray) -> np.ndarray:
        member_probabilities = np.asarray(_sigmoid(self._member_logits(features)))
        return np.asarray(member_probabilities.mean(axis=-1))

    def logits(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.clip(self.raw_probabilities(features), 1e-9, 1 - 1e-9)
        return np.asarray(np.log(probabilities / (1.0 - probabilities)))

    def _calibration_parameters(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.calibration_kind != "per_character_platt":
            return self.calibration_slope, self.calibration_intercept
        matrix = np.asarray(features, dtype=np.float64)
        character_index = np.argmax(matrix[..., : len(VALUE_CHARACTERS)], axis=-1)
        return (
            self.calibration_slope[character_index],
            self.calibration_intercept[character_index],
        )

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        raw = self.logits(features)
        slope, intercept = self._calibration_parameters(features)
        return np.asarray(_sigmoid(slope * raw + intercept))

    def evaluate(self, env: Any) -> PolicyValueEstimate:
        observation = env.observation
        features = public_value_features(observation, feature_schema=self.feature_schema)
        raw_logit = float(self.logits(features))
        slope, intercept = self._calibration_parameters(features)
        scalar_slope = float(slope)
        probability = float(_sigmoid(scalar_slope * raw_logit + float(intercept)))
        mean_weights = self.weights if self.weights.ndim == 1 else self.weights.mean(axis=0)
        contributions = scalar_slope * mean_weights * features
        ranked = sorted(
            range(len(contributions)),
            key=lambda index: abs(float(contributions[index])),
            reverse=True,
        )[:5]
        factors = []
        for index in ranked:
            name = self.feature_names[index]
            contribution = float(contributions[index])
            if name.startswith("interaction:"):
                _, interaction_character, base_feature = name.split(":", 2)
                label = (
                    f"{interaction_character.title()} × "
                    f"{_FEATURE_LABELS.get(base_feature, base_feature)}"
                )
            else:
                label = _FEATURE_LABELS.get(name, name.replace("character:", "character "))
            factors.append(
                {
                    "feature": name,
                    "label": label,
                    "value": round(float(features[index]), 6),
                    "log_odds_contribution": round(contribution, 6),
                    "direction": "raises" if contribution >= 0 else "lowers",
                }
            )
        character = _canonical_character(observation.get("character"))
        return PolicyValueEstimate(
            value=probability,
            metadata={
                "source": (
                    "calibrated_nonlinear_public_value_ensemble_v4"
                    if self.ensemble_members > 1
                    else "calibrated_linear_public_value_v1"
                ),
                "target": f"survive_next_{self.horizon_floors}_floors",
                "character": character,
                "trained": True,
                "calibrated": True,
                "calibration_kind": self.calibration_kind,
                "ensemble_members": self.ensemble_members,
                "calibration_status": self.calibration_status,
                "search_eligible": self.search_eligible,
                "uses_hidden_state": False,
                "training_source": "first_party_headless_trajectories",
                "explanation": {
                    "summary": (
                        f"{character.rsplit('.', 1)[-1].title()} has an estimated "
                        f"{probability:.1%} chance to survive the next "
                        f"{self.horizon_floors} floors under the logged-policy distribution."
                    ),
                    "top_factors": factors,
                    "caveat": (
                        "This is a policy-conditional short-horizon estimate, not win probability."
                    ),
                },
            },
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        metadata = {
            "schema_version": 1,
            "model_type": "calibrated_linear_public_value",
            "feature_names": self.feature_names,
            "feature_schema": self.feature_schema,
            "characters": VALUE_CHARACTERS,
            "horizon_floors": self.horizon_floors,
            "build_id": self.build_id,
            "training_source": "first_party_headless_trajectories",
            "search_labels_used": False,
            "calibration_status": self.calibration_status,
            "search_eligible": self.search_eligible,
            "ensemble_members": self.ensemble_members,
            "calibration_kind": self.calibration_kind,
        }
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                    weights=self.weights,
                    bias=self.bias,
                    calibration_slope=self.calibration_slope,
                    calibration_intercept=self.calibration_intercept,
                )
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> CalibratedLinearValueEvaluator:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.get("model_type") != "calibrated_linear_public_value":
                raise ValueError("Not an STSAI calibrated public-value checkpoint")
            feature_schema = str(metadata.get("feature_schema", "linear_v1"))
            if tuple(metadata.get("feature_names") or ()) != value_feature_names(feature_schema):
                raise ValueError("Value checkpoint uses a different feature schema")
            return cls(
                data["weights"],
                data["bias"],
                data["calibration_slope"],
                data["calibration_intercept"],
                horizon_floors=int(metadata["horizon_floors"]),
                build_id=str(metadata["build_id"]),
                calibration_status=str(metadata.get("calibration_status", "unverified")),
                search_eligible=metadata.get("search_eligible") is True,
                feature_schema=feature_schema,
                calibration_kind=str(metadata.get("calibration_kind", "global_platt")),
            )


def _matrix(examples: Iterable[ValueExample]) -> tuple[np.ndarray, np.ndarray]:
    values = list(examples)
    return (
        np.stack([example.features for example in values]),
        np.asarray([example.label for example in values], dtype=np.float64),
    )


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    sample_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    observation_weights = (
        np.ones(labels.shape, dtype=np.float64)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    if observation_weights.shape != labels.shape or np.any(observation_weights < 0):
        raise ValueError("Logistic sample weights must align with nonnegative labels")
    weight_total = float(observation_weights.sum())
    if weight_total <= 0:
        raise ValueError("Logistic sample weights must have positive total weight")
    weights = np.zeros(features.shape[1], dtype=np.float64)
    bias = 0.0
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = v_b = 0.0
    for step in range(1, epochs + 1):
        probability = np.asarray(_sigmoid(features @ weights + bias))
        error = probability - labels
        weighted_error = error * observation_weights
        grad_w = features.T @ weighted_error / weight_total + l2 * weights
        grad_b = float(weighted_error.sum() / weight_total)
        m_w = 0.9 * m_w + 0.1 * grad_w
        v_w = 0.999 * v_w + 0.001 * grad_w**2
        m_b = 0.9 * m_b + 0.1 * grad_b
        v_b = 0.999 * v_b + 0.001 * grad_b**2
        weights -= (
            learning_rate * (m_w / (1 - 0.9**step)) / (np.sqrt(v_w / (1 - 0.999**step)) + 1e-8)
        )
        bias -= (
            learning_rate * (m_b / (1 - 0.9**step)) / (math.sqrt(v_b / (1 - 0.999**step)) + 1e-8)
        )
    return weights, bias


def _fit_platt(
    raw_logits: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    log_slope = 0.0
    intercept = 0.0
    learning_rate = 0.02
    for _ in range(500):
        slope = math.exp(log_slope)
        probabilities = np.asarray(_sigmoid(slope * raw_logits + intercept))
        error = (probabilities - labels) * weights
        log_slope -= learning_rate * float(np.mean(error * raw_logits * slope))
        intercept -= learning_rate * float(error.mean())
        log_slope = min(max(log_slope, -4.0), 4.0)
    return math.exp(log_slope), intercept


def evaluate_value_examples(
    model: CalibratedLinearValueEvaluator,
    examples: Sequence[ValueExample],
) -> dict[str, Any]:
    features, labels = _matrix(examples)
    calibrated = model.probabilities(features)
    raw = np.asarray(_sigmoid(model.logits(features)))
    result: dict[str, Any] = {
        "calibrated": calibration_metrics(labels, calibrated),
        "uncalibrated": calibration_metrics(labels, raw),
        "by_character": {},
    }
    for character in VALUE_CHARACTERS:
        selected = [example for example in examples if example.character == character]
        if not selected:
            continue
        char_x, char_y = _matrix(selected)
        result["by_character"][character] = {
            "calibrated": calibration_metrics(char_y, model.probabilities(char_x)),
            "uncalibrated": calibration_metrics(char_y, np.asarray(_sigmoid(model.logits(char_x)))),
            "calibrated_ece_10_group_bootstrap_95_interval": (
                _bootstrap_ece_interval(model, selected)
            ),
        }
    return result


def _promotion_criteria(evaluation: Mapping[str, Any]) -> dict[str, bool]:
    calibrated = _mapping(evaluation.get("calibrated"))
    uncalibrated = _mapping(evaluation.get("uncalibrated"))
    per_character_ece = {
        character: float(_mapping(metrics.get("calibrated")).get("ece_10", 1.0))
        for character, raw_metrics in _mapping(evaluation.get("by_character")).items()
        if isinstance(raw_metrics, Mapping)
        for metrics in [raw_metrics]
    }
    return {
        "overall_ece_improved": (
            _number(calibrated.get("ece_10"), 1.0) <= _number(uncalibrated.get("ece_10"), 1.0)
        ),
        "brier_degradation_at_most_0_002": (
            _number(calibrated.get("brier"), 1.0) <= _number(uncalibrated.get("brier"), 1.0) + 0.002
        ),
        "every_character_ece_at_most_0_15": (
            set(per_character_ece) == set(VALUE_CHARACTERS)
            and all(value <= 0.15 for value in per_character_ece.values())
        ),
    }


def _per_character_ece(evaluation: Mapping[str, Any]) -> dict[str, float]:
    return {
        character: float(_mapping(metrics.get("calibrated")).get("ece_10", 1.0))
        for character, raw_metrics in _mapping(evaluation.get("by_character")).items()
        if isinstance(raw_metrics, Mapping)
        for metrics in [raw_metrics]
    }


def _bootstrap_ece_interval(
    model: CalibratedLinearValueEvaluator,
    examples: Sequence[ValueExample],
    *,
    samples: int = 500,
) -> list[float]:
    grouped: dict[str, list[ValueExample]] = {}
    for example in examples:
        grouped.setdefault(example.group_sha256, []).append(example)
    keys = sorted(grouped)
    seed_material = "|".join(keys).encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big"))
    estimates = []
    for _ in range(samples):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        resampled = [example for key in chosen for example in grouped[str(key)]]
        features, labels = _matrix(resampled)
        estimates.append(calibration_metrics(labels, model.probabilities(features))["ece_10"])
    lower, upper = np.quantile(np.asarray(estimates), [0.025, 0.975])
    return [round(float(lower), 6), round(float(upper), 6)]


def train_value_model(
    inputs: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
    *,
    metrics_output: str | os.PathLike[str] | None = None,
    horizon_floors: int = 3,
    epochs: int = 800,
    learning_rate: float = 0.03,
    l2: float = 1e-3,
    build_id: str = "v0.107.1",
    feature_schema: str = "linear_v1",
    require_independent_audit: bool = False,
) -> dict[str, Any]:
    corpus = build_value_corpus(
        inputs,
        horizon_floors=horizon_floors,
        feature_schema=feature_schema,
    )
    grouped = {
        split: [example for example in corpus.examples if example.split == split]
        for split in ("train", "validation", "test")
    }
    if any(not grouped[split] for split in grouped):
        raise ValueError("Value corpus must contain non-empty train/validation/test splits")
    train_x, train_y = _matrix(grouped["train"])
    validation_x, validation_y = _matrix(grouped["validation"])
    weights, bias = _fit_logistic(
        train_x,
        train_y,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    validation_logits = validation_x @ weights + bias
    validation_group_counts: dict[str, int] = {}
    for example in grouped["validation"]:
        validation_group_counts[example.group_sha256] = (
            validation_group_counts.get(example.group_sha256, 0) + 1
        )
    calibration_weights = np.asarray(
        [1.0 / validation_group_counts[example.group_sha256] for example in grouped["validation"]],
        dtype=np.float64,
    )
    calibration_weights /= calibration_weights.mean()
    slope, intercept = _fit_platt(validation_logits, validation_y, calibration_weights)
    model = CalibratedLinearValueEvaluator(
        weights,
        bias,
        slope,
        intercept,
        horizon_floors=horizon_floors,
        build_id=build_id,
        feature_schema=feature_schema,
    )
    validation_evaluation = evaluate_value_examples(model, grouped["validation"])
    test_evaluation = evaluate_value_examples(model, grouped["test"])
    per_character_ece = _per_character_ece(test_evaluation)
    promotion_criteria = _promotion_criteria(test_evaluation)
    development_gate_passed = all(promotion_criteria.values())
    model.search_eligible = development_gate_passed and not require_independent_audit
    if require_independent_audit:
        model.calibration_status = "awaiting_independent_audit"
    else:
        model.calibration_status = (
            "passed_held_out_gate" if model.search_eligible else "rejected_held_out_gate"
        )
    model.save(output)
    report = {
        "schema_version": 1,
        "model": str(Path(output)),
        "target": f"survive_next_{horizon_floors}_floors",
        "training_source": "first_party_headless_trajectories",
        "search_labels_used": False,
        "corpus": {
            "trajectories": corpus.trajectories,
            "excluded_trajectories": corpus.excluded_trajectories,
            "duplicate_states": corpus.duplicate_states,
            "examples": len(corpus.examples),
            "splits": {key: len(value) for key, value in grouped.items()},
            "groups": {
                key: len({example.group_sha256 for example in value})
                for key, value in grouped.items()
            },
        },
        "configuration": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "calibration": "monotonic_platt_on_validation",
            "split": "per_character_sha256_ranked_seed_groups_80_10_10",
            "character_conditioned": True,
            "feature_schema": feature_schema,
            "explainable_linear_features": list(model.feature_names),
        },
        "calibration_promotion_gate": {
            "passed": model.search_eligible,
            "development_held_out_passed": development_gate_passed,
            "independent_audit_required": require_independent_audit,
            "criteria": promotion_criteria,
            "per_character_test_ece_10": per_character_ece,
            "decision": (
                "eligible_for_paired_search_smoke"
                if model.search_eligible
                else (
                    "await_independent_audit"
                    if require_independent_audit
                    else "do_not_use_for_search"
                )
            ),
        },
        "validation": validation_evaluation,
        "test": test_evaluation,
    }
    if metrics_output is not None:
        destination = Path(metrics_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(destination)
    return report


def _group_balanced_weights(examples: Sequence[ValueExample]) -> np.ndarray:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.group_sha256] = counts.get(example.group_sha256, 0) + 1
    weights = np.asarray(
        [1.0 / counts[example.group_sha256] for example in examples],
        dtype=np.float64,
    )
    return weights / weights.mean()


def _bootstrap_groups(examples: Sequence[ValueExample], *, seed: int) -> list[ValueExample]:
    grouped: dict[str, dict[str, list[ValueExample]]] = {}
    for example in examples:
        grouped.setdefault(example.character, {}).setdefault(example.group_sha256, []).append(
            example
        )
    rng = np.random.default_rng(seed)
    selected: list[ValueExample] = []
    for character in VALUE_CHARACTERS:
        character_groups = grouped.get(character) or {}
        keys = sorted(character_groups)
        if not keys:
            raise ValueError(f"No ensemble training groups for {character}")
        chosen = rng.choice(keys, size=len(keys), replace=True)
        for key in chosen:
            selected.extend(character_groups[str(key)])
    return selected


def _fit_calibrator(
    kind: str,
    raw_logits: np.ndarray,
    labels: np.ndarray,
    examples: Sequence[ValueExample],
) -> tuple[np.ndarray, np.ndarray]:
    if kind == "identity":
        return np.asarray(1.0), np.asarray(0.0)
    if kind == "global_platt":
        slope, intercept = _fit_platt(
            raw_logits,
            labels,
            _group_balanced_weights(examples),
        )
        return np.asarray(slope), np.asarray(intercept)
    if kind != "per_character_platt":
        raise ValueError(f"Unsupported calibrator candidate: {kind}")
    slopes = np.ones(len(VALUE_CHARACTERS), dtype=np.float64)
    intercepts = np.zeros(len(VALUE_CHARACTERS), dtype=np.float64)
    for character_index, character in enumerate(VALUE_CHARACTERS):
        selected_indices = [
            index for index, example in enumerate(examples) if example.character == character
        ]
        if not selected_indices:
            raise ValueError(f"No calibration examples for {character}")
        selected_examples = [examples[index] for index in selected_indices]
        selected = np.asarray(selected_indices, dtype=np.int64)
        slopes[character_index], intercepts[character_index] = _fit_platt(
            raw_logits[selected],
            labels[selected],
            _group_balanced_weights(selected_examples),
        )
    return slopes, intercepts


def _apply_calibrator(
    kind: str,
    raw_logits: np.ndarray,
    examples: Sequence[ValueExample],
    slope: np.ndarray,
    intercept: np.ndarray,
) -> np.ndarray:
    if kind != "per_character_platt":
        return np.asarray(_sigmoid(slope * raw_logits + intercept))
    character_indices = np.asarray(
        [VALUE_CHARACTERS.index(example.character) for example in examples],
        dtype=np.int64,
    )
    return np.asarray(
        _sigmoid(slope[character_indices] * raw_logits + intercept[character_indices])
    )


def _prediction_evaluation(
    examples: Sequence[ValueExample], probabilities: np.ndarray
) -> dict[str, Any]:
    labels = np.asarray([example.label for example in examples], dtype=np.float64)
    result: dict[str, Any] = {
        "overall": calibration_metrics(labels, probabilities),
        "by_character": {},
    }
    for character in VALUE_CHARACTERS:
        selected = np.asarray([example.character == character for example in examples], dtype=bool)
        result["by_character"][character] = calibration_metrics(
            labels[selected], probabilities[selected]
        )
    return result


def _select_grouped_calibrator(
    model: CalibratedLinearValueEvaluator,
    examples: Sequence[ValueExample],
    *,
    folds: int = 5,
) -> tuple[str, dict[str, Any]]:
    """Select identity/global/per-character Platt without opening internal test."""

    if folds < 2:
        raise ValueError("Grouped calibrator selection requires at least two folds")
    group_folds: dict[str, int] = {}
    for character in VALUE_CHARACTERS:
        keys = sorted(
            {example.group_sha256 for example in examples if example.character == character},
            key=lambda key: hashlib.sha256(
                f"value-v4-calibration-fold\x1f{character}\x1f{key}".encode()
            ).digest(),
        )
        if len(keys) < folds:
            raise ValueError(f"Too few validation groups for {character}")
        for index, key in enumerate(keys):
            group_folds[key] = index % folds
    features, labels = _matrix(examples)
    raw_logits = model.logits(features)
    candidates = ("identity", "global_platt", "per_character_platt")
    predictions = {candidate: np.zeros(labels.shape, dtype=np.float64) for candidate in candidates}
    for fold in range(folds):
        held_indices = [
            index
            for index, example in enumerate(examples)
            if group_folds[example.group_sha256] == fold
        ]
        fit_indices = [
            index
            for index, example in enumerate(examples)
            if group_folds[example.group_sha256] != fold
        ]
        fit_examples = [examples[index] for index in fit_indices]
        held_examples = [examples[index] for index in held_indices]
        fit_array = np.asarray(fit_indices, dtype=np.int64)
        held_array = np.asarray(held_indices, dtype=np.int64)
        for candidate in candidates:
            slope, intercept = _fit_calibrator(
                candidate,
                raw_logits[fit_array],
                labels[fit_array],
                fit_examples,
            )
            predictions[candidate][held_array] = _apply_calibrator(
                candidate,
                raw_logits[held_array],
                held_examples,
                slope,
                intercept,
            )
    evaluations = {
        candidate: _prediction_evaluation(examples, predictions[candidate])
        for candidate in candidates
    }
    identity = evaluations["identity"]["overall"]
    summaries: dict[str, Any] = {}
    admissible: list[str] = []
    for candidate in candidates:
        evaluation = evaluations[candidate]
        worst_character_ece = max(
            float(evaluation["by_character"][character]["ece_10"]) for character in VALUE_CHARACTERS
        )
        criteria = {
            "overall_ece_no_worse_than_identity": (
                float(evaluation["overall"]["ece_10"]) <= float(identity["ece_10"])
            ),
            "brier_degradation_from_identity_at_most_0_002": (
                float(evaluation["overall"]["brier"]) <= float(identity["brier"]) + 0.002
            ),
            "every_character_ece_at_most_0_15": worst_character_ece <= 0.15,
        }
        if all(criteria.values()):
            admissible.append(candidate)
        summaries[candidate] = {
            "overall": evaluation["overall"],
            "by_character_ece_10": {
                character: evaluation["by_character"][character]["ece_10"]
                for character in VALUE_CHARACTERS
            },
            "worst_character_ece_10": round(worst_character_ece, 6),
            "admissibility": criteria,
            "admissible": all(criteria.values()),
        }
    priority = {candidate: index for index, candidate in enumerate(candidates)}
    selected = min(
        admissible or ["identity"],
        key=lambda candidate: (
            summaries[candidate]["worst_character_ece_10"],
            summaries[candidate]["overall"]["brier"],
            summaries[candidate]["overall"]["ece_10"],
            priority[candidate],
        ),
    )
    return selected, {
        "method": "five-fold character-balanced seed-group out-of-fold selection",
        "candidates": summaries,
        "selection_rule": (
            "among admissible candidates minimize worst-character ECE, then Brier, "
            "then overall ECE; if none are admissible, select identity"
        ),
        "selected": selected,
        "internal_test_inspected_during_selection": False,
    }


def train_value_model_v4(
    inputs: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
    *,
    metrics_output: str | os.PathLike[str] | None = None,
    horizon_floors: int = 3,
    epochs: int = 400,
    learning_rate: float = 0.02,
    l2: float = 3e-3,
    build_id: str = "v0.107.1",
    ensemble_seeds: Sequence[int] = (101, 211, 307),
    require_independent_audit: bool = True,
) -> dict[str, Any]:
    """Fit the single bounded nonlinear ensemble and select its calibrator once."""

    feature_schema = "character_nonlinear_v3"
    if len(ensemble_seeds) < 2 or len(set(ensemble_seeds)) != len(ensemble_seeds):
        raise ValueError("Value-v4 requires at least two unique ensemble seeds")
    corpus = build_value_corpus(
        inputs,
        horizon_floors=horizon_floors,
        feature_schema=feature_schema,
    )
    grouped = {
        split: [example for example in corpus.examples if example.split == split]
        for split in ("train", "validation", "test")
    }
    if any(not grouped[split] for split in grouped):
        raise ValueError("Value corpus must contain non-empty train/validation/test splits")
    member_weights: list[np.ndarray] = []
    member_biases: list[float] = []
    for seed in ensemble_seeds:
        member_examples = _bootstrap_groups(grouped["train"], seed=seed)
        member_x, member_y = _matrix(member_examples)
        weights, bias = _fit_logistic(
            member_x,
            member_y,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            sample_weights=_group_balanced_weights(member_examples),
        )
        member_weights.append(weights)
        member_biases.append(bias)
    identity_model = CalibratedLinearValueEvaluator(
        np.stack(member_weights),
        np.asarray(member_biases),
        1.0,
        0.0,
        horizon_floors=horizon_floors,
        build_id=build_id,
        feature_schema=feature_schema,
        calibration_kind="identity",
    )
    selected_calibrator, selection_report = _select_grouped_calibrator(
        identity_model, grouped["validation"]
    )
    validation_x, validation_y = _matrix(grouped["validation"])
    final_slope, final_intercept = _fit_calibrator(
        selected_calibrator,
        identity_model.logits(validation_x),
        validation_y,
        grouped["validation"],
    )
    model = CalibratedLinearValueEvaluator(
        np.stack(member_weights),
        np.asarray(member_biases),
        final_slope,
        final_intercept,
        horizon_floors=horizon_floors,
        build_id=build_id,
        feature_schema=feature_schema,
        calibration_kind=selected_calibrator,
    )
    validation_evaluation = evaluate_value_examples(model, grouped["validation"])
    test_evaluation = evaluate_value_examples(model, grouped["test"])
    promotion_criteria = _promotion_criteria(test_evaluation)
    development_gate_passed = all(promotion_criteria.values())
    model.search_eligible = development_gate_passed and not require_independent_audit
    if development_gate_passed and require_independent_audit:
        model.calibration_status = "awaiting_independent_audit"
    elif development_gate_passed:
        model.calibration_status = "passed_held_out_gate"
    else:
        model.calibration_status = "rejected_held_out_gate"
    model.save(output)
    report = {
        "schema_version": 1,
        "experiment": "bounded_value_v4",
        "model": str(Path(output)),
        "target": f"survive_next_{horizon_floors}_floors",
        "training_source": "first_party_headless_trajectories",
        "search_labels_used": False,
        "opened_v3_audit_reclassified_as_development": True,
        "corpus": {
            "trajectories": corpus.trajectories,
            "excluded_trajectories": corpus.excluded_trajectories,
            "duplicate_states": corpus.duplicate_states,
            "examples": len(corpus.examples),
            "splits": {key: len(value) for key, value in grouped.items()},
            "groups": {
                key: len({example.group_sha256 for example in value})
                for key, value in grouped.items()
            },
        },
        "configuration": {
            "architecture": "three-member group-balanced bootstrap nonlinear GAM ensemble",
            "ensemble_seeds": list(ensemble_seeds),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "feature_schema": feature_schema,
            "feature_count": len(model.feature_names),
            "explainable_features": list(model.feature_names),
            "split": "per_character_sha256_ranked_seed_groups_80_10_10",
        },
        "calibrator_selection": selection_report,
        "frozen_calibrator": {
            "kind": selected_calibrator,
            "slope": final_slope.tolist(),
            "intercept": final_intercept.tolist(),
        },
        "calibration_promotion_gate": {
            "passed": model.search_eligible,
            "development_held_out_passed": development_gate_passed,
            "independent_audit_required": require_independent_audit,
            "criteria": promotion_criteria,
            "per_character_test_ece_10": _per_character_ece(test_evaluation),
            "decision": (
                "eligible_for_paired_search_smoke"
                if model.search_eligible
                else (
                    "await_independent_audit"
                    if model.calibration_status == "awaiting_independent_audit"
                    else "do_not_use_for_search"
                )
            ),
        },
        "validation": validation_evaluation,
        "test": test_evaluation,
    }
    if metrics_output is not None:
        _atomic_json(Path(metrics_output), report)
    return report


def evaluate_value_model(
    inputs: Sequence[str | os.PathLike[str]],
    model_path: str | os.PathLike[str],
    *,
    split: str = "test",
) -> dict[str, Any]:
    if split not in {"train", "validation", "test", "all"}:
        raise ValueError(f"Unsupported value split: {split}")
    model = CalibratedLinearValueEvaluator.load(model_path)
    corpus = build_value_corpus(
        inputs,
        horizon_floors=model.horizon_floors,
        feature_schema=model.feature_schema,
    )
    examples = (
        list(corpus.examples)
        if split == "all"
        else [example for example in corpus.examples if example.split == split]
    )
    if not examples:
        raise ValueError(f"No value examples in split: {split}")
    return {
        "model": str(Path(model_path)),
        "split": split,
        "target": f"survive_next_{model.horizon_floors}_floors",
        **evaluate_value_examples(model, examples),
    }


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def audit_value_model(
    inputs: Sequence[str | os.PathLike[str]],
    model_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    *,
    metrics_output: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Open a frozen audit pool once and apply the unchanged promotion gate."""

    model_path = Path(model_path)
    manifest_path = Path(manifest_path)
    model = CalibratedLinearValueEvaluator.load(model_path)
    if model.calibration_status != "awaiting_independent_audit":
        raise ValueError("Independent audit requires an awaiting_independent_audit checkpoint")
    if model.search_eligible:
        raise ValueError("An unaudited checkpoint cannot already be search eligible")
    preaudit_checkpoint_sha256 = _sha256_file(model_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    target = _mapping(manifest.get("target"))
    if str(target.get("build_id")) != model.build_id:
        raise ValueError("Audit manifest and checkpoint build ids differ")
    if str(target.get("value")) != f"survive_next_{model.horizon_floors}_floors":
        raise ValueError("Audit manifest and checkpoint targets differ")
    audit_cases = [
        case
        for case in manifest.get("cases") or []
        if isinstance(case, Mapping) and case.get("pool") == "audit"
    ]
    expected_groups = {
        hashlib.sha256(
            f"{_canonical_character(case.get('character'))}\x1f{case.get('seed')}".encode()
        ).hexdigest()
        for case in audit_cases
    }
    if not audit_cases or len(expected_groups) != len(audit_cases):
        raise ValueError("Audit manifest must contain unique audit cases")
    corpus = build_value_corpus(
        inputs,
        horizon_floors=model.horizon_floors,
        feature_schema=model.feature_schema,
    )
    actual_groups = {example.group_sha256 for example in corpus.examples}
    if (
        corpus.excluded_trajectories
        or corpus.trajectories != len(audit_cases)
        or actual_groups != expected_groups
    ):
        raise ValueError("Audit trajectories do not exactly match the frozen audit manifest")
    evaluation = evaluate_value_examples(model, list(corpus.examples))
    criteria = _promotion_criteria(evaluation)
    passed = all(criteria.values())
    model.search_eligible = passed
    model.calibration_status = (
        "passed_independent_audit_gate" if passed else "rejected_independent_audit_gate"
    )
    model.save(model_path)
    report = {
        "schema_version": 1,
        "model": str(model_path),
        "target": f"survive_next_{model.horizon_floors}_floors",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "preaudit_checkpoint_sha256": preaudit_checkpoint_sha256,
        "final_checkpoint_sha256": _sha256_file(model_path),
        "parameters_frozen_during_audit": True,
        "audit_pool_opened": True,
        "audit_examples": len(corpus.examples),
        "audit_groups": len(actual_groups),
        "audit_trajectories": corpus.trajectories,
        "calibration_promotion_gate": {
            "passed": passed,
            "criteria": criteria,
            "per_character_audit_ece_10": _per_character_ece(evaluation),
            "decision": ("eligible_for_paired_search_smoke" if passed else "do_not_use_for_search"),
        },
        "audit": evaluation,
    }
    if metrics_output is not None:
        _atomic_json(Path(metrics_output), report)
    return report
