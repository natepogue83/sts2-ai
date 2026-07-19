"""A small masked neural baseline for card-reward behavior cloning.

The model scores each legal option independently, then applies a softmax only
over the options offered for that decision. This is preferable to a fixed
output head: new cards introduced by Early Access patches remain representable
through stable feature hashing, and illegal cards can never be selected.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised in dependency-light installs
    raise RuntimeError('Neural policy commands require: pip install -e ".[train]"') from exc


Array = np.ndarray


def read_dataset(
    path: str | os.PathLike[str], split: str | None = None
) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            if split is None or record.get("split") == split:
                yield record


class FeatureHasher:
    """Turn a state/candidate pair into a bounded, patch-tolerant vector."""

    NUMERIC_FEATURES = 8

    def __init__(self, dimension: int = 512):
        if dimension <= self.NUMERIC_FEATURES:
            raise ValueError(f"dimension must be greater than {self.NUMERIC_FEATURES}")
        self.dimension = dimension

    def _add(self, vector: Array, token: str, value: float = 1.0) -> None:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        index = self.NUMERIC_FEATURES + (
            int.from_bytes(digest[:8], "big") % (self.dimension - self.NUMERIC_FEATURES)
        )
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[index] += sign * value

    def encode(self, record: Mapping[str, Any], option: str) -> Array:
        observation = record.get("observation") or {}
        vector = np.zeros(self.dimension, dtype=np.float32)
        max_hp = max(float(observation.get("max_hp") or 0), 1.0)
        deck = [str(value) for value in (observation.get("deck") or [])]
        relics = [str(value) for value in (observation.get("relics") or [])]
        monsters = [str(value) for value in (observation.get("monsters") or [])]
        turns = observation.get("turns_taken")
        vector[: self.NUMERIC_FEATURES] = (
            float(observation.get("current_hp") or 0) / max_hp,
            min(float(observation.get("current_gold") or 0) / 500.0, 3.0),
            min(float(observation.get("ascension") or 0) / 10.0, 2.0),
            min(float(observation.get("act") or 0) / 4.0, 1.5),
            min(float(observation.get("floor") or 0) / 60.0, 1.5),
            min(len(deck) / 50.0, 2.0),
            min(len(relics) / 30.0, 2.0),
            min(float(turns or 0) / 20.0, 2.0),
        )

        character = str(observation.get("character", "unknown"))
        build_id = str((record.get("patch") or {}).get("build_id", "unknown"))
        decision_type = str(record.get("decision_type", "unknown"))
        self._add(vector, f"decision={decision_type}", 2.0)
        self._add(vector, f"character={character}")
        self._add(vector, f"build={build_id}")
        self._add(vector, f"map={observation.get('map_point_type')}")
        self._add(vector, f"room={observation.get('room_id')}")
        self._add(vector, f"option={option}", 2.0)
        self._add(vector, f"character={character}|option={option}", 2.0)

        deck_counts: dict[str, int] = {}
        for card in deck:
            deck_counts[card] = deck_counts.get(card, 0) + 1
        for card, count in deck_counts.items():
            weight = math.sqrt(count)
            self._add(vector, f"deck={card}", weight)
            self._add(vector, f"deck={card}|option={option}", weight)
        for relic in relics:
            self._add(vector, f"relic={relic}")
            self._add(vector, f"relic={relic}|option={option}")
        for monster in monsters:
            self._add(vector, f"monster={monster}")
        return vector


@dataclass(frozen=True)
class PreparedDecision:
    features: Array
    chosen_index: int
    patch: str
    decision_type: str
    character: str


def prepare_records(
    records: Iterable[Mapping[str, Any]], hasher: FeatureHasher
) -> list[PreparedDecision]:
    prepared: list[PreparedDecision] = []
    for record in records:
        options = [str(value) for value in (record.get("legal_options") or [])]
        chosen = str(record.get("chosen", ""))
        if not options or chosen not in options:
            continue
        features = np.stack([hasher.encode(record, option) for option in options])
        patch = str((record.get("patch") or {}).get("build_id", "unknown"))
        decision_type = str(record.get("decision_type", "unknown"))
        observation = record.get("observation") or {}
        character = str(observation.get("character", "unknown"))
        prepared.append(
            PreparedDecision(features, options.index(chosen), patch, decision_type, character)
        )
    return prepared


@dataclass
class Metrics:
    decisions: int = 0
    loss: float = 0.0
    top1_accuracy: float = 0.0
    mean_reciprocal_rank: float = 0.0
    random_top1_accuracy: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "loss": round(self.loss, 6),
            "top1_accuracy": round(self.top1_accuracy, 6),
            "mean_reciprocal_rank": round(self.mean_reciprocal_rank, 6),
            "random_top1_accuracy": round(self.random_top1_accuracy, 6),
        }


class MaskedMLPPolicy:
    """One-hidden-layer MLP trained with listwise cross entropy and Adam."""

    def __init__(self, dimension: int = 512, hidden: int = 64, seed: int = 7):
        if hidden < 1:
            raise ValueError("hidden must be positive")
        self.dimension = dimension
        self.hidden = hidden
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.w1 = (
            rng.standard_normal((dimension, hidden), dtype=np.float32)
            * np.float32(math.sqrt(2.0 / dimension))
        )
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = (
            rng.standard_normal((hidden,), dtype=np.float32)
            * np.float32(math.sqrt(2.0 / hidden))
        )
        self.b2 = np.float32(0.0)
        self._step = 0
        self._m = [np.zeros_like(value) for value in self.parameters]
        self._v = [np.zeros_like(value) for value in self.parameters]

    @property
    def parameters(self) -> list[Array]:
        return [self.w1, self.b1, self.w2, np.asarray(self.b2).reshape(())]

    def scores(self, features: Array) -> Array:
        hidden = np.tanh(features @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2

    def _batch_gradients(
        self, decisions: Sequence[PreparedDecision]
    ) -> tuple[float, int, list[Array]]:
        features = np.concatenate([decision.features for decision in decisions], axis=0)
        hidden = np.tanh(features @ self.w1 + self.b1)
        scores = hidden @ self.w2 + self.b2
        score_gradient = np.zeros_like(scores)
        total_loss = 0.0
        correct = 0
        offset = 0
        for decision in decisions:
            size = len(decision.features)
            group = scores[offset : offset + size]
            shifted = group - np.max(group)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum()
            chosen = decision.chosen_index
            total_loss -= math.log(max(float(probabilities[chosen]), 1e-12))
            correct += int(int(np.argmax(group)) == chosen)
            score_gradient[offset : offset + size] = probabilities
            score_gradient[offset + chosen] -= 1.0
            offset += size

        scale = np.float32(1.0 / len(decisions))
        score_gradient *= scale
        grad_w2 = hidden.T @ score_gradient
        grad_b2 = np.asarray(score_gradient.sum(), dtype=np.float32).reshape(())
        grad_hidden = score_gradient[:, None] * self.w2[None, :]
        grad_pre_activation = grad_hidden * (1.0 - hidden * hidden)
        grad_w1 = features.T @ grad_pre_activation
        grad_b1 = grad_pre_activation.sum(axis=0)
        gradients = [grad_w1, grad_b1, grad_w2, grad_b2]
        return total_loss, correct, gradients

    def _adam(self, gradients: Sequence[Array], learning_rate: float) -> None:
        self._step += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        norm = math.sqrt(sum(float(np.sum(gradient * gradient)) for gradient in gradients))
        if norm > 5.0:
            gradients = [gradient * np.float32(5.0 / norm) for gradient in gradients]
        parameters = self.parameters
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
            self._m[index] = beta1 * self._m[index] + (1.0 - beta1) * gradient
            self._v[index] = beta2 * self._v[index] + (1.0 - beta2) * gradient * gradient
            m_hat = self._m[index] / (1.0 - beta1**self._step)
            v_hat = self._v[index] / (1.0 - beta2**self._step)
            parameter -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        self.b2 = np.float32(parameters[3])

    def fit(
        self,
        decisions: Sequence[PreparedDecision],
        *,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ) -> list[Metrics]:
        if not decisions:
            raise ValueError("No training decisions were provided")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        rng = np.random.default_rng(self.seed)
        history: list[Metrics] = []
        for _ in range(epochs):
            order = rng.permutation(len(decisions))
            loss = 0.0
            correct = 0
            for start in range(0, len(order), batch_size):
                batch = [decisions[int(index)] for index in order[start : start + batch_size]]
                batch_loss, batch_correct, gradients = self._batch_gradients(batch)
                self._adam(gradients, learning_rate)
                loss += batch_loss
                correct += batch_correct
            history.append(
                Metrics(
                    decisions=len(decisions),
                    loss=loss / len(decisions),
                    top1_accuracy=correct / len(decisions),
                )
            )
        return history

    def evaluate(self, decisions: Sequence[PreparedDecision]) -> Metrics:
        if not decisions:
            return Metrics()
        loss = 0.0
        correct = 0
        reciprocal_rank = 0.0
        random_accuracy = 0.0
        for decision in decisions:
            scores = self.scores(decision.features)
            shifted = scores - np.max(scores)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum()
            chosen = decision.chosen_index
            loss -= math.log(max(float(probabilities[chosen]), 1e-12))
            ranking = np.argsort(-scores)
            rank = int(np.flatnonzero(ranking == chosen)[0]) + 1
            correct += int(rank == 1)
            reciprocal_rank += 1.0 / rank
            random_accuracy += 1.0 / len(scores)
        count = len(decisions)
        return Metrics(
            decisions=count,
            loss=loss / count,
            top1_accuracy=correct / count,
            mean_reciprocal_rank=reciprocal_rank / count,
            random_top1_accuracy=random_accuracy / count,
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        metadata = json.dumps(
            {"schema_version": 1, "dimension": self.dimension, "hidden": self.hidden}
        )
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    metadata=np.asarray(metadata),
                    w1=self.w1,
                    b1=self.b1,
                    w2=self.w2,
                    b2=np.asarray(self.b2),
                )
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "MaskedMLPPolicy":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            model = cls(int(metadata["dimension"]), int(metadata["hidden"]))
            model.w1[...] = data["w1"]
            model.b1[...] = data["b1"]
            model.w2[...] = data["w2"]
            model.b2 = np.float32(data["b2"])
            model._m = [np.zeros_like(value) for value in model.parameters]
            model._v = [np.zeros_like(value) for value in model.parameters]
            return model


def train_policy(
    dataset: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    dimension: int = 512,
    hidden: int = 64,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 7,
) -> dict[str, Any]:
    hasher = FeatureHasher(dimension)
    train = prepare_records(read_dataset(dataset, "train"), hasher)
    validation = prepare_records(read_dataset(dataset, "validation"), hasher)
    model = MaskedMLPPolicy(dimension, hidden, seed)
    history = model.fit(
        train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )
    model.save(output)
    return {
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


def evaluate_policy(
    dataset: str | os.PathLike[str], model_path: str | os.PathLike[str], split: str = "test"
) -> dict[str, Any]:
    model = MaskedMLPPolicy.load(model_path)
    hasher = FeatureHasher(model.dimension)
    decisions = prepare_records(read_dataset(dataset, split), hasher)
    overall = model.evaluate(decisions)
    patches: dict[str, Any] = {}
    for patch in sorted({decision.patch for decision in decisions}):
        patches[patch] = model.evaluate(
            [decision for decision in decisions if decision.patch == patch]
        ).as_dict()
    decision_types: dict[str, Any] = {}
    for decision_type in sorted({decision.decision_type for decision in decisions}):
        decision_types[decision_type] = model.evaluate(
            [decision for decision in decisions if decision.decision_type == decision_type]
        ).as_dict()
    characters: dict[str, Any] = {}
    for character in sorted({decision.character for decision in decisions}):
        characters[character] = model.evaluate(
            [decision for decision in decisions if decision.character == character]
        ).as_dict()
    return {
        "split": split,
        "overall": overall.as_dict(),
        "by_patch": patches,
        "by_decision_type": decision_types,
        "by_character": characters,
    }
