"""Compact typed Deep Sets scorer for dynamic native combat actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from stsai.combat import (
    action_key,
    headless_action_to_combat,
    normalize_headless_combat_observation,
    safe_potion_indexes,
)
from stsai.headless import HeadlessConfig
from stsai.headless_env import Sts2Env, run_headless_rollout
from stsai.headless_policy import HeadlessHybridPolicy
from stsai.policy import MaskedMLPPolicy, read_dataset


GROUP_NAMES = ("context", "hand", "enemies", "support")
GLOBAL_NUMERIC = 16
ENTITY_NUMERIC = 8


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _objects(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ratio(value: Any, maximum: Any) -> float:
    return _number(value) / max(_number(maximum), 1.0)


def _by_int(
    items: Sequence[Mapping[str, Any]], field: str, value: Any
) -> Mapping[str, Any]:
    try:
        target = int(value)
    except (TypeError, ValueError):
        return {}
    return next(
        (
            item
            for item in items
            if item.get(field) is not None and int(item[field]) == target
        ),
        {},
    )


def _token_id(token: str, buckets: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


@dataclass(frozen=True)
class SetEntity:
    tokens: tuple[int, ...]
    numeric: tuple[float, ...]


@dataclass(frozen=True)
class PreparedSetDecision:
    global_numeric: tuple[float, ...]
    groups: tuple[tuple[SetEntity, ...], ...]
    actions: tuple[SetEntity, ...]
    chosen_index: int
    character: str
    action_type: str


@dataclass(frozen=True)
class SetPolicyConfig:
    hash_buckets: int = 8192
    embedding_dim: int = 32
    numeric_hidden: int = 32
    entity_hidden: int = 64
    global_hidden: int = 64
    score_hidden: int = 128


def _entity(
    tokens: Iterable[str],
    numeric: Sequence[float],
    *,
    buckets: int,
) -> SetEntity:
    token_ids = tuple(sorted({_token_id(token, buckets) for token in tokens}))
    if not token_ids:
        token_ids = (_token_id("entity=unknown", buckets),)
    values = tuple(float(value) for value in numeric[:ENTITY_NUMERIC])
    values += (0.0,) * (ENTITY_NUMERIC - len(values))
    return SetEntity(token_ids, values)


def _status_tokens(prefix: str, statuses: Any) -> list[str]:
    return [
        f"{prefix}_status={status.get('id')}"
        for status in _objects(statuses)
        if status.get("id")
    ]


def _card_entity(card: Mapping[str, Any], *, buckets: int) -> SetEntity:
    tokens = [
        "entity=card",
        f"card_id={card.get('id')}",
        f"card_type={card.get('type')}",
        f"card_target={card.get('target_type')}",
        f"card_upgraded={card.get('is_upgraded')}",
    ]
    numeric = (
        _number(card.get("cost")) / 5.0,
        _number(card.get("star_cost")) / 10.0,
        float(card.get("is_upgraded") is True),
        float(card.get("can_play") is True),
    )
    return _entity(tokens, numeric, buckets=buckets)


def _enemy_entity(enemy: Mapping[str, Any], *, buckets: int) -> SetEntity:
    intents = _objects(enemy.get("intents"))
    tokens = [
        "entity=enemy",
        f"enemy_id={enemy.get('id', enemy.get('entity_id'))}",
        *(f"enemy_intent={intent.get('type')}" for intent in intents),
        *_status_tokens("enemy", enemy.get("status")),
    ]
    intent_damage = sum(
        _number(intent.get("damage")) * max(_number(intent.get("hits"), 1.0), 1.0)
        for intent in intents
    )
    numeric = (
        _ratio(enemy.get("hp"), enemy.get("max_hp")),
        _ratio(enemy.get("block"), enemy.get("max_hp")),
        min(intent_damage / 100.0, 3.0),
        min(len(intents) / 4.0, 2.0),
        min(len(_objects(enemy.get("status"))) / 8.0, 2.0),
    )
    return _entity(tokens, numeric, buckets=buckets)


def _support_entities(player: Mapping[str, Any], *, buckets: int) -> list[SetEntity]:
    entities: list[SetEntity] = []
    for relic in _objects(player.get("relics")):
        entities.append(
            _entity(
                ("entity=relic", f"relic_id={relic.get('id')}"),
                (_number(relic.get("counter")) / 20.0,),
                buckets=buckets,
            )
        )
    for potion in _objects(player.get("potions")):
        entities.append(
            _entity(
                (
                    "entity=potion",
                    f"potion_id={potion.get('id')}",
                    f"potion_target={potion.get('target_type')}",
                ),
                (
                    _number(potion.get("slot")) / 5.0,
                    float(potion.get("can_use_in_combat") is True),
                ),
                buckets=buckets,
            )
        )
    for status in _objects(player.get("status")):
        entities.append(
            _entity(
                ("entity=player_status", f"player_status={status.get('id')}"),
                (_number(status.get("amount")) / 20.0,),
                buckets=buckets,
            )
        )
    for orb in _objects(player.get("orbs")):
        entities.append(
            _entity(
                ("entity=orb", f"orb_id={orb.get('id')}"),
                (
                    _number(orb.get("passive_val")) / 20.0,
                    _number(orb.get("evoke_val")) / 20.0,
                ),
                buckets=buckets,
            )
        )
    for pet in _objects(player.get("pets")):
        entities.append(
            _entity(
                (
                    "entity=pet",
                    f"pet_id={pet.get('id', pet.get('entity_id'))}",
                    *_status_tokens("pet", pet.get("status")),
                ),
                (
                    _ratio(pet.get("hp"), pet.get("max_hp")),
                    _ratio(pet.get("block"), pet.get("max_hp")),
                ),
                buckets=buckets,
            )
        )
    return entities


def _action_entity(
    action: Mapping[str, Any],
    *,
    hand: Sequence[Mapping[str, Any]],
    enemies: Sequence[Mapping[str, Any]],
    potions: Sequence[Mapping[str, Any]],
    buckets: int,
) -> SetEntity:
    name = str(action.get("action", "unknown"))
    tokens = ["entity=action", f"action={name}"]
    numeric = [0.0] * ENTITY_NUMERIC
    candidate: Mapping[str, Any] = {}
    if name == "play_card":
        candidate = _by_int(hand, "index", action.get("card_index"))
        tokens.extend(
            (
                f"action_card={candidate.get('id')}",
                f"action_card_type={candidate.get('type')}",
                f"action_card_target={candidate.get('target_type')}",
                f"action_card_upgraded={candidate.get('is_upgraded')}",
            )
        )
        numeric[0] = _number(candidate.get("cost")) / 5.0
        numeric[1] = _number(candidate.get("star_cost")) / 10.0
        numeric[2] = float(candidate.get("is_upgraded") is True)
    elif name == "use_potion":
        candidate = _by_int(potions, "slot", action.get("slot"))
        tokens.extend(
            (
                f"action_potion={candidate.get('id')}",
                f"action_potion_target={candidate.get('target_type')}",
            )
        )
        numeric[0] = _number(candidate.get("slot")) / 5.0

    target_value = action.get("target")
    target = (
        next(
            (
                enemy
                for enemy in enemies
                if str(enemy.get("entity_id")) == str(target_value)
                or (
                    enemy.get("combat_id") is not None
                    and str(enemy.get("combat_id")) == str(target_value)
                )
            ),
            {},
        )
        if target_value is not None
        else {}
    )
    if target:
        tokens.extend(
            (
                f"action_target={target.get('id', target.get('entity_id'))}",
                "action_target_side=enemy",
                *_status_tokens("action_target", target.get("status")),
            )
        )
        numeric[3] = _ratio(target.get("hp"), target.get("max_hp"))
        numeric[4] = _ratio(target.get("block"), target.get("max_hp"))
        numeric[5] = 1.0
    elif target_value is not None:
        tokens.extend((f"action_target={target_value}", "action_target_side=ally"))
        numeric[5] = -1.0
    return _entity(tokens, numeric, buckets=buckets)


def prepare_set_record(
    record: Mapping[str, Any], *, hash_buckets: int = 8192
) -> PreparedSetDecision | None:
    actions = _objects(record.get("legal_actions"))
    chosen = _mapping(record.get("chosen_action"))
    if not actions or not chosen:
        return None
    keys = [action_key(action) for action in actions]
    chosen_key = action_key(chosen)
    if chosen_key not in keys:
        return None

    observation = _mapping(record.get("observation"))
    run = _mapping(observation.get("run"))
    player = _mapping(observation.get("player"))
    battle = _mapping(observation.get("battle"))
    hand = _objects(player.get("hand"))
    enemies = _objects(battle.get("enemies"))
    alive_enemies = [enemy for enemy in enemies if _number(enemy.get("hp")) > 0]
    potions = _objects(player.get("potions"))
    max_hp = max(_number(player.get("max_hp")), 1.0)
    max_energy = max(_number(player.get("max_energy")), 1.0)
    global_numeric = (
        _number(player.get("hp")) / max_hp,
        min(_number(player.get("block")) / max_hp, 2.0),
        _number(player.get("energy")) / max_energy,
        min(_number(player.get("stars")) / 10.0, 2.0),
        min(_number(run.get("act")) / 4.0, 1.5),
        min(_number(run.get("floor")) / 60.0, 1.5),
        min(_number(run.get("ascension")) / 10.0, 2.0),
        min(_number(battle.get("round")) / 20.0, 2.0),
        min(len(alive_enemies) / 5.0, 2.0),
        min(sum(_number(enemy.get("hp")) for enemy in alive_enemies) / 300.0, 3.0),
        min(sum(_number(enemy.get("block")) for enemy in alive_enemies) / 100.0, 3.0),
        min(len(hand) / 10.0, 2.0),
        min(_number(player.get("draw_pile_count")) / 30.0, 2.0),
        min(_number(player.get("discard_pile_count")) / 30.0, 2.0),
        min(_number(player.get("exhaust_pile_count")) / 30.0, 2.0),
        min(len(potions) / 5.0, 2.0),
    )
    character = str(player.get("character", "unknown"))
    build_id = str(_mapping(record.get("patch")).get("build_id", "unknown"))
    context = _entity(
        (
            "entity=context",
            f"character={character}",
            f"build={build_id}",
            f"state={observation.get('state_type')}",
        ),
        (),
        buckets=hash_buckets,
    )
    groups = (
        (context,),
        tuple(_card_entity(card, buckets=hash_buckets) for card in hand),
        tuple(_enemy_entity(enemy, buckets=hash_buckets) for enemy in alive_enemies),
        tuple(_support_entities(player, buckets=hash_buckets)),
    )
    prepared_actions = tuple(
        _action_entity(
            action,
            hand=hand,
            enemies=alive_enemies,
            potions=potions,
            buckets=hash_buckets,
        )
        for action in actions
    )
    return PreparedSetDecision(
        global_numeric=global_numeric,
        groups=groups,
        actions=prepared_actions,
        chosen_index=keys.index(chosen_key),
        character=character,
        action_type=str(chosen.get("action", "unknown")),
    )


def prepare_set_records(
    records: Iterable[Mapping[str, Any]], *, hash_buckets: int = 8192
) -> list[PreparedSetDecision]:
    return [
        prepared
        for record in records
        if (prepared := prepare_set_record(record, hash_buckets=hash_buckets)) is not None
    ]


class DeepSetCombatPolicy(nn.Module):
    """Encode typed entity sets once, then score every current legal action."""

    def __init__(self, config: SetPolicyConfig = SetPolicyConfig()):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.hash_buckets, config.embedding_dim)
        self.numeric_encoder = nn.Sequential(
            nn.Linear(ENTITY_NUMERIC, config.numeric_hidden),
            nn.ReLU(),
        )
        self.entity_encoder = nn.Sequential(
            nn.Linear(config.embedding_dim + config.numeric_hidden, config.entity_hidden),
            nn.ReLU(),
            nn.Linear(config.entity_hidden, config.entity_hidden),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_NUMERIC, config.global_hidden),
            nn.ReLU(),
            nn.Linear(config.global_hidden, config.global_hidden),
            nn.ReLU(),
        )
        state_width = config.global_hidden + len(GROUP_NAMES) * config.entity_hidden
        self.score_head = nn.Sequential(
            nn.Linear(state_width + config.entity_hidden, config.score_hidden),
            nn.ReLU(),
            nn.Linear(config.score_hidden, 1),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _encode_entities(
        self, entities: Sequence[SetEntity], device: torch.device
    ) -> torch.Tensor:
        if not entities:
            return torch.zeros((0, self.config.entity_hidden), device=device)
        token_ids: list[int] = []
        owners: list[int] = []
        for index, entity in enumerate(entities):
            token_ids.extend(entity.tokens)
            owners.extend([index] * len(entity.tokens))
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        owner_tensor = torch.tensor(owners, dtype=torch.long, device=device)
        embedded = self.embedding(token_tensor)
        pooled = torch.zeros(
            (len(entities), self.config.embedding_dim), device=device
        )
        pooled.index_add_(0, owner_tensor, embedded)
        counts = torch.bincount(owner_tensor, minlength=len(entities)).clamp_min(1)
        pooled = pooled / counts.unsqueeze(1)
        numeric = torch.tensor(
            [entity.numeric for entity in entities], dtype=torch.float32, device=device
        )
        return self.entity_encoder(
            torch.cat((pooled, self.numeric_encoder(numeric)), dim=1)
        )

    def forward(
        self, decisions: Sequence[PreparedSetDecision]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not decisions:
            raise ValueError("Deep Sets batch cannot be empty")
        device = self.embedding.weight.device
        global_values = torch.tensor(
            [decision.global_numeric for decision in decisions],
            dtype=torch.float32,
            device=device,
        )
        state_parts = [self.global_encoder(global_values)]
        for group_index in range(len(GROUP_NAMES)):
            entities: list[SetEntity] = []
            owners: list[int] = []
            for decision_index, decision in enumerate(decisions):
                group = decision.groups[group_index]
                entities.extend(group)
                owners.extend([decision_index] * len(group))
            pooled = torch.zeros(
                (len(decisions), self.config.entity_hidden), device=device
            )
            if entities:
                encoded = self._encode_entities(entities, device)
                owner_tensor = torch.tensor(owners, dtype=torch.long, device=device)
                pooled.index_add_(0, owner_tensor, encoded)
                counts = torch.bincount(
                    owner_tensor, minlength=len(decisions)
                ).clamp_min(1)
                pooled = pooled / counts.unsqueeze(1)
            state_parts.append(pooled)
        state = torch.cat(state_parts, dim=1)

        actions: list[SetEntity] = []
        action_owners: list[int] = []
        action_positions: list[int] = []
        for decision_index, decision in enumerate(decisions):
            actions.extend(decision.actions)
            action_owners.extend([decision_index] * len(decision.actions))
            action_positions.extend(range(len(decision.actions)))
        encoded_actions = self._encode_entities(actions, device)
        owner_tensor = torch.tensor(action_owners, dtype=torch.long, device=device)
        flat_scores = self.score_head(
            torch.cat((state[owner_tensor], encoded_actions), dim=1)
        ).squeeze(1)
        max_actions = max(len(decision.actions) for decision in decisions)
        scores = torch.full(
            (len(decisions), max_actions), -1e9, dtype=torch.float32, device=device
        )
        position_tensor = torch.tensor(action_positions, dtype=torch.long, device=device)
        scores[owner_tensor, position_tensor] = flat_scores
        mask = scores > -1e8
        return scores, mask

    def save(self, path: str | os.PathLike[str]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            with temporary.open("wb") as handle:
                torch.save(
                    {
                        "schema_version": 1,
                        "model_type": "typed_deep_sets_candidate_scorer",
                        "config": asdict(self.config),
                        "state_dict": self.state_dict(),
                    },
                    handle,
                )
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> DeepSetCombatPolicy:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("model_type") != "typed_deep_sets_candidate_scorer":
            raise ValueError("Not an STSAI typed Deep Sets checkpoint")
        model = cls(SetPolicyConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model


class HeadlessSetCombatPolicy:
    """Use typed Deep Sets in combat and the exact heuristic elsewhere."""

    def __init__(
        self,
        macro_model: MaskedMLPPolicy,
        combat_model: DeepSetCombatPolicy,
        *,
        build_id: str = "v0.107.1",
        safe_potion_ids: frozenset[str] | None = None,
    ):
        self.build_id = build_id
        self.combat_model = combat_model.eval()
        self.safe_potion_ids = safe_potion_ids
        self.fallback = HeadlessHybridPolicy(
            macro_model, build_id=build_id, use_macro=False
        )

    def select(self, env: Sts2Env, step: int = 0) -> tuple[int, Mapping[str, Any]]:
        if str(env.raw_state.get("decision", "")) != "combat_play":
            return self.fallback.select(env, step)
        actions = [
            headless_action_to_combat(action.as_dict())
            for action in env.legal_actions
        ]
        record = {
            "decision_type": "combat_action",
            "patch": {"build_id": self.build_id},
            "observation": normalize_headless_combat_observation(env.observation),
            "legal_actions": actions,
            "chosen_action": actions[0],
        }
        prepared = prepare_set_record(
            record, hash_buckets=self.combat_model.config.hash_buckets
        )
        if prepared is None:
            raise ValueError("Unable to prepare live Deep Sets combat candidates")
        with torch.no_grad():
            scores, _ = self.combat_model([prepared])
            probabilities = torch.softmax(scores[0, : len(actions)], dim=0)
        order = torch.argsort(scores[0, : len(actions)], descending=True).tolist()
        eligible_order = order
        if self.safe_potion_ids is not None:
            allowed = safe_potion_indexes(
                _mapping(env.observation.get("player")), self.safe_potion_ids
            )
            eligible_order = [
                index
                for index in order
                if env.legal_actions[int(index)].command != "use_potion"
                or int(
                    env.legal_actions[int(index)]
                    .argument_dict()
                    .get("potion_index", -1)
                )
                in allowed
            ]
        if not eligible_order:
            raise ValueError("No runtime-safe Deep Sets combat action is available")
        best = int(eligible_order[0])
        return best, {
            "source": "combat_deepsets",
            "model_type": "typed_deep_sets_candidate_scorer",
            "character": record["observation"]["player"]["character"],
            "ranked_actions": [
                {
                    "action_index": int(index),
                    "action": env.legal_actions[int(index)].as_dict(),
                    "score": round(float(scores[0, int(index)]), 6),
                    "probability": round(float(probabilities[int(index)]), 6),
                }
                for index in eligible_order[:5]
            ],
        }


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_set_combat_rollout(
    *,
    macro_model_path: str | Path,
    combat_model_path: str | Path,
    output: str | Path,
    character: str,
    ascension: int,
    seed: str,
    max_steps: int = 1_000,
    build_id: str = "v0.107.1",
    safe_potion_ids: Sequence[str] | None = None,
    config: HeadlessConfig | None = None,
) -> dict[str, Any]:
    """Run typed Deep Sets only in combat and the exact heuristic elsewhere."""

    macro_model_path = Path(macro_model_path)
    combat_model_path = Path(combat_model_path)
    policy = HeadlessSetCombatPolicy(
        MaskedMLPPolicy.load(macro_model_path),
        DeepSetCombatPolicy.load(combat_model_path),
        build_id=build_id,
        safe_potion_ids=(
            None if safe_potion_ids is None else frozenset(safe_potion_ids)
        ),
    )
    return run_headless_rollout(
        output=output,
        character=character,
        ascension=ascension,
        seed=seed,
        policy_name="combat_deepsets_greedy_noncombat_v1",
        select_action=policy.select,
        policy_metadata={
            "combat_model": str(combat_model_path.resolve()),
            "combat_model_sha256": _sha256_file(combat_model_path),
            "combat_model_type": "typed_deep_sets_candidate_scorer",
            "macro_model_loaded_for_interface_only": str(macro_model_path.resolve()),
            "macro_model_sha256": _sha256_file(macro_model_path),
            "build_id": build_id,
            "learned_decisions": ["combat_play"],
            "label_source": "dagger_heuristic_teacher",
            "noncombat_policy": "greedy_rules_v1",
            "safe_potion_ids": (
                "all" if safe_potion_ids is None else sorted(safe_potion_ids)
            ),
        },
        max_steps=max_steps,
        config=config,
    )


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


@torch.no_grad()
def evaluate_prepared_set_policy(
    model: DeepSetCombatPolicy,
    decisions: Sequence[PreparedSetDecision],
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    if not decisions:
        return {
            "decisions": 0,
            "loss": 0.0,
            "top1_accuracy": 0.0,
            "mean_reciprocal_rank": 0.0,
            "random_top1_accuracy": 0.0,
        }
    model.eval()
    loss = 0.0
    correct = 0
    reciprocal_rank = 0.0
    random_accuracy = 0.0
    for start in range(0, len(decisions), batch_size):
        batch = decisions[start : start + batch_size]
        scores, _ = model(batch)
        labels = torch.tensor(
            [decision.chosen_index for decision in batch],
            dtype=torch.long,
            device=scores.device,
        )
        loss += float(nn.functional.cross_entropy(scores, labels, reduction="sum"))
        ranking = torch.argsort(scores, dim=1, descending=True)
        ranks = (ranking == labels.unsqueeze(1)).nonzero()[:, 1] + 1
        correct += int((ranks == 1).sum())
        reciprocal_rank += float((1.0 / ranks.float()).sum())
        random_accuracy += sum(1.0 / len(decision.actions) for decision in batch)
    count = len(decisions)
    return {
        "decisions": count,
        "loss": round(loss / count, 6),
        "top1_accuracy": round(correct / count, 6),
        "mean_reciprocal_rank": round(reciprocal_rank / count, 6),
        "random_top1_accuracy": round(random_accuracy / count, 6),
    }


def evaluate_set_policy(
    dataset: str | os.PathLike[str],
    model_path: str | os.PathLike[str],
    *,
    split: str = "test",
    batch_size: int = 128,
    output: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    model = DeepSetCombatPolicy.load(model_path)
    decisions = prepare_set_records(
        read_dataset(dataset, split), hash_buckets=model.config.hash_buckets
    )
    result = {
        "split": split,
        "model_type": "typed_deep_sets_candidate_scorer",
        "parameter_count": model.parameter_count,
        "overall": evaluate_prepared_set_policy(model, decisions, batch_size=batch_size),
        "by_character": {
            character: evaluate_prepared_set_policy(
                model,
                [decision for decision in decisions if decision.character == character],
                batch_size=batch_size,
            )
            for character in sorted({decision.character for decision in decisions})
        },
        "by_action": {
            action: evaluate_prepared_set_policy(
                model,
                [decision for decision in decisions if decision.action_type == action],
                batch_size=batch_size,
            )
            for action in sorted({decision.action_type for decision in decisions})
        },
    }
    if output is not None:
        _atomic_json(output, result)
    return result


def train_set_policy(
    dataset: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    metrics_output: str | os.PathLike[str] | None = None,
    epochs: int = 12,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 7,
    config: SetPolicyConfig = SetPolicyConfig(),
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("Epochs and batch size must be positive")
    torch.manual_seed(seed)
    model = DeepSetCombatPolicy(config)
    train = prepare_set_records(
        read_dataset(dataset, "train"), hash_buckets=config.hash_buckets
    )
    validation = prepare_set_records(
        read_dataset(dataset, "validation"), hash_buckets=config.hash_buckets
    )
    if not train or not validation:
        raise ValueError("Deep Sets training requires non-empty train and validation splits")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        epoch_loss = 0.0
        correct = 0
        for start in range(0, len(order), batch_size):
            batch = [train[index] for index in order[start : start + batch_size]]
            scores, _ = model(batch)
            labels = torch.tensor(
                [decision.chosen_index for decision in batch], dtype=torch.long
            )
            loss = nn.functional.cross_entropy(scores, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)
            correct += int((scores.detach().argmax(dim=1) == labels).sum())
        history.append(
            {
                "epoch": epoch + 1,
                "decisions": len(train),
                "loss": round(epoch_loss / len(train), 6),
                "top1_accuracy": round(correct / len(train), 6),
            }
        )
    model.save(output)
    result = {
        "model": str(Path(output)),
        "model_type": "typed_deep_sets_candidate_scorer",
        "parameter_count": model.parameter_count,
        "configuration": {
            **asdict(config),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
        },
        "train_final": history[-1],
        "validation": evaluate_prepared_set_policy(
            model, validation, batch_size=batch_size
        ),
    }
    if metrics_output is not None:
        _atomic_json(metrics_output, result)
    return result
