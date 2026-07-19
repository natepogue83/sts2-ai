"""Leakage-aware process-fork search interfaces for the native STS2 runtime.

This module deliberately does not create training labels.  Its first search mode
is a correctness harness: one ply, sequential fresh processes, and a conservative
allowlist of card plays whose immediate result depends only on public state.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Protocol

from stsai.headless_env import HeadlessAction, Sts2Env


PUBLIC_DETERMINISTIC_MODE = "public_deterministic_only"


class SelectionPolicy(Protocol):
    def select(self, env: Sts2Env, step: int = 0) -> tuple[int, Mapping[str, Any]]:
        """Select one index aligned to ``env.legal_actions``."""


@dataclass(frozen=True)
class PolicyValueEstimate:
    """Future joint-head boundary; policy logits may be absent for leaf-only evaluators."""

    value: float
    policy_logits: tuple[float, ...] | None = None
    metadata: Mapping[str, Any] | None = None


class PolicyValueEvaluator(Protocol):
    def evaluate(self, env: Sts2Env) -> PolicyValueEstimate:
        """Evaluate only information intentionally exposed by the environment."""


@dataclass(frozen=True)
class OnePlySearchConfig:
    visibility_mode: str = PUBLIC_DETERMINISTIC_MODE
    max_nodes: int = 32

    def __post_init__(self) -> None:
        if self.visibility_mode != PUBLIC_DETERMINISTIC_MODE:
            raise ValueError(
                "The first search milestone only supports public_deterministic_only"
            )
        if self.max_nodes <= 0:
            raise ValueError("max_nodes must be positive")


def public_deterministic_action(action: HeadlessAction) -> tuple[bool, str]:
    """Conservatively allow only immediate, non-drawing basic card effects.

    Strike, Defend, Bash, and Neutralize do not consume hidden RNG, reveal the draw
    pile, or open a subordinate choice.  Everything else remains excluded until it
    has an explicit effect-level proof rather than a guessed card-name heuristic.
    """

    if action.command == "end_turn":
        return False, "end_turn_reveals_hidden_draw_and_enemy_rng"
    if action.command != "play_card":
        return False, "action_type_not_public_deterministic_allowlisted"
    card_id = action.option_id.split("|", 1)[0].upper()
    entry = card_id.rsplit(".", 1)[-1]
    if entry.startswith("STRIKE_") or entry.startswith("DEFEND_"):
        return True, "basic_card_effect_allowlisted"
    if entry in {"BASH", "NEUTRALIZE"}:
        return True, "basic_card_effect_allowlisted"
    return False, "card_effect_not_public_deterministic_proven"


class PublicCombatHeuristicEvaluator:
    """Untrained public-state leaf value used only to validate search plumbing."""

    name = "public_combat_heuristic_v1"

    def evaluate(self, env: Sts2Env) -> PolicyValueEstimate:
        state = env.raw_state
        if state.get("decision") == "game_over":
            return PolicyValueEstimate(
                value=1.0 if state.get("victory") is True else -1.0,
                metadata={"source": self.name, "terminal": True},
            )
        if state.get("decision") != "combat_play":
            raise ValueError(
                "Public combat heuristic only evaluates combat_play or game_over"
            )

        player = state.get("player")
        player = player if isinstance(player, Mapping) else {}
        hp = float(player.get("hp") or 0.0)
        max_hp = max(float(player.get("max_hp") or 1.0), 1.0)
        block = max(float(player.get("block") or 0.0), 0.0)
        hp_fraction = hp / max_hp
        block_fraction = min(block / max_hp, 1.0)

        enemies = [
            enemy
            for enemy in (state.get("enemies") or [])
            if isinstance(enemy, Mapping) and float(enemy.get("hp") or 0.0) > 0
        ]
        total_hp = sum(float(enemy.get("hp") or 0.0) for enemy in enemies)
        total_max_hp = max(
            sum(float(enemy.get("max_hp") or enemy.get("hp") or 0.0) for enemy in enemies),
            1.0,
        )
        damage_progress = 1.0 - total_hp / total_max_hp
        alive_penalty = min(len(enemies) / 5.0, 1.0)
        value = (
            0.45 * hp_fraction
            + 0.15 * block_fraction
            + 0.45 * damage_progress
            - 0.05 * alive_penalty
        )
        return PolicyValueEstimate(
            value=float(value),
            metadata={
                "source": self.name,
                "terminal": False,
                "trained": False,
                "uses_hidden_state": False,
            },
        )


class OnePlyForkSearchPolicy:
    """Wrap an existing policy with conservative, process-isolated one-ply search."""

    def __init__(
        self,
        base_policy: SelectionPolicy,
        evaluator: PolicyValueEvaluator,
        *,
        config: OnePlySearchConfig = OnePlySearchConfig(),
    ):
        self.base_policy = base_policy
        self.evaluator = evaluator
        self.config = config

    def select(self, env: Sts2Env, step: int = 0) -> tuple[int, Mapping[str, Any]]:
        if getattr(self.evaluator, "search_eligible", True) is not True:
            raise RuntimeError(
                "Value evaluator did not pass its held-out search-eligibility gate"
            )
        base_index, base_info = self.base_policy.select(env, step)
        if base_index < 0 or base_index >= len(env.legal_actions):
            raise IndexError("Base policy selected an illegal action")
        if env.raw_state.get("decision") != "combat_play":
            return base_index, {
                **dict(base_info),
                "search": {
                    "used": False,
                    "reason": "noncombat_fallback",
                    "visibility_mode": self.config.visibility_mode,
                    "training_labels_emitted": False,
                },
            }

        base_action = env.legal_actions[base_index]
        base_safe, base_reason = public_deterministic_action(base_action)
        safety = [public_deterministic_action(action) for action in env.legal_actions]
        if not base_safe:
            return base_index, {
                **dict(base_info),
                "search": {
                    "used": False,
                    "reason": "base_action_crosses_information_boundary",
                    "base_action_safety": base_reason,
                    "visibility_mode": self.config.visibility_mode,
                    "training_labels_emitted": False,
                },
            }

        safe_indices = [index for index, (allowed, _) in enumerate(safety) if allowed]
        safe_indices.sort(key=lambda index: (index != base_index, index))
        safe_indices = safe_indices[: self.config.max_nodes]
        root_snapshot = env.snapshot()
        parent_state = json.dumps(
            env.raw_state, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        parent_actions = env.legal_actions
        evaluated: list[dict[str, Any]] = []
        for action_index in safe_indices:
            action = env.legal_actions[action_index]
            with env.fork(root_snapshot) as branch:
                try:
                    branch_index = branch.legal_actions.index(action)
                except ValueError as exc:
                    raise RuntimeError(
                        "Exact fork did not preserve the root legal action identity"
                    ) from exc
                result = branch.step(branch_index)
                estimate = self.evaluator.evaluate(branch)
                evaluated.append(
                    {
                        "action_index": action_index,
                        "action": action.as_dict(),
                        "leaf_value": float(estimate.value),
                        "terminated": result.terminated,
                        "evaluator": dict(estimate.metadata or {}),
                    }
                )

        current_parent_state = json.dumps(
            env.raw_state, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        if current_parent_state != parent_state or env.legal_actions != parent_actions:
            raise RuntimeError("Search branches mutated the parent environment")
        if not evaluated:
            raise RuntimeError("Public-deterministic search had no evaluable branches")

        chosen = max(
            evaluated,
            key=lambda item: (
                float(item["leaf_value"]),
                int(item["action_index"] == base_index),
                -int(item["action_index"]),
            ),
        )
        chosen_index = int(chosen["action_index"])
        excluded = [
            {
                "action_index": index,
                "action": env.legal_actions[index].as_dict(),
                "reason": reason,
            }
            for index, (allowed, reason) in enumerate(safety)
            if not allowed
        ]
        return chosen_index, {
            "source": "public_deterministic_one_ply_search",
            "base_policy": dict(base_info),
            "search": {
                "used": True,
                "depth": 1,
                "nodes": len(evaluated),
                "visibility_mode": self.config.visibility_mode,
                "selected_action_index": chosen_index,
                "base_action_index": base_index,
                "branches": evaluated,
                "excluded_actions": excluded,
                "training_labels_emitted": False,
            },
        }
