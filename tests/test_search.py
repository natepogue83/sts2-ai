from __future__ import annotations

import copy
from dataclasses import dataclass
import unittest

from stsai.headless_env import HeadlessAction, StepResult
from stsai.search import (
    OnePlyForkSearchPolicy,
    PolicyValueEstimate,
    PublicCombatHeuristicEvaluator,
    public_deterministic_action,
)


STRIKE = HeadlessAction.create(
    "play_card", "CARD.STRIKE_IRONCLAD", card_index=0, target_index=0
)
DEFEND = HeadlessAction.create("play_card", "CARD.DEFEND_IRONCLAD", card_index=1)
END_TURN = HeadlessAction.create("end_turn", "END_TURN")


class FixedPolicy:
    def __init__(self, index: int):
        self.index = index

    def select(self, env, step=0):
        return self.index, {"source": "fixed", "step": step}


@dataclass(frozen=True)
class FakeSnapshot:
    state: dict
    hidden_rng: int


class FakeTreeEnv:
    def __init__(self, *, hidden_rng: int = 0, actions=(STRIKE, DEFEND, END_TURN)):
        self._raw_state = {
            "type": "decision",
            "decision": "combat_play",
            "player": {"hp": 50, "max_hp": 50, "block": 0},
            "enemies": [{"index": 0, "hp": 10, "max_hp": 10}],
        }
        self._legal_actions = tuple(actions)
        self.hidden_rng = hidden_rng
        self.fork_count = 0
        self.closed = False

    @property
    def raw_state(self):
        return self._raw_state

    @property
    def legal_actions(self):
        return self._legal_actions

    def snapshot(self):
        return FakeSnapshot(copy.deepcopy(self._raw_state), self.hidden_rng)

    def fork(self, snapshot):
        self.fork_count += 1
        child = FakeTreeEnv(hidden_rng=snapshot.hidden_rng, actions=self._legal_actions)
        child._raw_state = copy.deepcopy(snapshot.state)
        return child

    def step(self, action_index):
        action = self._legal_actions[action_index]
        if action == STRIKE:
            self._raw_state["enemies"][0]["hp"] -= 6
        elif action == DEFEND:
            self._raw_state["player"]["block"] += 5
        else:
            raise AssertionError("Unsafe end_turn must not be searched")
        return StepResult({}, 0.0, False, False, {})

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class StubValueEvaluator:
    def evaluate(self, env):
        hp = env.raw_state["enemies"][0]["hp"]
        return PolicyValueEstimate(
            value=-float(hp),
            metadata={"source": "stub", "hidden_rng_read": False},
        )


class RejectedValueEvaluator(StubValueEvaluator):
    search_eligible = False


class SearchTests(unittest.TestCase):
    def test_guard_only_allows_proven_basic_card_effects(self) -> None:
        self.assertTrue(public_deterministic_action(STRIKE)[0])
        self.assertTrue(
            public_deterministic_action(
                HeadlessAction.create(
                    "play_card",
                    "CARD.STRIKE_SILENT|TARGET.2",
                    card_index=0,
                    target_index=2,
                )
            )[0]
        )
        self.assertTrue(public_deterministic_action(DEFEND)[0])
        self.assertFalse(public_deterministic_action(END_TURN)[0])
        self.assertFalse(
            public_deterministic_action(
                HeadlessAction.create(
                    "play_card", "CARD.POMMEL_STRIKE", card_index=2, target_index=0
                )
            )[0]
        )

    def test_one_ply_search_selects_best_safe_branch_without_mutating_parent(self) -> None:
        env = FakeTreeEnv(hidden_rng=17)
        before = copy.deepcopy(env.raw_state)
        policy = OnePlyForkSearchPolicy(FixedPolicy(1), StubValueEvaluator())

        selected, info = policy.select(env, step=4)

        self.assertEqual(selected, 0)
        self.assertEqual(env.raw_state, before)
        self.assertEqual(env.fork_count, 2)
        self.assertTrue(info["search"]["used"])
        self.assertEqual(info["search"]["nodes"], 2)
        self.assertFalse(info["search"]["training_labels_emitted"])
        self.assertEqual(
            info["search"]["excluded_actions"][0]["reason"],
            "end_turn_reveals_hidden_draw_and_enemy_rng",
        )

    def test_hidden_rng_does_not_change_public_deterministic_choice(self) -> None:
        policy = OnePlyForkSearchPolicy(FixedPolicy(1), StubValueEvaluator())
        first, _ = policy.select(FakeTreeEnv(hidden_rng=1))
        second, _ = policy.select(FakeTreeEnv(hidden_rng=999_999))
        self.assertEqual(first, second)

    def test_unsafe_base_action_falls_back_without_opening_forks(self) -> None:
        env = FakeTreeEnv()
        selected, info = OnePlyForkSearchPolicy(
            FixedPolicy(2), StubValueEvaluator()
        ).select(env)
        self.assertEqual(selected, 2)
        self.assertEqual(env.fork_count, 0)
        self.assertFalse(info["search"]["used"])
        self.assertEqual(
            info["search"]["reason"],
            "base_action_crosses_information_boundary",
        )

    def test_public_leaf_evaluator_ignores_hidden_rng(self) -> None:
        evaluator = PublicCombatHeuristicEvaluator()
        first = evaluator.evaluate(FakeTreeEnv(hidden_rng=3))
        second = evaluator.evaluate(FakeTreeEnv(hidden_rng=4))
        self.assertEqual(first.value, second.value)
        self.assertFalse(first.metadata["uses_hidden_state"])

    def test_rejected_value_checkpoint_cannot_drive_search(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "search-eligibility"):
            OnePlyForkSearchPolicy(
                FixedPolicy(1), RejectedValueEvaluator()
            ).select(FakeTreeEnv())


if __name__ == "__main__":
    unittest.main()
