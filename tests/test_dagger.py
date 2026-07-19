from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from stsai.dagger import (
    DaggerMixturePolicy,
    _publish_runtime_failure,
    build_dagger_dataset,
    create_dagger_manifest,
    merge_dagger_training_dataset,
)
from stsai.headless_env import HEADLESS_CHARACTERS, Sts2Env


class FakeClient:
    version = "test"

    def __init__(self, state: dict):
        self.state = state

    def start_run(self, **_: object) -> dict:
        return self.state


class FakeModel:
    dimension = 64

    def scores(self, features: np.ndarray) -> np.ndarray:
        return np.arange(len(features), dtype=np.float32)


def _combat_state() -> dict:
    return {
        "type": "decision",
        "decision": "combat_play",
        "context": {"act": 1, "floor": 2, "room_type": "Monster"},
        "round": 1,
        "energy": 3,
        "max_energy": 3,
        "hand": [
            {
                "index": 0,
                "id": "CARD.STRIKE_DEFECT",
                "type": "Attack",
                "cost": 1,
                "can_play": True,
                "target_type": "AnyEnemy",
                "damage_by_target": [
                    {"target_index": 0, "damage": 6},
                    {"target_index": 1, "damage": 6},
                ],
            }
        ],
        "enemies": [
            {"index": 0, "id": "MONSTER.A", "hp": 20, "max_hp": 20},
            {"index": 1, "id": "MONSTER.B", "hp": 5, "max_hp": 10},
        ],
        "player": {
            "hp": 50,
            "max_hp": 75,
            "block": 0,
            "gold": 99,
            "deck": [],
            "relics": [],
            "potions": [],
        },
    }


def _reset_env() -> Sts2Env:
    env = Sts2Env(client=FakeClient(_combat_state()))
    env.reset(character="Defect", ascension=10, seed="DAGGER_TEST")
    return env


def _write_trajectory(path: Path, *, character: str, seed: str) -> None:
    env = Sts2Env(client=FakeClient(_combat_state()))
    observation = env.reset(character=character, ascension=10, seed=seed)
    actions = [action.as_dict() for action in env.legal_actions]
    teacher_action = actions[1]
    learner_action = actions[-1]
    events = [
        {
            "schema_version": 1,
            "event_type": "episode_start",
            "character": f"CHARACTER.{character.upper()}",
            "seed": seed,
            "policy": "dagger_round_1_beta_0.500",
            "policy_metadata": {
                "dagger_round": 1,
                "beta": 0.5,
                "validation_data": False,
            },
        },
        {
            "schema_version": 1,
            "event_type": "transition",
            "step": 1,
            "observation": observation,
            "legal_actions": actions,
            "action": learner_action,
            "next_observation": {"decision_type": "game_over"},
            "terminated": True,
            "info": {
                "policy": {
                    "source": "dagger_mixture",
                    "dagger": {
                        "round": 1,
                        "beta": 0.5,
                        "mixing_draw": 0.75,
                        "executed_by": "learner",
                        "teacher_action": teacher_action,
                        "learner_action": learner_action,
                        "teacher_learner_agree": False,
                    },
                }
            },
        },
        {
            "schema_version": 1,
            "event_type": "episode_end",
            "steps": 1,
            "truncated": False,
            "floor": 2,
        },
    ]
    path.write_text(
        "".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events),
        encoding="utf-8",
    )


class DaggerTests(unittest.TestCase):
    def test_manifest_precommits_balanced_aggregation_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dagger.json"
            payload = create_dagger_manifest(
                output,
                aggregate_per_character=2,
                validation_per_character=1,
            )

        self.assertFalse(payload["selection"]["outcomes_inspected"])
        self.assertFalse(payload["selection"]["validation_aggregation_allowed"])
        self.assertEqual(payload["dagger"]["beta"], 0.5)
        self.assertEqual(len(payload["cases"]), len(HEADLESS_CHARACTERS) * 3)
        identities = {(case["character"], case["seed"]) for case in payload["cases"]}
        self.assertEqual(len(identities), len(payload["cases"]))
        for character in HEADLESS_CHARACTERS:
            cases = [
                case
                for case in payload["cases"]
                if case["character"] == f"CHARACTER.{character.upper()}"
            ]
            self.assertEqual(
                Counter(case["pool"] for case in cases),
                Counter({"aggregate": 2, "validation": 1}),
            )

    def test_manifest_can_bind_the_exact_learner_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            learner = root / "learner.npz"
            learner.write_bytes(b"frozen learner")
            payload = create_dagger_manifest(
                root / "manifest.json",
                round_number=2,
                beta=0.25,
                aggregate_per_character=1,
                validation_per_character=1,
                learner_name="combat_policy.dagger-r1.v0.107.1",
                learner_model_path=learner,
            )

        self.assertEqual(
            payload["dagger"]["learner"], "combat_policy.dagger-r1.v0.107.1"
        )
        self.assertEqual(
            payload["dagger"]["learner_model_sha256"],
            "4a5e0ebbc558711890a87929aeffd9da8161eb5c705db0b3cbc689e278f53a21",
        )

    def test_beta_mix_records_executed_learner_and_teacher_label_separately(self) -> None:
        env = _reset_env()
        policy = DaggerMixturePolicy(
            FakeModel(), FakeModel(), beta=0.0, mixing_seed=7
        )

        index, info = policy.select(env)

        dagger = info["dagger"]
        self.assertEqual(info["source"], "dagger_mixture")
        self.assertEqual(dagger["executed_by"], "learner")
        self.assertEqual(env.legal_actions[index].command, "end_turn")
        self.assertEqual(dagger["learner_action"]["action"], "end_turn")
        self.assertEqual(dagger["teacher_action"]["action"], "play_card")
        self.assertFalse(dagger["teacher_learner_agree"])

    def test_dataset_uses_teacher_label_and_preserves_executed_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest = create_dagger_manifest(
                manifest_path,
                aggregate_per_character=1,
                validation_per_character=1,
            )
            case = next(case for case in manifest["cases"] if case["pool"] == "aggregate")
            character = case["character"].rsplit(".", 1)[-1].title()
            trajectory = root / "aggregate.jsonl"
            _write_trajectory(trajectory, character=character, seed=case["seed"])
            output = root / "dataset.jsonl"

            summary = build_dagger_dataset(
                [trajectory],
                output,
                manifest_path=manifest_path,
                build_id="v0.107.1",
            )
            record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(summary.decisions_written, 1)
        self.assertEqual(record["split"], "train")
        self.assertEqual(record["chosen_action"], record["teacher_action"])
        self.assertNotEqual(record["teacher_action"], record["executed_action"])
        self.assertEqual(record["executed_action"], record["learner_action"])

    def test_dataset_rejects_reserved_validation_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest = create_dagger_manifest(
                manifest_path,
                aggregate_per_character=1,
                validation_per_character=1,
            )
            case = next(case for case in manifest["cases"] if case["pool"] == "validation")
            character = case["character"].rsplit(".", 1)[-1].title()
            trajectory = root / "validation.jsonl"
            _write_trajectory(trajectory, character=character, seed=case["seed"])

            with self.assertRaisesRegex(ValueError, "Validation trajectory"):
                build_dagger_dataset(
                    [trajectory],
                    root / "dataset.jsonl",
                    manifest_path=manifest_path,
                    build_id="v0.107.1",
                )

    def test_runtime_failure_atomically_publishes_partial_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            temporary = path.with_suffix(".jsonl.tmp")
            events = [
                {
                    "schema_version": 1,
                    "event_type": "episode_start",
                    "trajectory_id": "dagger-test",
                    "character": "CHARACTER.IRONCLAD",
                    "seed": "DGR1I000000",
                },
                {
                    "schema_version": 1,
                    "event_type": "transition",
                    "step": 1,
                    "reward": 0.0,
                    "observation": {"context": {"act": 1, "floor": 4}},
                    "next_observation": {"context": {"act": 1, "floor": 4}},
                    "info": {"policy": {"source": "dagger_mixture"}},
                },
            ]
            temporary.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

            ending = _publish_runtime_failure(path, TimeoutError("known timeout"))
            published = [json.loads(line) for line in path.read_text().splitlines()]

            self.assertFalse(temporary.exists())

        self.assertTrue(ending["truncated"])
        self.assertEqual(ending["floor"], 4)
        self.assertEqual(ending["runtime_error"]["type"], "TimeoutError")
        self.assertEqual(published[-1], ending)

    def test_merge_preserves_base_validation_and_test_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.jsonl"
            dagger = root / "dagger.jsonl"
            output = root / "combined.jsonl"
            base_records = [
                {"split": split, "trajectory_key": f"base-{split}"}
                for split in ("train", "validation", "test")
            ]
            dagger_records = [
                {
                    "split": "train",
                    "trajectory_key": "dagger-train",
                    "patch": {"label_source": "dagger_heuristic_teacher"},
                }
            ]
            base.write_text(
                "".join(json.dumps(record) + "\n" for record in base_records),
                encoding="utf-8",
            )
            dagger.write_text(
                "".join(json.dumps(record) + "\n" for record in dagger_records),
                encoding="utf-8",
            )

            summary = merge_dagger_training_dataset(base, dagger, output)
            combined = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(summary["splits"], {"train": 2, "validation": 1, "test": 1})
        self.assertEqual(combined[:3], base_records)
        self.assertEqual(combined[3:], dagger_records)


if __name__ == "__main__":
    unittest.main()
