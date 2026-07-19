from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from stsai.human_telemetry import (
    HUMAN_COMBAT_CHARACTERS,
    build_human_combat_dataset,
    create_human_combat_corpus_manifest,
)


def _seed_hash(build_id: str, seed: str) -> str:
    return hashlib.sha256(f"{build_id}\x1f{seed}".encode()).hexdigest()


def _state(character_id: str) -> dict:
    return {
        "state_type": "monster",
        "run": {"act": 1, "floor": 2, "ascension": 10},
        "player": {
            "character": "localized title must not matter",
            "character_id": character_id.removeprefix("CHARACTER."),
            "hp": 60,
            "max_hp": 70,
            "energy": 3,
            "hand": [],
            "potions": [],
        },
        "battle": {
            "round": 1,
            "turn": "player",
            "is_play_phase": True,
            "player_actions_disabled": False,
            "can_end_turn": True,
            "encounter_id": "ENCOUNTER.TEST",
            "enemies": [{"entity_id": "enemy_0", "hp": 12, "max_hp": 12}],
        },
    }


class HumanTelemetryTests(unittest.TestCase):
    def _manifest(self, root: Path, *, exclusions: list[Path] | None = None) -> Path:
        game = root / "sts2.dll"
        bridge = root / "STS2_MCP.dll"
        patch = root / "telemetry.patch"
        manifest = root / "human-manifest.json"
        game.write_bytes(b"game-v1071")
        bridge.write_bytes(b"bridge-v2")
        patch.write_bytes(b"telemetry-v2")
        create_human_combat_corpus_manifest(
            manifest,
            game_assembly_path=game,
            bridge_dll_path=bridge,
            telemetry_patch_path=patch,
            excluded_manifests=exclusions or [],
            decisions_per_character=1,
        )
        return manifest

    def _event(
        self,
        manifest: Path,
        character: str,
        index: int,
        *,
        build_id: str = "v0.107.1",
        seed: str | None = None,
    ) -> dict:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return {
            "schema_version": 2,
            "event_type": "human_combat_action",
            "trajectory_id": f"run-{index}",
            "step": 0,
            "source": "game_ui",
            "collector": "STS2MCP",
            "mode": "singleplayer",
            "build_id": build_id,
            "game_assembly_sha256": payload["source"]["game_assembly_sha256"],
            "run_seed_sha256": _seed_hash("v0.107.1", seed or f"human-{index}"),
            "observation": _state(character),
            "action": {"action": "end_turn"},
        }

    def test_balanced_builder_emits_train_only_provenance_safe_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            source = root / "human.jsonl"
            dataset = root / "human-dataset.jsonl"
            receipt = root / "receipt.json"
            events = [
                self._event(manifest, character, index)
                for index, character in enumerate(HUMAN_COMBAT_CHARACTERS)
            ]
            source.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

            result = build_human_combat_dataset(
                [source],
                dataset,
                manifest_path=manifest,
                receipt_output=receipt,
            )
            records = [json.loads(line) for line in dataset.read_text().splitlines()]
            encoded_dataset = dataset.read_text()

        self.assertEqual(len(records), 5)
        self.assertTrue(all(record["split"] == "train" for record in records))
        self.assertTrue(
            all(
                record["patch"]["label_source"] == "human_ui_demonstration"
                for record in records
            )
        )
        self.assertEqual(
            {record["observation"]["player"]["character"] for record in records},
            set(HUMAN_COMBAT_CHARACTERS),
        )
        self.assertNotIn("human-", encoded_dataset)
        self.assertEqual(set(result["selected_events_by_character"].values()), {1})

    def test_wrong_build_fails_without_publishing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            source = root / "human.jsonl"
            dataset = root / "human-dataset.jsonl"
            event = self._event(manifest, HUMAN_COMBAT_CHARACTERS[0], 0, build_id="v0.106.1")
            source.write_text(json.dumps(event) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Wrong telemetry build"):
                build_human_combat_dataset(
                    [source], dataset, manifest_path=manifest
                )

            self.assertFalse(dataset.exists())

    def test_reserved_seed_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            excluded = root / "excluded.json"
            excluded.write_text(
                json.dumps({"cases": [{"character": "Silent", "seed": "reserved"}]}),
                encoding="utf-8",
            )
            manifest = self._manifest(root, exclusions=[excluded])
            source = root / "human.jsonl"
            dataset = root / "human-dataset.jsonl"
            event = self._event(
                manifest, HUMAN_COMBAT_CHARACTERS[0], 0, seed="reserved"
            )
            source.write_text(json.dumps(event) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "reserved evaluation seed"):
                build_human_combat_dataset(
                    [source], dataset, manifest_path=manifest
                )


if __name__ == "__main__":
    unittest.main()
