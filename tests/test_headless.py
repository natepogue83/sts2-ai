from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

from stsai.headless import (
    HeadlessConfig,
    HeadlessRuntimeError,
    HeadlessSnapshotDivergence,
    HeadlessSnapshotError,
    Sts2HeadlessClient,
)


FAKE_RUNTIME = r"""
import json
import sys
print(json.dumps({"type": "ready", "version": "test"}), flush=True)
for line in sys.stdin:
    command = json.loads(line)
    if command["cmd"] == "quit":
        print(json.dumps({"type": "quit_result", "success": True}), flush=True)
        break
    if command["cmd"] == "snapshot_checkpoint":
        print(json.dumps({
            "type": "snapshot_checkpoint",
            "schema_version": 1,
            "snapshot_safe": True,
            "stage": "test",
            "reasons": [],
            "semantic": {"test": True},
        }), flush=True)
        continue
    print(json.dumps({"type": "decision", "received": command}), flush=True)
"""


class HeadlessClientTests(unittest.TestCase):
    def config(self) -> HeadlessConfig:
        return HeadlessConfig(
            runtime=Path("unused"),
            command=(sys.executable, "-u", "-c", FAKE_RUNTIME),
            startup_timeout=5,
            command_timeout=5,
        )

    def test_process_protocol_start_action_and_close(self) -> None:
        with Sts2HeadlessClient(self.config()) as client:
            self.assertEqual(client.version, "test")
            state = client.start_run(
                character="Silent", ascension=10, seed="ABC", language="en"
            )
            self.assertEqual(state["received"]["character"], "Silent")
            action = client.action("play_card", card_index=2, target_index=0)
            self.assertEqual(action["received"]["args"]["card_index"], 2)

    def test_rejects_commands_after_close(self) -> None:
        client = Sts2HeadlessClient(self.config())
        client.close()
        with self.assertRaises(HeadlessRuntimeError):
            client.action("end_turn")

    def test_snapshot_fork_reconstructs_frozen_prefix_in_independent_process(self) -> None:
        with Sts2HeadlessClient(self.config()) as parent:
            parent.start_run(character="Silent", ascension=10, seed="ABC")
            expected = parent.action("play_card", card_index=2, target_index=0)
            snapshot = parent.snapshot()
            self.assertEqual(
                snapshot.information_scope, "native_reconstructive_semantic_v1"
            )
            self.assertNotIn("ABC", repr(snapshot))

            parent.action("end_turn")
            with parent.fork(snapshot) as child:
                self.assertNotEqual(child.pid, parent.pid)
                self.assertEqual(child.reconstructed_state, expected)
                self.assertEqual(child.reconstructed_state, snapshot.state())

                child.action("discard_potion", potion_index=1)
                self.assertNotEqual(child.reconstructed_state, parent.reconstructed_state)

    def test_snapshot_requires_explicit_seed(self) -> None:
        with Sts2HeadlessClient(self.config()) as client:
            client.start_run(character="Ironclad", ascension=0)
            with self.assertRaisesRegex(HeadlessSnapshotError, "explicit seed"):
                client.snapshot()

    def test_unclassified_command_taints_snapshot_until_new_genesis(self) -> None:
        with Sts2HeadlessClient(self.config()) as client:
            client.start_run(character="Ironclad", ascension=0, seed="ONE")
            client.command({"cmd": "future_mutation"})
            with self.assertRaisesRegex(HeadlessSnapshotError, "unclassified"):
                client.snapshot()

            client.start_run(character="Ironclad", ascension=0, seed="TWO")
            self.assertEqual(len(client.snapshot().steps), 1)

    def test_fork_fails_closed_on_response_digest_mismatch(self) -> None:
        with Sts2HeadlessClient(self.config()) as client:
            client.start_run(character="Defect", ascension=5, seed="ABC")
            snapshot = client.snapshot()
            bad_step = replace(snapshot.steps[0], response_sha256="0" * 64)
            corrupted = replace(snapshot, steps=(bad_step,))
            with self.assertRaises(HeadlessSnapshotDivergence):
                client.fork(corrupted)

    def test_transient_mismatch_is_allowed_only_with_exact_final_checkpoint(self) -> None:
        with Sts2HeadlessClient(self.config()) as client:
            client.start_run(character="Defect", ascension=5, seed="ABC")
            client.action("end_turn")
            snapshot = client.snapshot()
            bad_first = replace(snapshot.steps[0], response_sha256="0" * 64)
            transient = replace(
                snapshot, steps=(bad_first, *snapshot.steps[1:])
            )
            with client.fork(transient) as child:
                self.assertEqual(child.fork_replay_mismatches, (0,))
                self.assertEqual(child.reconstructed_state, snapshot.state())

    def test_fork_rejects_runtime_fingerprint_mismatch(self) -> None:
        with Sts2HeadlessClient(self.config()) as client:
            client.start_run(character="Regent", ascension=10, seed="ABC")
            snapshot = replace(client.snapshot(), runtime_sha256="f" * 64)
            with self.assertRaisesRegex(HeadlessSnapshotError, "fingerprint"):
                client.fork(snapshot)

    def test_path_load_is_frozen_inline_before_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            save_path = Path(directory) / "run.save"
            save_path.write_text('{"schema_version":1}', encoding="utf-8")
            with Sts2HeadlessClient(self.config()) as client:
                response = client.command(
                    {"cmd": "load_save", "path": str(save_path), "lang": "en"}
                )
                snapshot = client.snapshot()
                request = snapshot.steps[0].request()
                self.assertNotIn("path", request)
                self.assertEqual(request["json"], '{"schema_version":1}')

                save_path.unlink()
                with client.fork(snapshot) as child:
                    self.assertEqual(child.reconstructed_state, response)


if __name__ == "__main__":
    unittest.main()
