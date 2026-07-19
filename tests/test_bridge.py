from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

from stsai.bridge import BridgeConfig, STS2MCPClient, TransitionRecorder


class _BridgeHandler(BaseHTTPRequestHandler):
    counter = 0

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/v1/replay/latest":
            self._send(
                {
                    "schema_version": 1,
                    "format": "sts2_combat_replay_summary",
                    "actions": [{"action_type": "NetEndPlayerTurnAction"}],
                }
            )
            return
        payload = {
            "state_type": "monster",
            "run": {"act": 1, "floor": 2, "ascension": 0},
            "player": {"hp": 70, "energy": 3 - self.counter},
        }
        self._send(payload)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(length))
        type(self).counter += 1
        self._send({"status": "ok", "message": "action accepted"})

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class BridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        _BridgeHandler.counter = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _BridgeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_rejects_remote_bridge_urls(self) -> None:
        with self.assertRaises(ValueError):
            BridgeConfig("https://example.com")
        with self.assertRaises(ValueError):
            BridgeConfig("http://192.168.1.2:15526")

    def test_records_before_action_after_transition(self) -> None:
        client = STS2MCPClient(BridgeConfig(self.url))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trajectory.jsonl"
            recorder = TransitionRecorder(client, output, settle_timeout_seconds=0.2)
            event = recorder.act({"action": "end_turn"})
            resumed = TransitionRecorder(client, output, settle_timeout_seconds=0.2)
            saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(event["state_changed"])
        self.assertEqual(saved["observation"]["player"]["energy"], 3)
        self.assertEqual(saved["next_observation"]["player"]["energy"], 2)
        self.assertEqual(saved["action"], {"action": "end_turn"})
        self.assertEqual(event["trajectory_id"], resumed.trajectory_id)

    def test_reads_native_replay_summary(self) -> None:
        client = STS2MCPClient(BridgeConfig(self.url))
        replay = client.get_latest_replay()
        self.assertEqual(replay["format"], "sts2_combat_replay_summary")
        self.assertEqual(replay["actions"][0]["action_type"], "NetEndPlayerTurnAction")


if __name__ == "__main__":
    unittest.main()
