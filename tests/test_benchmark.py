from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from stsai.benchmark import create_benchmark, evaluate_benchmark
from tests.test_runs import sample_run


class BenchmarkTests(unittest.TestCase):
    def test_manifest_selection_does_not_depend_on_outcome(self) -> None:
        left = sample_run(win=False)
        right = sample_run(win=True)
        right["seed"] = "SEED-456"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "runs.json"
            manifest = Path(directory) / "benchmark.json"
            source.write_text(json.dumps([left, right]), encoding="utf-8")
            value = create_benchmark(
                [source], manifest, build_id="build-107", ascension=4, count_per_character=2
            )
        self.assertEqual(len(value["cases"]), 2)
        self.assertNotIn("win", json.dumps(value).lower())

    def test_evaluation_uses_first_attempt_and_reports_retries(self) -> None:
        first = sample_run(win=False)
        first["start_time"] = 200
        retry = sample_run(win=True)
        retry["start_time"] = 300
        manifest = {
            "schema_version": 1,
            "name": "test",
            "started_after_unix": 100,
            "target": {"build_id": "build-107", "ascension": 4},
            "cases": [{"seed": "SEED-123", "character": "CHARACTER.TEST"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "runs.json"
            manifest_path = Path(directory) / "manifest.json"
            runs.write_text(json.dumps([retry, first]), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = evaluate_benchmark(manifest_path, [runs])
        self.assertEqual(result["overall"]["completed"], 1)
        self.assertEqual(result["overall"]["wins"], 0)
        self.assertEqual(result["retry_attempts_ignored"], 1)


if __name__ == "__main__":
    unittest.main()
