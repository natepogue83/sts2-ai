from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from stsai.dataset import (
    SKIP,
    DatasetBuildSummary,
    build_card_dataset,
    build_decision_dataset,
    card_reward_records,
    decision_records,
    split_for_run,
)
from tests.test_runs import sample_run


class DatasetTests(unittest.TestCase):
    def test_card_record_has_pre_choice_deck_and_no_identity(self) -> None:
        record = list(card_reward_records(sample_run()))[0]
        self.assertEqual(record["observation"]["deck"], ["CARD.START"])
        self.assertEqual(record["chosen"], "CARD.B")
        self.assertEqual(record["legal_options"], ["CARD.A", "CARD.B", "CARD.C", SKIP])
        encoded = json.dumps(record)
        self.assertNotIn("private-player-id", encoded)
        self.assertNotIn("seed", encoded.lower())
        self.assertNotIn("run_won", json.dumps(record["observation"]))

    def test_skip_is_a_real_legal_label(self) -> None:
        run = sample_run()
        for choice in run["map_point_history"][0][0]["player_stats"][0]["card_choices"]:
            choice["was_picked"] = False
        record = list(card_reward_records(run))[0]
        self.assertEqual(record["chosen"], SKIP)

    def test_multi_decision_records_require_recorded_alternatives(self) -> None:
        run = sample_run()
        records = list(decision_records(run))
        by_type = {record["decision_type"]: record for record in records}
        self.assertEqual(
            set(by_type), {"card_reward", "relic_reward", "potion_reward", "ancient"}
        )
        self.assertEqual(by_type["relic_reward"]["chosen"], "RELIC.B")
        self.assertEqual(by_type["potion_reward"]["chosen"], SKIP)
        self.assertEqual(by_type["ancient"]["chosen"], "ANCIENT.A")
        self.assertNotIn(SKIP, by_type["ancient"]["legal_options"])

        stats = run["map_point_history"][0][0]["player_stats"][0]
        stats["relic_choices"] = [{"choice": "RELIC.A", "was_picked": True}]
        self.assertNotIn(
            "relic_reward", {record["decision_type"] for record in decision_records(run)}
        )

    def test_split_is_seed_grouped(self) -> None:
        left = sample_run()
        right = sample_run()
        right["start_time"] = 999999
        right["players"][0]["id"] = "different-player"
        self.assertEqual(split_for_run(left), split_for_run(right))

    def test_deck_reversal_failures_are_counted_not_hidden(self) -> None:
        run = sample_run()
        # The final deck holds the upgraded id while the gain delta recorded
        # the base id, so reversal cannot remove it and the future card leaks
        # into the pre-choice snapshot. The build must count that, not hide it.
        run["players"][0]["deck"] = [{"id": "CARD.START"}, {"id": "CARD.B+1"}]

        clean_summary = DatasetBuildSummary()
        list(decision_records(sample_run(), summary=clean_summary))
        self.assertEqual(clean_summary.deck_reversal_failures, 0)
        self.assertEqual(clean_summary.runs_with_reversal_failures, 0)

        summary = DatasetBuildSummary()
        records = list(decision_records(run, summary=summary))
        self.assertEqual(summary.deck_reversal_failures, 1)
        self.assertEqual(summary.runs_with_reversal_failures, 1)
        self.assertEqual(summary.as_dict()["deck_reversal_failures"], 1)
        by_type = {record["decision_type"]: record for record in records}
        self.assertIn("CARD.B+1", by_type["card_reward"]["observation"]["deck"])

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "run.run"
            output = Path(directory) / "dataset.jsonl"
            source.write_text(json.dumps(run), encoding="utf-8")
            build_summary = build_decision_dataset([source], output)
        self.assertEqual(build_summary.deck_reversal_failures, 1)
        self.assertEqual(build_summary.runs_with_reversal_failures, 1)

    def test_builds_atomic_ndjson(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "run.run"
            output = Path(directory) / "dataset.jsonl"
            source.write_text(json.dumps(sample_run()), encoding="utf-8")
            summary = build_card_dataset([source], output)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary.decisions_written, 1)
            self.assertEqual(len(records), 1)
            self.assertFalse(output.with_name(output.name + ".tmp").exists())

    def test_builder_deduplicates_the_same_run_across_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.run"
            second = Path(directory) / "second.run"
            output = Path(directory) / "dataset.jsonl"
            encoded = json.dumps(sample_run())
            first.write_text(encoded, encoding="utf-8")
            second.write_text(encoded, encoding="utf-8")
            summary = build_card_dataset([first, second], output)
        self.assertEqual(summary.decisions_written, 1)
        self.assertEqual(summary.duplicate_runs, 1)

    def test_unified_builder_counts_each_decision_type(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "run.run"
            output = Path(directory) / "dataset.jsonl"
            source.write_text(json.dumps(sample_run()), encoding="utf-8")
            summary = build_decision_dataset([source], output)
        self.assertEqual(summary.decisions_written, 4)
        self.assertEqual(
            summary.decision_types,
            {
                "ancient": 1,
                "card_reward": 1,
                "potion_reward": 1,
                "relic_reward": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
