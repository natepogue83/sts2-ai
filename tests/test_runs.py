from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from stsai.runs import RunFilter, discover_sources, read_runs, run_key, summarize_runs


def sample_run(*, win: bool = True, abandoned: bool = False, cheated: bool = False) -> dict:
    return {
        "acts": ["ACT_1"],
        "ascension": 4,
        "build_id": "build-107",
        "game_mode": "standard",
        "map_point_history": [
            [
                {
                    "map_point_type": "combat",
                    "player_stats": [
                        {
                            "player_id": "private-player-id",
                            "current_hp": 70,
                            "max_hp": 75,
                            "current_gold": 110,
                            "cards_gained": [{"id": "CARD.B"}],
                            "card_choices": [
                                {"card": {"id": "CARD.A"}, "was_picked": False},
                                {"card": {"id": "CARD.B"}, "was_picked": True},
                                {"card": {"id": "CARD.C"}, "was_picked": False},
                            ],
                            "relic_choices": [
                                {"choice": "RELIC.A", "was_picked": False},
                                {"choice": "RELIC.B", "was_picked": True},
                                {"choice": "RELIC.C", "was_picked": False},
                            ],
                            "potion_choices": [
                                {"choice": "POTION.A", "was_picked": False},
                                {"choice": "POTION.B", "was_picked": False},
                                {"choice": "POTION.C", "was_picked": False},
                            ],
                            "ancient_choice": [
                                {"TextKey": "A", "was_chosen": True},
                                {"TextKey": "B", "was_chosen": False},
                                {"TextKey": "C", "was_chosen": False},
                            ],
                        }
                    ],
                    "rooms": [
                        {
                            "model_id": "ENCOUNTER.TEST",
                            "monster_ids": ["MONSTER.TEST"],
                            "room_type": "monster",
                            "turns_taken": 3,
                        }
                    ],
                }
            ]
        ],
        "players": [
            {
                "id": "private-player-id",
                "character": "CHARACTER.TEST",
                "deck": [
                    {"id": "CARD.START"},
                    {"id": "CARD.B", "floor_added_to_deck": 1},
                ],
                "relics": [{"id": "RELIC.START"}],
                "potions": [],
            }
        ],
        "run_time": 123,
        "schema_version": 1,
        "seed": "SEED-123",
        "start_time": 456,
        "was_abandoned": abandoned,
        "win": win,
        "_isCheated": cheated,
    }


class RunReaderTests(unittest.TestCase):
    def test_reads_gzip_ndjson_and_discovers_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.ndjson.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(sample_run()) + "\n")
                handle.write(json.dumps(sample_run(win=False)) + "\n")
            self.assertEqual(discover_sources([directory]), [path])
            self.assertEqual(len(list(read_runs(path))), 2)

    def test_filter_defaults_exclude_abandoned_coop_and_cheated(self) -> None:
        run_filter = RunFilter()
        self.assertTrue(run_filter.accepts(sample_run()))
        self.assertFalse(run_filter.accepts(sample_run(abandoned=True)))
        self.assertFalse(run_filter.accepts(sample_run(cheated=True)))
        coop = sample_run()
        coop["players"].append(dict(coop["players"][0]))
        self.assertFalse(run_filter.accepts(coop))

    def test_summary_counts_decisions_and_does_not_need_player_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.run"
            path.write_text(json.dumps(sample_run()), encoding="utf-8")
            summary = summarize_runs([path])
        self.assertEqual(summary.runs_accepted, 1)
        self.assertEqual(summary.card_reward_decisions, 1)
        self.assertEqual(summary.characters["CHARACTER.TEST"], 1)

    def test_run_key_excludes_player_id(self) -> None:
        left = sample_run()
        right = sample_run()
        right["players"][0]["id"] = "another-private-id"
        self.assertEqual(run_key(left), run_key(right))


if __name__ == "__main__":
    unittest.main()
