from __future__ import annotations

from threading import Event
import unittest

from stsai.headless_batch import (
    HeadlessBatchError,
    HeadlessBatchJob,
    HeadlessBatchManager,
)


class HeadlessBatchManagerTests(unittest.TestCase):
    def test_preserves_input_order_when_jobs_finish_out_of_order(self) -> None:
        first_started = Event()
        release_first = Event()

        def first() -> str:
            first_started.set()
            self.assertTrue(release_first.wait(timeout=2.0))
            return "first"

        def second() -> str:
            self.assertTrue(first_started.wait(timeout=2.0))
            release_first.set()
            return "second"

        report = HeadlessBatchManager(max_workers=2).run(
            [HeadlessBatchJob("first", first), HeadlessBatchJob("second", second)]
        )

        self.assertEqual(report.values(), ("first", "second"))
        self.assertEqual([outcome.index for outcome in report.outcomes], [0, 1])

    def test_failure_does_not_cancel_sibling_jobs(self) -> None:
        sibling_ran = Event()

        def fail() -> str:
            raise ValueError("bad seed")

        def sibling() -> str:
            sibling_ran.set()
            return "kept"

        report = HeadlessBatchManager(max_workers=2).run(
            [HeadlessBatchJob("bad", fail), HeadlessBatchJob("sibling", sibling)]
        )

        self.assertTrue(sibling_ran.is_set())
        self.assertEqual(len(report.failures), 1)
        self.assertEqual(report.failures[0].key, "bad")
        self.assertIsInstance(report.failures[0].error, ValueError)
        with self.assertRaises(HeadlessBatchError) as caught:
            report.values()
        self.assertIs(caught.exception.report, report)

    def test_rejects_invalid_worker_counts_and_duplicate_keys(self) -> None:
        with self.assertRaises(ValueError):
            HeadlessBatchManager(max_workers=0)
        with self.assertRaises(ValueError):
            HeadlessBatchManager(max_workers=1).run(
                [HeadlessBatchJob("same", lambda: 1), HeadlessBatchJob("same", lambda: 2)]
            )


if __name__ == "__main__":
    unittest.main()
