"""Readers, filters, and summaries for STS2 run-history files."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


Run = dict[str, Any]
SUPPORTED_PLAIN_SUFFIXES = {".run", ".json", ".jsonl", ".ndjson"}
DIRECTORY_SCAN_SUFFIXES = {".run", ".jsonl", ".ndjson"}


class RunReadError(ValueError):
    """Raised when a run source cannot be parsed."""


@dataclass(frozen=True)
class RunFilter:
    """Conservative defaults for producing trustworthy training data."""

    min_ascension: int = 0
    include_abandoned: bool = False
    include_coop: bool = False
    include_cheated: bool = False
    build_ids: frozenset[str] = frozenset()

    def accepts(self, run: Mapping[str, Any]) -> bool:
        players = run.get("players")
        if not isinstance(players, list) or not isinstance(run.get("map_point_history"), list):
            return False
        if not self.include_abandoned and bool(run.get("was_abandoned", False)):
            return False
        if not self.include_coop and len(players) != 1:
            return False
        if not self.include_cheated and bool(run.get("_isCheated", False)):
            return False
        if int(run.get("ascension") or 0) < self.min_ascension:
            return False
        if self.build_ids and str(run.get("build_id", "")) not in self.build_ids:
            return False
        return True


@dataclass
class RunSummary:
    files_read: int = 0
    runs_seen: int = 0
    runs_accepted: int = 0
    runs_rejected: int = 0
    wins: int = 0
    abandoned: int = 0
    cheated: int = 0
    solo: int = 0
    coop: int = 0
    card_reward_decisions: int = 0
    builds: Counter[str] = field(default_factory=Counter)
    characters: Counter[str] = field(default_factory=Counter)
    schema_versions: Counter[str] = field(default_factory=Counter)

    @property
    def win_rate(self) -> float:
        return self.wins / self.runs_accepted if self.runs_accepted else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "files_read": self.files_read,
            "runs_seen": self.runs_seen,
            "runs_accepted": self.runs_accepted,
            "runs_rejected": self.runs_rejected,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 6),
            "abandoned_seen": self.abandoned,
            "cheated_seen": self.cheated,
            "solo_seen": self.solo,
            "coop_seen": self.coop,
            "card_reward_decisions": self.card_reward_decisions,
            "builds": dict(self.builds.most_common()),
            "characters": dict(self.characters.most_common()),
            "schema_versions": dict(self.schema_versions.most_common()),
        }


def default_run_history_root() -> Path:
    """Return the platform's likely STS2 save root.

    The Steam-id/profile/history layers below this root are intentionally found
    by recursive discovery so no account identifier needs to be configured.
    """

    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "SlayTheSpire2"
    return Path.home() / ".local" / "share" / "SlayTheSpire2"


def discover_sources(inputs: Sequence[str | os.PathLike[str]] | None = None) -> list[Path]:
    """Expand directories into stable, de-duplicated run source paths."""

    roots = [Path(value).expanduser() for value in (inputs or [default_run_history_root()])]
    found: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        scanning_directory = root.is_dir()
        candidates = root.rglob("*") if scanning_directory else [root]
        for path in candidates:
            if not path.is_file():
                continue
            lower = path.name.lower()
            suffixes = DIRECTORY_SCAN_SUFFIXES if scanning_directory else SUPPORTED_PLAIN_SUFFIXES
            if path.suffix.lower() in suffixes or lower.endswith(
                (".jsonl.gz", ".ndjson.gz", ".run.gz", ".json.gz")
            ):
                try:
                    key = str(path.resolve()).casefold()
                except OSError:
                    key = str(path.absolute()).casefold()
                found[key] = path
    return sorted(found.values(), key=lambda value: str(value).casefold())


def _objects_from_json_value(value: Any, source: Path) -> Iterator[Run]:
    if isinstance(value, dict):
        yield dict(value)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise RunReadError(f"{source}: item {index} is not a JSON object")
            yield dict(item)
        return
    raise RunReadError(f"{source}: expected a run object or list of run objects")


def _read_json_lines(lines: Iterable[str], source: Path) -> Iterator[Run]:
    for line_number, line in enumerate(lines, 1):
        text = line.strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RunReadError(f"{source}:{line_number}: {exc.msg}") from exc
        yield from _objects_from_json_value(value, source)


def read_runs(path: str | os.PathLike[str]) -> Iterator[Run]:
    """Yield runs from `.run`, JSON, NDJSON, or gzip-compressed NDJSON."""

    source = Path(path)
    lower = source.name.lower()
    try:
        if lower.endswith(".gz"):
            with gzip.open(source, "rt", encoding="utf-8-sig") as handle:
                yield from _read_json_lines(handle, source)
            return
        if source.suffix.lower() in {".jsonl", ".ndjson"}:
            with source.open("r", encoding="utf-8-sig") as handle:
                yield from _read_json_lines(handle, source)
            return
        with source.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
        yield from _objects_from_json_value(value, source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunReadError(f"Unable to read {source}: {exc}") from exc


def iter_map_points(run: Mapping[str, Any]) -> Iterator[tuple[int, int, Mapping[str, Any]]]:
    """Yield `(act, global_floor, point)` tuples from a run."""

    floor = 0
    history = run.get("map_point_history") or []
    for act_index, act_points in enumerate(history, 1):
        if not isinstance(act_points, list):
            continue
        for point in act_points:
            if not isinstance(point, dict):
                continue
            floor += 1
            yield act_index, floor, point


def player_stats(point: Mapping[str, Any], player: Mapping[str, Any], index: int = 0) -> Mapping[str, Any]:
    """Find the per-floor stats belonging to a player without persisting its id."""

    stats = point.get("player_stats") or []
    player_id = player.get("id")
    if player_id is not None:
        for candidate in stats:
            if isinstance(candidate, dict) and candidate.get("player_id") == player_id:
                return candidate
    if index < len(stats) and isinstance(stats[index], dict):
        return stats[index]
    return {}


def count_card_reward_decisions(run: Mapping[str, Any]) -> int:
    players = run.get("players") or []
    if not isinstance(players, list) or not players or not isinstance(players[0], dict):
        return 0
    total = 0
    for _, _, point in iter_map_points(run):
        choices = player_stats(point, players[0]).get("card_choices") or []
        if isinstance(choices, list) and choices:
            total += 1
    return total


def run_key(run: Mapping[str, Any]) -> str:
    """Return a stable anonymized run key that excludes the Steam/player id."""

    raw_players = run.get("players")
    players = raw_players if isinstance(raw_players, list) else []
    characters = sorted(
        str(player.get("character", ""))
        for player in players
        if isinstance(player, dict)
    )
    parts = [
        str(run.get("build_id", "")),
        str(run.get("seed", "")),
        str(run.get("start_time", "")),
        ",".join(characters),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def summarize_runs(
    sources: Sequence[str | os.PathLike[str]], run_filter: RunFilter | None = None
) -> RunSummary:
    run_filter = run_filter or RunFilter()
    summary = RunSummary(files_read=len(sources))
    for source in sources:
        for run in read_runs(source):
            summary.runs_seen += 1
            raw_players = run.get("players")
            players = raw_players if isinstance(raw_players, list) else []
            if len(players) == 1:
                summary.solo += 1
            elif len(players) > 1:
                summary.coop += 1
            if bool(run.get("was_abandoned", False)):
                summary.abandoned += 1
            if bool(run.get("_isCheated", False)):
                summary.cheated += 1
            if not run_filter.accepts(run):
                summary.runs_rejected += 1
                continue
            summary.runs_accepted += 1
            summary.wins += int(bool(run.get("win", False)))
            summary.builds[str(run.get("build_id", "unknown"))] += 1
            summary.schema_versions[str(run.get("schema_version", "unknown"))] += 1
            for player in players:
                if isinstance(player, dict):
                    summary.characters[str(player.get("character", "unknown"))] += 1
            summary.card_reward_decisions += count_card_reward_decisions(run)
    return summary
