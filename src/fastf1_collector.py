"""
Resilient FastF1 session collector with per-session checkpoints and run-until-complete.

Writes yearly ALL_{DATASET}_{year}.csv files under data/raw/fastf1_2018plus/.
Progress is persisted after each session so runs can resume after crashes or rate limits.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

SESSION_DATASETS: Tuple[str, ...] = ("RESULTS", "LAPS", "TELEMETRY", "WEATHER")
DEFAULT_SESSION_TYPES: Tuple[str, ...] = ("R", "Q", "FP1", "FP2", "FP3", "Sprint")

# Minimum bytes for a non-empty yearly export.
MIN_YEARLY_FILE_BYTES = 512

# For feature pipeline we require race-session laps/weather for (almost) every event.
MIN_RACE_COVERAGE_RATIO = 0.95


@dataclass
class CollectorConfig:
    years: Iterable[int] = range(2025, 2026)
    session_types: Tuple[str, ...] = DEFAULT_SESSION_TYPES
    datasets: Tuple[str, ...] = SESSION_DATASETS
    request_delay_seconds: float = 1.0
    rate_limit_wait_seconds: float = 60.0
    rate_limit_max_wait_seconds: float = 600.0
    rate_limit_backoff_factor: float = 1.5
    pass_delay_seconds: float = 30.0
    max_passes: Optional[int] = None  # None = run until complete


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    slug = re.sub(r"[-\s]+", "_", slug.strip())
    return slug[:120] or "event"


def session_id(year: int, event_name: str, session_type: str) -> str:
    return f"{year}|{event_name}|{session_type}"


def to_dataframe(obj: Any) -> Optional[pd.DataFrame]:
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        try:
            df = obj.copy()
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
        except Exception:
            try:
                df = pd.DataFrame(obj)
            except Exception:
                return None
    if getattr(df, "empty", True):
        return None
    return df.reset_index(drop=True)


def is_transient_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    transient_markers = (
        "limit",
        "429",
        "too many requests",
        "timeout",
        "timed out",
        "connection",
        "temporarily",
        "503",
        "502",
        "504",
        "reset by peer",
        "remote end closed",
        "service unavailable",
    )
    return any(marker in msg for marker in transient_markers)


def is_unavailable_session_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    unavailable_markers = (
        "cannot find session",
        "session not found",
        "no session",
        "invalid session",
        "does not exist",
        "not available",
        "no data for this session",
    )
    return any(marker in msg for marker in unavailable_markers)


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = {"sessions": {}}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Could not read manifest %s (%s); starting fresh", path, exc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def get(self, sid: str) -> Dict[str, Any]:
        return self.data.setdefault("sessions", {}).setdefault(sid, {})

    def mark_unavailable(self, sid: str, reason: str) -> None:
        entry = self.get(sid)
        entry.update({"status": "unavailable", "reason": reason, "updated": _utc_now()})
        self.save()

    def mark_complete(self, sid: str, datasets: List[str]) -> None:
        entry = self.get(sid)
        entry.update(
            {
                "status": "complete",
                "datasets": sorted(set(datasets)),
                "updated": _utc_now(),
            }
        )
        self.save()

    def is_complete(self, sid: str, required_datasets: Set[str]) -> bool:
        entry = self.get(sid)
        if entry.get("status") == "unavailable":
            return True
        collected = set(entry.get("datasets") or [])
        return required_datasets.issubset(collected)

    def session_has_checkpoints(
        self,
        collector: "FastF1Collector",
        year: int,
        event_name: str,
        session_type: str,
        required_datasets: Set[str],
    ) -> bool:
        sid = session_id(year, event_name, session_type)
        entry = self.get(sid)
        if entry.get("status") == "unavailable":
            return True
        for dataset in required_datasets:
            if not collector.checkpoint_path(year, event_name, session_type, dataset).exists():
                return False
        return True

    def pending_sessions(
        self,
        collector: "FastF1Collector",
        year: int,
        schedule_events: List[str],
        session_types: Iterable[str],
        required_datasets: Set[str],
    ) -> List[Tuple[str, str, str]]:
        pending: List[Tuple[str, str, str]] = []
        for event in schedule_events:
            for sess in session_types:
                if sess == "Sprint" and year < 2021:
                    continue
                sid = session_id(year, event, sess)
                if not self.session_has_checkpoints(collector, year, event, sess, required_datasets):
                    pending.append((event, sess, sid))
        return pending


class FastF1Collector:
    def __init__(
        self,
        save_root: Path,
        cache_dir: Path,
        config: CollectorConfig,
    ):
        self.save_root = save_root
        self.cache_dir = cache_dir
        self.config = config
        self.checkpoint_root = save_root / "_checkpoints"
        self.manifest = Manifest(save_root / "_collection_manifest.json")

        import fastf1 as ff1

        self.ff1 = ff1
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_root.mkdir(parents=True, exist_ok=True)
        ff1.Cache.enable_cache(str(cache_dir))

    def yearly_path(self, dataset: str, year: int) -> Path:
        return self.save_root / f"ALL_{dataset}_{year}.csv"

    def checkpoint_path(self, year: int, event_name: str, session_type: str, dataset: str) -> Path:
        return (
            self.checkpoint_root
            / str(year)
            / _slugify(event_name)
            / session_type
            / f"{dataset}.csv"
        )

    def call_with_retry(self, func: Callable[[], Any], description: str) -> Any:
        attempt = 0
        while True:
            try:
                result = func()
                if self.config.request_delay_seconds > 0:
                    time.sleep(self.config.request_delay_seconds)
                return result
            except Exception as exc:
                if is_unavailable_session_error(exc):
                    raise
                if is_transient_error(exc):
                    wait = min(
                        self.config.rate_limit_max_wait_seconds,
                        self.config.rate_limit_wait_seconds
                        * (self.config.rate_limit_backoff_factor ** min(attempt, 8)),
                    )
                    logger.warning(
                        "%s → transient error (attempt %d): %s; sleeping %.0fs",
                        description,
                        attempt + 1,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                    attempt += 1
                    continue
                logger.warning("%s → non-transient error: %s; retrying in 30s", description, exc)
                time.sleep(30)
                attempt += 1

    def datasets_needed_for_year(self, year: int, events: Optional[List[str]] = None) -> Set[str]:
        needed = set()
        for dataset in self.config.datasets:
            if not self.is_yearly_dataset_valid(dataset, year):
                needed.add(dataset)
                continue
            if events:
                ok, _ = self.validate_yearly_dataset(dataset, year, events)
                if not ok:
                    needed.add(dataset)
        return needed

    def is_yearly_dataset_valid(self, dataset: str, year: int) -> bool:
        path = self.yearly_path(dataset, year)
        if not path.exists() or path.stat().st_size < MIN_YEARLY_FILE_BYTES:
            return False
        try:
            header = pd.read_csv(path, nrows=5, low_memory=False)
        except Exception:
            return False
        if header.empty:
            return False
        if "Year" in header.columns and not (pd.to_numeric(header["Year"], errors="coerce") == year).any():
            return False
        return True

    def get_schedule(self, year: int) -> List[str]:
        schedule = self.call_with_retry(
            lambda: self.ff1.get_event_schedule(year, include_testing=False),
            f"{year} schedule",
        )
        df = to_dataframe(schedule)
        if df is None or "EventName" not in df.columns:
            return []
        return df["EventName"].astype(str).tolist()

    def validate_yearly_dataset(self, dataset: str, year: int, events: List[str]) -> Tuple[bool, str]:
        path = self.yearly_path(dataset, year)
        if not path.exists():
            return False, "file missing"
        try:
            usecols = ["Year", "Event", "Session"] if dataset in {"LAPS", "WEATHER", "RESULTS"} else None
            df = pd.read_csv(path, usecols=usecols, low_memory=False) if usecols else pd.read_csv(path, low_memory=False)
        except Exception as exc:
            return False, f"unreadable: {exc}"
        if df.empty:
            return False, "empty file"
        if dataset in {"LAPS", "WEATHER"}:
            if "Session" not in df.columns or "Event" not in df.columns:
                return False, "missing Session/Event columns"
            race = df[(df["Session"] == "R") & (pd.to_numeric(df["Year"], errors="coerce") == year)]
            events_with_r = set(race["Event"].astype(str).unique())
            expected = set(events)
            coverage = len(events_with_r & expected) / max(len(expected), 1)
            if coverage < MIN_RACE_COVERAGE_RATIO:
                return False, f"race-session coverage {coverage:.1%} < {MIN_RACE_COVERAGE_RATIO:.0%}"
        return True, "ok"

    def rebuild_yearly_csv(self, dataset: str, year: int) -> bool:
        files = sorted(self.checkpoint_root.glob(f"{year}/**/{dataset}.csv"))
        if not files:
            return False
        frames: List[pd.DataFrame] = []
        for fp in files:
            try:
                part = pd.read_csv(fp, low_memory=False)
                if not part.empty:
                    frames.append(part)
            except Exception as exc:
                logger.warning("Skipping corrupt checkpoint %s (%s)", fp, exc)
        if not frames:
            return False
        combined = pd.concat(frames, ignore_index=True, sort=False)
        out = self.yearly_path(dataset, year)
        combined.to_csv(out, index=False)
        logger.info("  [REBUILT] %s %s: %d rows from %d checkpoints → %s", dataset, year, len(combined), len(frames), out.name)
        return True

    def save_checkpoint(self, df: pd.DataFrame, year: int, event: str, session_type: str, dataset: str) -> None:
        path = self.checkpoint_path(year, event, session_type, dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def collect_session(
        self,
        year: int,
        event_name: str,
        session_type: str,
        required_datasets: Set[str],
    ) -> List[str]:
        label = f"{year} {event_name} {session_type}"
        sid = session_id(year, event_name, session_type)

        if self.manifest.session_has_checkpoints(self, year, event_name, session_type, required_datasets):
            return list(required_datasets)

        session = self.call_with_retry(
            lambda: self.ff1.get_session(year, event_name, session_type),
            f"{label} get_session()",
        )

        load_laps = bool({"LAPS", "TELEMETRY", "WEATHER"} & required_datasets)
        load_telemetry = "TELEMETRY" in required_datasets
        load_weather = "WEATHER" in required_datasets

        def load_session() -> bool:
            session.load(
                laps=load_laps,
                telemetry=load_telemetry,
                weather=load_weather,
                messages=False,
            )
            return True

        self.call_with_retry(load_session, f"{label} load()")

        existing = [
            ds
            for ds in required_datasets
            if self.checkpoint_path(year, event_name, session_type, ds).exists()
        ]
        collected: List[str] = list(existing)

        if "RESULTS" in required_datasets:
            df = to_dataframe(session.results)
            if df is not None:
                df["Year"] = year
                df["Event"] = event_name
                df["Session"] = session_type
                self.save_checkpoint(df, year, event_name, session_type, "RESULTS")
                collected.append("RESULTS")
                logger.info("    %s → RESULTS: %d rows (checkpointed)", label, len(df))

        if "LAPS" in required_datasets:
            df = to_dataframe(session.laps)
            if df is not None:
                df["Year"] = year
                df["Event"] = event_name
                df["Session"] = session_type
                self.save_checkpoint(df, year, event_name, session_type, "LAPS")
                collected.append("LAPS")
                logger.info("    %s → LAPS: %d rows (checkpointed)", label, len(df))

        if "TELEMETRY" in required_datasets:
            tel_frames: List[pd.DataFrame] = []
            car_data = session.car_data
            if isinstance(car_data, dict) and car_data:
                for driver, tel in car_data.items():
                    df = to_dataframe(tel)
                    if df is not None:
                        df["Year"] = year
                        df["Event"] = event_name
                        df["Session"] = session_type
                        df["Driver"] = driver
                        tel_frames.append(df)
            else:
                df = to_dataframe(car_data)
                if df is not None:
                    df["Year"] = year
                    df["Event"] = event_name
                    df["Session"] = session_type
                    tel_frames.append(df)
            if tel_frames:
                combined = pd.concat(tel_frames, ignore_index=True, sort=False)
                self.save_checkpoint(combined, year, event_name, session_type, "TELEMETRY")
                collected.append("TELEMETRY")
                logger.info("    %s → TELEMETRY: %d rows (checkpointed)", label, len(combined))

        if "WEATHER" in required_datasets:
            df = to_dataframe(session.weather_data)
            if df is not None:
                df["Year"] = year
                df["Event"] = event_name
                df["Session"] = session_type
                self.save_checkpoint(df, year, event_name, session_type, "WEATHER")
                collected.append("WEATHER")
                logger.info("    %s → WEATHER: %d rows (checkpointed)", label, len(df))

        if set(collected) >= required_datasets:
            self.manifest.mark_complete(sid, collected)
        else:
            missing = required_datasets - set(collected)
            logger.warning(
                "    %s → incomplete (%s missing); will retry later",
                label,
                ", ".join(sorted(missing)),
            )

        return collected

    def process_year(self, year: int) -> Dict[str, Any]:
        logger.info("%s YEAR %s %s", "=" * 20, year, "=" * 20)
        events = self.get_schedule(year)
        if not events:
            logger.warning("No schedule for %s", year)
            return {"year": year, "complete": False, "reason": "no schedule"}

        required = self.datasets_needed_for_year(year, events)
        if not required:
            logger.info("  All datasets valid for %s", year)
            return {"year": year, "complete": True, "required": []}

        logger.info("  Pending datasets: %s", ", ".join(sorted(required)))
        pending = self.manifest.pending_sessions(self, year, events, self.config.session_types, required)
        logger.info("  Pending sessions: %d", len(pending))

        for event_name, sess_type, sid in pending:
            label = f"{year} {event_name} {sess_type}"
            try:
                self.collect_session(year, event_name, sess_type, required)
            except Exception as exc:
                if is_unavailable_session_error(exc):
                    logger.info("    %s → unavailable (%s)", label, exc)
                    self.manifest.mark_unavailable(sid, str(exc))
                else:
                    logger.warning("    %s → failed (%s); will retry next pass", label, exc)

        # Rebuild yearly files from checkpoints for datasets still missing/invalid
        for dataset in sorted(required):
            if self.is_yearly_dataset_valid(dataset, year):
                ok, _ = self.validate_yearly_dataset(dataset, year, events)
                if ok:
                    continue
            self.rebuild_yearly_csv(dataset, year)
            ok, reason = self.validate_yearly_dataset(dataset, year, events)
            if ok:
                logger.info("  [VALID] %s %s — %s", dataset, year, reason)
            else:
                logger.warning("  [INVALID] %s %s — %s (will retry)", dataset, year, reason)

        still_needed = self.datasets_needed_for_year(year, events)
        return {"year": year, "complete": len(still_needed) == 0, "required": sorted(still_needed)}

    def is_year_complete(self, year: int) -> bool:
        events = self.get_schedule(year)
        if not events:
            return False
        return len(self.datasets_needed_for_year(year, events)) == 0


def run_collector_until_complete(
    save_root: Path,
    cache_dir: Path,
    config: Optional[CollectorConfig] = None,
) -> None:
    """Run collection passes until all configured years/datasets validate."""
    config = config or CollectorConfig()
    collector = FastF1Collector(save_root, cache_dir, config)

    cache_info = collector.ff1.Cache.get_cache_info()
    logger.info("FastF1 cache: %s @ %s", cache_info[0], cache_info[1])
    logger.info("Save root: %s", save_root)

    pass_num = 0
    while True:
        pass_num += 1
        if config.max_passes is not None and pass_num > config.max_passes:
            logger.error("Reached max_passes=%s without completing", config.max_passes)
            break

        logger.info("%s COLLECTION PASS %d %s", "=" * 16, pass_num, "=" * 16)
        all_complete = True
        summaries = []

        for year in config.years:
            if collector.is_year_complete(year):
                logger.info("Year %s already complete", year)
                summaries.append({"year": year, "complete": True})
                continue
            summary = collector.process_year(year)
            summaries.append(summary)
            if not summary.get("complete"):
                all_complete = False

        if all_complete:
            logger.info("=== ALL YEARS COMPLETE ===")
            for s in summaries:
                logger.info("  %s: complete", s["year"])
            break

        logger.info(
            "Pass %d finished — not complete yet; sleeping %ds before next pass",
            pass_num,
            int(config.pass_delay_seconds),
        )
        time.sleep(config.pass_delay_seconds)
