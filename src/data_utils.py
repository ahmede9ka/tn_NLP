"""Utilities for inspecting and preparing TEDxTN metadata."""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_COLUMNS = ("audio_filename", "start_time", "end_time", "transcription")
TARGET_COLUMNS = ("audio_filename", "start_time", "end_time", "translation")
SPLITS = ("train", "dev", "test")
HF_REVISION = "2f25fbea14b584a181f01f00aa4c6acbe352a6f7"
HF_REVISION_URL = f"https://huggingface.co/datasets/fbougares/TEDxTN/raw/{HF_REVISION}"


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV and reject files without a header."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def validate_rows(
    rows: list[dict[str, str]], *, text_column: str, allow_empty_text: bool = False
) -> dict[str, Any]:
    """Validate segment fields and return counts without silently dropping rows."""
    required = {"audio_filename", "start_time", "end_time", text_column}
    if not rows:
        raise ValueError(f"No rows found for column {text_column!r}")
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing_columns))}")

    malformed: list[int] = []
    empty_text: list[int] = []
    duplicate_keys: Counter[tuple[str, float, float]] = Counter()
    durations: list[float] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            start = float(row["start_time"])
            end = float(row["end_time"])
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError("invalid interval")
        except (TypeError, ValueError):
            malformed.append(row_number)
            continue
        durations.append(end - start)
        duplicate_keys[(row["audio_filename"], start, end)] += 1
        if not (row.get(text_column) or "").strip():
            empty_text.append(row_number)

    if malformed:
        raise ValueError(
            f"{len(malformed)} malformed segment interval(s); first CSV row: {malformed[0]}"
        )
    if empty_text and not allow_empty_text:
        raise ValueError(
            f"{len(empty_text)} empty {text_column!r} value(s); first CSV row: {empty_text[0]}"
        )

    return {
        "rows": len(rows),
        "talks": len({row["audio_filename"] for row in rows}),
        "empty_text_rows": empty_text,
        "duplicate_interval_rows": sum(count - 1 for count in duplicate_keys.values() if count > 1),
        "duration_seconds": {
            "min": min(durations),
            "median": statistics.median(durations),
            "max": max(durations),
            "total": sum(durations),
        },
        "text_characters": {
            "min": min(len(row.get(text_column) or "") for row in rows),
            "mean": sum(len(row.get(text_column) or "") for row in rows) / len(rows),
            "max": max(len(row.get(text_column) or "") for row in rows),
        },
    }


def align_source_translation(
    source_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    *,
    minimum_interval_coverage: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair rows by talk and chronological ordinal only when interval overlap supports it.

    TEDxTN publishes separate source and translation CSVs whose segment boundaries
    are not identical. Pairing is therefore intentionally conservative: a talk
    with a different number of rows is withheld, and each ordinal pair must have
    sufficient overlap of both intervals. The resulting candidates still need
    manual spot checks.
    """
    source_by_talk: dict[str, list[dict[str, str]]] = defaultdict(list)
    target_by_talk: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        source_by_talk[row["audio_filename"]].append(row)
    for row in target_rows:
        target_by_talk[row["audio_filename"]].append(row)

    if not 0 < minimum_interval_coverage <= 1:
        raise ValueError("minimum_interval_coverage must be in (0, 1]")

    pairs: list[dict[str, Any]] = []
    skipped_talks: dict[str, str] = {}
    overlap_ratios: list[float] = []
    target_overlap_ratios: list[float] = []
    for talk in sorted(source_by_talk.keys() | target_by_talk.keys()):
        source = sorted(source_by_talk.get(talk, []), key=lambda r: (float(r["start_time"]), float(r["end_time"])))
        target = sorted(target_by_talk.get(talk, []), key=lambda r: (float(r["start_time"]), float(r["end_time"])))
        if len(source) != len(target):
            skipped_talks[talk] = f"segment_count_mismatch:{len(source)}:{len(target)}"
            continue
        talk_pairs: list[dict[str, Any]] = []
        talk_overlap_ratios: list[float] = []
        for ordinal, (source_row, target_row) in enumerate(zip(source, target)):
            source_start, source_end = float(source_row["start_time"]), float(source_row["end_time"])
            target_start, target_end = float(target_row["start_time"]), float(target_row["end_time"])
            intersection = max(0.0, min(source_end, target_end) - max(source_start, target_start))
            source_duration = source_end - source_start
            overlap_ratio = intersection / source_duration
            target_duration = target_end - target_start
            target_overlap_ratio = intersection / target_duration
            if (
                overlap_ratio < minimum_interval_coverage
                or target_overlap_ratio < minimum_interval_coverage
            ):
                skipped_talks[talk] = (
                    f"insufficient_interval_overlap_at_ordinal:{ordinal}"
                )
                break
            talk_overlap_ratios.append(overlap_ratio)
            talk_pairs.append(
                {
                    "audio_filename": talk,
                    "ordinal": ordinal,
                    "start_time": source_start,
                    "end_time": source_end,
                    "translation_start_time": target_start,
                    "translation_end_time": target_end,
                    "transcription": source_row["transcription"],
                    "translation": target_row["translation"],
                    "source_overlap_ratio": overlap_ratio,
                    "target_overlap_ratio": target_overlap_ratio,
                }
            )
        else:
            pairs.extend(talk_pairs)
            overlap_ratios.extend(talk_overlap_ratios)
            target_overlap_ratios.extend(
                pair["target_overlap_ratio"] for pair in talk_pairs
            )

    return pairs, {
        "candidate_pairs": len(pairs),
        "skipped_talks": skipped_talks,
        "mean_source_interval_overlap": (
            sum(overlap_ratios) / len(overlap_ratios) if overlap_ratios else None
        ),
        "minimum_source_interval_overlap": min(overlap_ratios) if overlap_ratios else None,
        "mean_target_interval_overlap": (
            sum(target_overlap_ratios) / len(target_overlap_ratios)
            if target_overlap_ratios
            else None
        ),
        "empty_translation_candidates": sum(
            not pair["translation"].strip() for pair in pairs
        ),
        "note": "Ordinal/time-overlap candidates require manual spot checks before experiments.",
    }


def inspect_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Inspect local source/translation CSVs and split leakage."""
    root = Path(dataset_dir)
    report: dict[str, Any] = {
        "dataset": "fbougares/TEDxTN",
        "revision": HF_REVISION,
        "splits": {},
        "notes": [],
    }
    source_talks: dict[str, set[str]] = {}
    for split in SPLITS:
        source_path = root / f"{split}.csv"
        target_path = root / f"{split}.en.csv"
        if not source_path.is_file():
            raise FileNotFoundError(f"Required source metadata is missing: {source_path}")
        source_rows = read_csv_rows(source_path)
        source_summary = validate_rows(
            source_rows, text_column="transcription", allow_empty_text=True
        )
        source_talks[split] = {row["audio_filename"] for row in source_rows}
        split_report: dict[str, Any] = {"source": source_summary}

        if target_path.is_file():
            target_rows = read_csv_rows(target_path)
            split_report["translation"] = validate_rows(
                target_rows, text_column="translation", allow_empty_text=True
            )
            _, alignment = align_source_translation(source_rows, target_rows)
            split_report["alignment"] = alignment
        else:
            split_report["translation"] = None
            split_report["alignment"] = None
        report["splits"][split] = split_report

    report["talk_leakage"] = {
        f"{left}_{right}": sorted(source_talks[left] & source_talks[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    report["notes"].extend(
        [
            "The public repository contains segment metadata and talk URLs, not audio files.",
            "Sample rates and audio integrity cannot be measured until licensed audio is supplied.",
            "No speaker IDs, region labels, or dialect-variant labels occur in the inspected CSV schema.",
        ]
    )
    return report


def load_audio_segment(
    audio_root: str | Path,
    audio_filename: str,
    start_time: float,
    end_time: float,
    *,
    target_sample_rate: int = 16_000,
):
    """Load one mono WAV interval as float32 samples at the requested rate."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Audio loading requires soundfile and numpy; install requirements.txt first."
        ) from exc

    path = Path(audio_root) / f"{audio_filename}.wav"
    if Path(audio_filename).name != audio_filename or audio_filename in {"", ".", ".."}:
        raise ValueError(f"Audio filename must be a talk ID, not a path: {audio_filename!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing talk audio: {path}")
    if (
        not math.isfinite(start_time)
        or not math.isfinite(end_time)
        or start_time < 0
        or end_time <= start_time
    ):
        raise ValueError(f"Invalid audio interval: {start_time} to {end_time}")
    if not isinstance(target_sample_rate, int) or target_sample_rate <= 0:
        raise ValueError(f"target_sample_rate must be a positive integer, got {target_sample_rate!r}")

    with sf.SoundFile(path) as audio_file:
        source_rate = audio_file.samplerate
        first_frame = round(start_time * source_rate)
        last_frame = round(end_time * source_rate)
        if last_frame > len(audio_file):
            raise ValueError(
                f"Segment ends at {end_time:.3f}s but {path.name} is "
                f"{len(audio_file) / source_rate:.3f}s long"
            )
        audio_file.seek(first_frame)
        samples = audio_file.read(last_frame - first_frame, dtype="float32", always_2d=True)
    if samples.size == 0:
        raise ValueError(f"Audio interval produced no samples: {path} [{start_time}, {end_time}]")

    mono = samples.mean(axis=1)
    if source_rate != target_sample_rate:
        from scipy.signal import resample_poly

        divisor = math.gcd(source_rate, target_sample_rate)
        mono = resample_poly(
            mono, target_sample_rate // divisor, source_rate // divisor
        ).astype(np.float32, copy=False)
    return mono.astype(np.float32, copy=False)


def inspect_audio_directory(
    audio_root: str | Path,
    source_rows_by_split: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    """Audit expected local WAV files, headers, durations, and a real segment decode."""
    try:
        import numpy  # noqa: F401
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Audio inventory requires numpy and soundfile; install the project requirements."
        ) from exc

    root = Path(audio_root).expanduser().resolve()
    rows_by_talk: dict[str, list[tuple[str, int, dict[str, str]]]] = defaultdict(list)
    for split, rows in source_rows_by_split.items():
        for row_number, row in enumerate(rows, start=2):
            rows_by_talk[row["audio_filename"]].append((split, row_number, row))

    missing_files: list[str] = []
    invalid_files: list[dict[str, str]] = []
    out_of_bounds_segments: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    sample_rate_counts: Counter[int] = Counter()
    split_status: dict[str, dict[str, Any]] = {
        split: {
            "talks": set(),
            "missing_files": set(),
            "invalid_files": {},
            "out_of_bounds_segments": [],
        }
        for split in source_rows_by_split
    }

    for talk in sorted(rows_by_talk):
        relative_path = f"{talk}.wav"
        talk_rows = rows_by_talk[talk]
        for split, _, _ in talk_rows:
            split_status[split]["talks"].add(talk)

        if Path(talk).name != talk or talk in {"", ".", ".."}:
            issue = {"file": relative_path, "error": "talk ID must not contain a path"}
            invalid_files.append(issue)
            for split, _, _ in talk_rows:
                split_status[split]["invalid_files"][relative_path] = issue
            continue
        path = root / relative_path

        if not path.is_file():
            missing_files.append(relative_path)
            for split, _, _ in talk_rows:
                split_status[split]["missing_files"].add(relative_path)
            continue

        try:
            info = sf.info(path)
            if info.samplerate <= 0 or info.channels <= 0 or info.frames <= 0:
                raise ValueError("invalid WAV metadata")
            duration_seconds = info.frames / info.samplerate
        except (OSError, RuntimeError, ValueError) as exc:
            issue = {"file": relative_path, "error": str(exc)}
            invalid_files.append(issue)
            for split, _, _ in talk_rows:
                split_status[split]["invalid_files"][relative_path] = issue
            continue

        sample_rate_counts[info.samplerate] += 1
        inventory.append(
            {
                "file": relative_path,
                "sample_rate_hz": info.samplerate,
                "channels": info.channels,
                "duration_seconds": duration_seconds,
            }
        )
        for split, row_number, row in talk_rows:
            try:
                end_time = float(row["end_time"])
                beyond_end = (
                    not math.isfinite(end_time)
                    or round(end_time * info.samplerate) > info.frames
                )
            except (TypeError, ValueError):
                beyond_end = True
            if beyond_end:
                issue = {
                    "split": split,
                    "csv_row": row_number,
                    "file": relative_path,
                    "segment_end_seconds": row["end_time"],
                    "audio_duration_seconds": duration_seconds,
                }
                out_of_bounds_segments.append(issue)
                split_status[split]["out_of_bounds_segments"].append(issue)

        split, _, example_row = talk_rows[0]
        try:
            samples = load_audio_segment(
                root,
                talk,
                float(example_row["start_time"]),
                float(example_row["end_time"]),
            )
            del samples
        except (OSError, RuntimeError, ValueError, ImportError) as exc:
            issue = {"file": relative_path, "error": f"segment decode failed: {exc}"}
            invalid_files.append(issue)
            for row_split, _, _ in talk_rows:
                split_status[row_split]["invalid_files"][relative_path] = issue

    finalized_splits = {}
    for split, details in split_status.items():
        finalized_splits[split] = {
            "talks": len(details["talks"]),
            "missing_files": sorted(details["missing_files"]),
            "invalid_files": list(details["invalid_files"].values()),
            "out_of_bounds_segments": details["out_of_bounds_segments"],
            "ready": not (
                details["missing_files"]
                or details["invalid_files"]
                or details["out_of_bounds_segments"]
            ),
        }

    return {
        "audio_directory": str(root),
        "expected_format": "one readable WAV file named <audio_filename>.wav per talk",
        "expected_talks": len(rows_by_talk),
        "present_readable_files": len(inventory),
        "missing_files": missing_files,
        "invalid_files": invalid_files,
        "out_of_bounds_segments": out_of_bounds_segments,
        "sample_rate_counts": dict(sorted(sample_rate_counts.items())),
        "files": inventory,
        "splits": finalized_splits,
        "ready": not (missing_files or invalid_files or out_of_bounds_segments),
        "audio_decode_check": (
            "one metadata-selected segment decoded for every readable WAV"
            if inventory and not invalid_files
            else "not completed for missing or invalid WAV files"
        ),
    }
