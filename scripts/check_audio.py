"""Audit user-authorized local TEDxTN WAV files without downloading audio."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import inspect_audio_directory, read_csv_rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path(os.environ.get("TEDXTN_AUDIO_DIR", ROOT / "data" / "audio")),
        help="Directory containing <audio_filename>.wav files. Nothing is downloaded.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=ROOT / "data" / "metadata",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "metrics" / "audio_inventory.json",
    )
    args = parser.parse_args()

    try:
        rows_by_split = {
            split: read_csv_rows(args.metadata_dir / f"{split}.csv")
            for split in ("train", "dev", "test")
        }
        report = inspect_audio_directory(args.audio_dir, rows_by_split)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Audio inventory failed: {exc}\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Audio inventory written to {args.output}")
    print(
        f"Ready: {report['ready']}; present WAV files: "
        f"{report['present_readable_files']}/{report['expected_talks']}"
    )
    if report["missing_files"]:
        print("Missing expected files:")
        for filename in report["missing_files"]:
            print(f"  {filename}")
    if report["invalid_files"]:
        print("Invalid or undecodable files:")
        for issue in report["invalid_files"]:
            print(f"  {issue['file']}: {issue['error']}")
    if report["out_of_bounds_segments"]:
        print(
            "Source segment intervals exceeding audio duration: "
            f"{len(report['out_of_bounds_segments'])}"
        )
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
