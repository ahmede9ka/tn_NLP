"""Download or inspect TEDxTN metadata without downloading audio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import HF_REVISION_URL, inspect_dataset  # noqa: E402


METADATA_FILES = (
    "train.csv",
    "dev.csv",
    "test.csv",
    "train.en.csv",
    "dev.en.csv",
    "test.en.csv",
    "TEDxTN_URLS.txt",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "metadata")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "metrics" / "dataset_inspection.json")
    parser.add_argument(
        "--download-metadata",
        action="store_true",
        help="Fetch the public CSV/URL metadata when it is missing; audio is never downloaded.",
    )
    args = parser.parse_args()

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.download_metadata:
            for name in METADATA_FILES:
                target = args.dataset_dir / name
                if not target.is_file():
                    urlretrieve(f"{HF_REVISION_URL}/{name}", target)
        report = inspect_dataset(args.dataset_dir)
    except (OSError, ValueError, URLError) as exc:
        parser.exit(2, f"Dataset inspection failed: {exc}\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Inspection report written to {args.output}")
    for split, details in report["splits"].items():
        source = details["source"]
        alignment = details["alignment"]
        print(
            f"{split}: {source['rows']:,} source segments, "
            f"{source['talks']} talks; "
            f"{alignment['candidate_pairs'] if alignment else 'no'} translation candidates"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
