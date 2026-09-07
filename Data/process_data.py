from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ml_pipeline.data_processing.dataset_loader import (
    build_feature_table,
    get_processed_files,
    list_raw_data,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def run_processing(output_dir: str | Path | None = None, max_files: int | None = None) -> dict[str, Any]:
    target_dir = Path(output_dir) if output_dir is not None else PROCESSED_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    raw_summary = list_raw_data()
    merged = build_feature_table(max_files=max_files)
    output_path = target_dir / "climate_features.csv"
    merged.to_csv(output_path, index=False)

    return {
        "status": "success",
        "raw_files": raw_summary,
        "rows": int(len(merged)),
        "columns": list(merged.columns),
        "output_file": str(output_path),
        "processed_files": get_processed_files(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed climate feature CSVs from raw IMD and ERA5 sources.")
    parser.add_argument("--output-dir", type=str, default=str(PROCESSED_DIR), help="Directory for processed CSV output.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on IMD files processed for validation runs.")
    args = parser.parse_args()

    result = run_processing(args.output_dir, max_files=args.max_files)
    print(f"Processed {result['rows']} rows into {result['output_file']}")
    print(f"Columns: {', '.join(result['columns'])}")


if __name__ == "__main__":
    main()
