from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def list_raw_data() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"imd": [], "era5": []}

    if (RAW_DIR / "imd").exists():
        result["imd"] = sorted(str(p.name) for p in (RAW_DIR / "imd").rglob("*") if p.is_file())
    if (RAW_DIR / "era5").exists():
        result["era5"] = sorted(str(p.name) for p in (RAW_DIR / "era5").rglob("*") if p.is_file())
    return result


def get_processed_files() -> list[str]:
    if not PROCESSED_DIR.exists():
        return []
    return sorted(str(p.name) for p in PROCESSED_DIR.iterdir() if p.is_file())


def _load_era5_dataset() -> xr.Dataset:
    era5_dir = RAW_DIR / "era5"
    if not era5_dir.exists():
        raise FileNotFoundError(f"ERA5 dataset directory does not exist: {era5_dir}")

    candidates = sorted(era5_dir.rglob("*.nc"))
    if not candidates:
        raise FileNotFoundError(f"No NetCDF files found in {era5_dir}")

    preferred = [
        path for path in candidates
        if "avg" in path.name.lower() or "average" in path.name.lower() or "stepType-avg" in path.name.lower()
    ]
    dataset_path = preferred[0] if preferred else candidates[0]

    try:
        return xr.open_dataset(dataset_path)
    except Exception as exc:  # pragma: no cover - runtime validation path
        raise ValueError(f"Unable to open ERA5 dataset: {dataset_path}") from exc


def _extract_imd_rows(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"IMD file not found: {file_path}")

    records: list[dict[str, Any]] = []
    pattern = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$")

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = pattern.match(stripped)
            if not match:
                continue

            rainfall, temperature, humidity = [float(value) for value in match.groups()]
            parts = file_path.stem.split("_")
            lat = float(parts[-2]) if len(parts) >= 2 else float("nan")
            lon = float(parts[-1]) if len(parts) >= 2 else float("nan")

            records.append(
                {
                    "rainfall_mm": rainfall,
                    "temperature_c": temperature,
                    "relative_humidity_pct": humidity,
                    "latitude": lat,
                    "longitude": lon,
                }
            )

    if not records:
        raise ValueError(f"No valid rows found in IMD file: {file_path}")

    frame = pd.DataFrame(records)
    numeric_cols = ["rainfall_mm", "temperature_c", "relative_humidity_pct", "latitude", "longitude"]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["rainfall_mm", "temperature_c", "relative_humidity_pct"]).reset_index(drop=True)


def _read_imd_data(max_files: int | None = None) -> pd.DataFrame:
    imd_root = RAW_DIR / "imd"
    if not imd_root.exists():
        raise FileNotFoundError(f"IMD directory not found: {imd_root}")

    weather_dir = imd_root / "IMD_MetData"
    if not weather_dir.exists():
        weather_dir = imd_root

    files = sorted(path for path in weather_dir.rglob("*") if path.is_file() and path.suffix not in {".zip", ".nc"})
    if not files:
        raise FileNotFoundError(f"No IMD text files found under {weather_dir}")

    if max_files is not None:
        files = files[:max_files]

    records: list[dict[str, Any]] = []
    for file_path in files:
        try:
            frame = _extract_imd_rows(file_path)
        except ValueError:
            continue

        if frame.empty:
            continue

        file_summary = frame.groupby(["latitude", "longitude"], as_index=False)[["rainfall_mm", "temperature_c", "relative_humidity_pct"]].mean()
        records.append(file_summary)

    if not records:
        raise ValueError(f"No usable IMD records were parsed from {weather_dir}")

    return pd.concat(records, ignore_index=True)


def _summarize_era5_dataset(dataset: xr.Dataset) -> pd.DataFrame:
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("ERA5 source must be an xarray Dataset instance.")

    required_dims = {"valid_time", "latitude", "longitude"}
    if not required_dims.issubset(dataset.dims):
        missing = sorted(required_dims - set(dataset.dims))
        raise ValueError(f"ERA5 dataset missing required dimensions: {missing}")

    if not dataset.data_vars:
        raise ValueError("ERA5 dataset does not contain any data variables.")

    variable_names = list(dataset.data_vars)
    summary = dataset[variable_names].mean(dim="valid_time").to_dataframe().reset_index()
    summary = summary[["latitude", "longitude", *variable_names]]
    summary["latitude"] = pd.to_numeric(summary["latitude"], errors="coerce")
    summary["longitude"] = pd.to_numeric(summary["longitude"], errors="coerce")
    return summary.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def build_feature_table(max_files: int | None = None) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    era5_dataset = _load_era5_dataset()
    try:
        era5_frame = _summarize_era5_dataset(era5_dataset)
    finally:
        era5_dataset.close()

    if era5_frame.empty:
        raise ValueError("ERA5 summary dataset is empty.")

    imd_frame = _read_imd_data(max_files=max_files)
    if imd_frame.empty:
        raise ValueError("IMD summary dataset is empty.")

    imd_frame["latitude"] = pd.to_numeric(imd_frame["latitude"], errors="coerce")
    imd_frame["longitude"] = pd.to_numeric(imd_frame["longitude"], errors="coerce")
    imd_frame = imd_frame.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    era_latitudes = np.unique(era5_frame["latitude"].to_numpy())
    era_longitudes = np.unique(era5_frame["longitude"].to_numpy())

    def nearest_value(value: float, candidates: np.ndarray) -> float:
        if value is None or np.isnan(value):
            return float("nan")
        return float(candidates[np.abs(candidates - value).argmin()])

    imd_frame["latitude_match"] = imd_frame["latitude"].map(lambda value: nearest_value(value, era_latitudes))
    imd_frame["longitude_match"] = imd_frame["longitude"].map(lambda value: nearest_value(value, era_longitudes))

    merged = pd.merge(
        imd_frame,
        era5_frame,
        left_on=["latitude_match", "longitude_match"],
        right_on=["latitude", "longitude"],
        how="inner",
    )

    merged = merged.drop(columns=["latitude_x", "longitude_x", "latitude_y", "longitude_y", "latitude_match", "longitude_match"])
    merged = merged.rename(columns={"latitude": "latitude", "longitude": "longitude"})

    if merged.empty:
        raise ValueError("Merged climate dataset is empty after matching IMD records to ERA5 grid cells.")

    for column in ["rainfall_mm", "temperature_c", "relative_humidity_pct"]:
        merged[column] = pd.to_numeric(merged.get(column, 0), errors="coerce").fillna(0)
    merged = merged.fillna(0)

    target_path = PROCESSED_DIR / "climate_features.csv"
    merged.to_csv(target_path, index=False)
    return merged


if __name__ == "__main__":
    df = build_feature_table()
    print(f"Built climate feature dataset with {len(df)} rows and {len(df.columns)} columns.")
