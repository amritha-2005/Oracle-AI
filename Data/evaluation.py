from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "trained_models" / "random_forest_best.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "climate_features.csv"


def evaluate_model(data_path: str | Path | None = None, model_path: str | Path | None = None) -> dict[str, Any]:
    csv_path = Path(data_path) if data_path is not None else DATA_PATH
    model_file = Path(model_path) if model_path is not None else MODEL_PATH

    if not csv_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {csv_path}")
    if not model_file.exists():
        raise FileNotFoundError(f"Trained model not found: {model_file}")

    frame = pd.read_csv(csv_path)
    model = joblib.load(model_file)

    feature_columns = [
        "rainfall_mm",
        "temperature_c",
        "relative_humidity_pct",
        "tp",
        "avg_tprate",
        "avg_lsprate",
        "avg_cpr",
        "avg_ie",
        "avg_rorwe",
    ]
    selected = [column for column in feature_columns if column in frame.columns]

    if not selected:
        raise ValueError("No model feature columns were found in the feature CSV.")

    target_name = "risk_score" if "risk_score" in frame.columns else "target_risk"
    if target_name not in frame.columns:
        frame[target_name] = (
            frame["rainfall_mm"] * 0.35
            + frame["temperature_c"] * 0.3
            + frame["relative_humidity_pct"] * 0.2
            + frame.get("avg_tprate", 0) * 0.15
        ) / 100.0

    predictions = model.predict(frame[selected])
    actual = frame[target_name].astype(float)

    return {
        "status": "success",
        "rows_evaluated": int(len(frame)),
        "metrics": {
            "mae": float(mean_absolute_error(actual, predictions)),
            "mse": float(mean_squared_error(actual, predictions)),
            "rmse": float(mean_squared_error(actual, predictions, squared=False)),
            "r2": float(r2_score(actual, predictions)),
        },
    }


def main() -> None:
    result = evaluate_model()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
