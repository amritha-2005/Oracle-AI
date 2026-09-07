from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "trained_models" / "random_forest_best.pkl"


def _default_feature_map() -> dict[str, float]:
    return {
        "rainfall_mm": 0.0,
        "temperature_c": 25.0,
        "relative_humidity_pct": 60.0,
        "tp": 0.0,
        "avg_tprate": 0.0,
        "avg_lsprate": 0.0,
        "avg_cpr": 0.0,
        "avg_ie": 0.0,
        "avg_rorwe": 0.0,
    }


def predict_batch(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Prediction batch DataFrame is empty.")

    features = [column for column in _default_feature_map() if column in frame.columns]
    if not features:
        raise ValueError("Input batch DataFrame does not contain the expected feature columns.")

    model = joblib.load(MODEL_PATH)
    preds = model.predict(frame[features])
    result = frame.copy()
    result["predicted_risk_score"] = preds
    return {
        "status": "success",
        "rows_processed": int(len(result)),
        "predictions": result.to_dict(orient="records"),
    }


def load_feature_table(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Feature CSV is empty.")
    return frame


def main() -> None:
    feature_csv = BASE_DIR / "data" / "processed" / "climate_features.csv"
    if not feature_csv.exists():
        print(json.dumps({"status": "warning", "message": "No processed feature CSV is available yet."}))
        return

    frame = load_feature_table(feature_csv)
    result = predict_batch(frame)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
