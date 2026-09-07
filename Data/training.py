from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "trained_models"
SCALER_DIR = BASE_DIR / "models" / "scalers"


FEATURE_COLUMNS = [
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


def _prepare_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Training data not found: {file_path}")

    frame = pd.read_csv(file_path)
    if frame.empty:
        raise ValueError("Processed climate dataset is empty.")

    feature_frame = frame.copy()
    for column in FEATURE_COLUMNS:
        if column not in feature_frame.columns:
            feature_frame[column] = 0.0

    target_candidates = ["risk_score", "climate_risk_score", "target_risk"]
    target_column = next((name for name in target_candidates if name in feature_frame.columns), None)
    if target_column is None:
        feature_frame["risk_score"] = (
            feature_frame["rainfall_mm"] * 0.35
            + feature_frame["temperature_c"] * 0.3
            + feature_frame["relative_humidity_pct"] * 0.2
            + feature_frame.get("avg_tprate", 0) * 0.15
        ) / 100.0
        target_column = "risk_score"

    return feature_frame, target_column


def train_model(data_path: str | Path | None = None) -> dict[str, Any]:
    if data_path is None:
        data_path = DATA_DIR / "climate_features.csv"
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"No processed climate dataset was found at {data_path}")

    frame, target_column = _prepare_dataset(data_path)
    features = [column for column in FEATURE_COLUMNS if column in frame.columns]
    X = frame[features]
    y = frame[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=300, random_state=42, max_depth=None, min_samples_leaf=1)),
        ]
    )

    model_pipeline.fit(X_train, y_train)
    predictions = model_pipeline.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mean_squared_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions, squared=False)),
        "r2": float(r2_score(y_test, predictions)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SCALER_DIR.mkdir(parents=True, exist_ok=True)

    model_file = MODEL_DIR / "random_forest_best.pkl"
    scaler_file = SCALER_DIR / "standard_scaler.pkl"

    joblib.dump(model_pipeline, model_file)
    joblib.dump(model_pipeline.named_steps["scaler"], scaler_file)

    return {
        "status": "success",
        "model_path": str(model_file),
        "scaler_path": str(scaler_file),
        "metrics": metrics,
        "feature_columns": features,
        "rows_used": int(len(frame)),
        "training_set_size": int(len(X_train)),
        "test_set_size": int(len(X_test)),
    }


def main() -> None:
    result = train_model()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
