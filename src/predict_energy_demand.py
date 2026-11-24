
from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ridge_energy_forecasting_v1.joblib"
)

REQUIRED_HISTORY = 168


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    return artifact


def prepare_input(path):

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        )

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)

    elif file_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(file_path)

    else:
        raise ValueError(
            "Input must be a CSV or Parquet file."
        )

    required = ["Timestamp", "Total"]

    if not all(col in df.columns for col in required):
        raise ValueError(
            "Input must contain Timestamp and Total columns."
        )

    df = df[required].copy()

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    df["Total"] = pd.to_numeric(
        df["Total"],
        errors="coerce"
    )

    if df.isna().any().any():
        raise ValueError(
            "Missing or invalid Timestamp/Total values detected."
        )

    df = df.sort_values("Timestamp").reset_index(drop=True)

    if df["Timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps detected."
        )

    if len(df) < REQUIRED_HISTORY:
        raise ValueError(
            f"At least {REQUIRED_HISTORY} observations are required."
        )

    timestamps = df["Timestamp"]

    differences = timestamps.diff().dropna()

    if not (differences == pd.Timedelta(hours=1)).all():
        raise ValueError(
            "Input history must have continuous hourly timestamps."
        )

    if not np.isfinite(df["Total"]).all():
        raise ValueError("Non-finite Total values detected.")

    if (df["Total"] <= 0).any():
        raise ValueError("Total values must be positive.")

    return df


def build_features(history):

    values = history["Total"].to_numpy(dtype=float)

    latest_timestamp = history["Timestamp"].iloc[-1]

    target_timestamp = (
        latest_timestamp + pd.Timedelta(hours=1)
    )

    features = {
        "lag_1": values[-1],
        "lag_24": values[-24],
        "lag_168": values[-168],
        "rolling_mean_24": float(
            np.mean(values[-24:])
        ),
        "hour": target_timestamp.hour,
        "day_of_week": target_timestamp.day_of_week,
        "month": target_timestamp.month,
        "is_weekend": int(
            target_timestamp.day_of_week >= 5
        ),
    }

    return pd.DataFrame([features]), target_timestamp


def predict(input_path):

    artifact = load_model()

    history = prepare_input(input_path)

    X, target_timestamp = build_features(history)

    expected_features = artifact["features"]

    X = X[expected_features]

    prediction = float(
        artifact["model"].predict(X)[0]
    )

    print("\n" + "=" * 60)
    print("SMARTENERGY AI - ENERGY DEMAND FORECAST")
    print("=" * 60)

    print("\nModel:", artifact["model_name"])
    print("Forecast horizon: 1 hour")
    print("Latest observed timestamp:", history["Timestamp"].iloc[-1])
    print("Forecast target timestamp:", target_timestamp)

    print("\nLatest observed demand:", round(
        float(history["Total"].iloc[-1]), 2
    ))

    print("Predicted demand:", round(prediction, 2))

    print("\nInput observations:", len(history))
    print("Features used:", len(expected_features))

    print("\nNote:")
    print(
        "This is a model-based demand forecast, "
        "not a verified energy-cost or savings estimate."
    )

    print("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="SmartEnergy AI one-hour-ahead demand prediction"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV or Parquet containing Timestamp and Total"
    )

    args = parser.parse_args()

    predict(args.input)
