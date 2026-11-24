
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "SmartEnergy_1H_Forecasting_Dataset_v2.parquet"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "ridge_energy_forecasting_v1.joblib"
REPORT_PATH = REPORT_DIR / "ridge_model_test_evaluation.json"

TARGET_COLUMN = "target"
TEST_START = pd.Timestamp("2021-09-01")

FEATURES = [
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

ALPHA = 0.01


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

print("=" * 65)
print("SMARTENERGY AI - OFFICIAL RIDGE MODEL TRAINING")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_parquet(DATA_PATH)

required_columns = FEATURES + [
    TARGET_COLUMN,
    "target_timestamp",
    "forecast_origin",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

if df[required_columns].isna().any().any():
    raise ValueError("Missing values detected. Training stopped.")

df["target_timestamp"] = pd.to_datetime(
    df["target_timestamp"]
)

if df["target_timestamp"].duplicated().any():
    raise ValueError("Duplicate target timestamps detected.")

if not df["target_timestamp"].is_monotonic_increasing:
    raise ValueError("Target timestamps are not chronological.")

print(f"\nDataset rows: {len(df)}")
print(f"Features: {len(FEATURES)}")


# --------------------------------------------------
# 3. CHRONOLOGICAL TRAIN + VALIDATION / TEST SPLIT
# --------------------------------------------------

train_validation = df[
    df["target_timestamp"] < TEST_START
].copy()

test = df[
    df["target_timestamp"] >= TEST_START
].copy()

if len(train_validation) != 13612 or len(test) != 2410:
    raise ValueError(
        "Unexpected split sizes. "
        f"Train+Validation={len(train_validation)}, "
        f"Test={len(test)}"
    )

X_train_validation = train_validation[FEATURES]
y_train_validation = train_validation[TARGET_COLUMN]

X_test = test[FEATURES]
y_test = test[TARGET_COLUMN]

print("\nCHRONOLOGICAL SPLIT")
print("-" * 65)
print("Train + Validation rows:", len(train_validation))
print("Test rows:", len(test))
print("Test starts:", test["target_timestamp"].min())
print("Test ends:", test["target_timestamp"].max())


# --------------------------------------------------
# 4. BUILD AND TRAIN RIDGE PIPELINE
# --------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=ALPHA)),
])

print("\nTraining Ridge model...")

model.fit(
    X_train_validation,
    y_train_validation
)

print("Training completed.")


# --------------------------------------------------
# 5. TEST PREDICTIONS
# --------------------------------------------------

ridge_predictions = model.predict(X_test)

persistence_predictions = X_test["lag_1"].to_numpy()
seasonal_predictions = X_test["lag_24"].to_numpy()


# --------------------------------------------------
# 6. METRIC CALCULATION
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mape = np.mean(
        np.abs((actual - predicted) / actual)
    ) * 100

    wape = (
        np.sum(np.abs(actual - predicted))
        / np.sum(np.abs(actual))
    ) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_percent": float(mape),
        "WAPE_percent": float(wape),
    }


ridge_metrics = calculate_metrics(
    y_test,
    ridge_predictions
)

persistence_metrics = calculate_metrics(
    y_test,
    persistence_predictions
)

seasonal_metrics = calculate_metrics(
    y_test,
    seasonal_predictions
)


# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

results = {
    "RIDGE": ridge_metrics,
    "PERSISTENCE": persistence_metrics,
    "SEASONAL_NAIVE": seasonal_metrics,
}

print("\nFINAL TEST RESULTS")
print("=" * 65)

for model_name, metrics in results.items():

    print(f"\n{model_name}")

    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")


# --------------------------------------------------
# 8. VERIFY AGAINST PREVIOUS EVALUATION
# --------------------------------------------------

expected_wape = 3.0105

actual_wape = ridge_metrics["WAPE_percent"]

print("\nREPRODUCTION CHECK")
print("-" * 65)
print(f"Expected Ridge WAPE: {expected_wape:.4f}%")
print(f"Reproduced Ridge WAPE: {actual_wape:.4f}%")

if abs(actual_wape - expected_wape) > 0.02:
    raise RuntimeError(
        "Ridge result differs from the earlier evaluation. "
        "Model saving stopped. Investigate before proceeding."
    )

print("Reproduction check: PASS")


# --------------------------------------------------
# 9. SAVE MODEL ARTIFACT
# --------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

artifact = {
    "model": model,
    "model_name": "Ridge Regression",
    "alpha": ALPHA,
    "features": FEATURES,
    "target_column": TARGET_COLUMN,
    "forecast_horizon": "1 hour ahead",
    "test_start": str(TEST_START),
    "training_rows": len(train_validation),
    "test_rows": len(test),
    "test_metrics": ridge_metrics,
}

joblib.dump(artifact, MODEL_PATH)


# --------------------------------------------------
# 10. SAVE TEST EVALUATION REPORT
# --------------------------------------------------

report = {
    "project": "SmartEnergy AI",
    "model": "Ridge Regression",
    "alpha": ALPHA,
    "preprocessing": "StandardScaler",
    "features": FEATURES,
    "dataset": DATA_PATH.name,
    "training_rows": len(train_validation),
    "test_rows": len(test),
    "test_period_start": str(test["target_timestamp"].min()),
    "test_period_end": str(test["target_timestamp"].max()),
    "metrics": results,
    "reproduction_check": "PASS",
    "model_artifact": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
}

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)


# --------------------------------------------------
# 11. FINAL STATUS
# --------------------------------------------------

print("\n" + "=" * 65)
print("MODEL TRAINING AND SAVING COMPLETED")
print("=" * 65)

print("\nSaved model:")
print(MODEL_PATH)

print("\nSaved evaluation report:")
print(REPORT_PATH)

print("\nOriginal source data was not modified.")
print("Forecasting dataset was not modified.")
