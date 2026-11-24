# SmartEnergy AI
### AI-Powered Energy Demand Forecasting System

SmartEnergy AI is a machine-learning-based energy demand forecasting project that uses historical hourly electricity consumption data to predict demand one hour ahead.

The project combines data quality investigation, chronological time-series evaluation, baseline forecasting, Ridge Regression, and a reusable prediction pipeline.

## Project Overview

The system uses historical energy demand patterns to generate short-term forecasts.

**Key capabilities:**
- Historical energy data analysis
- Data quality and missing-value investigation
- Time-series feature engineering
- Chronological train-validation-test evaluation
- Baseline forecasting comparison
- Ridge Regression forecasting
- Reusable one-hour-ahead prediction pipeline

## Dataset

The project uses the SEWA hourly commercial electricity dataset.

- Organization: Sharjah Electricity and Water Authority (SEWA)
- Coverage: 2020-2021
- Original observations: 17,544 hourly records
- Forecasting dataset: 16,022 eligible rows
- Forecast horizon: 1 hour ahead

The forecasting dataset retains the original demand values.

**Dataset note:** The metadata identifies SEWA in Sharjah, UAE. It does not establish that the measurements represent a specific Dubai office building.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- PyArrow
- Joblib
- Ridge Regression
- StandardScaler
- PowerShell

## Machine Learning Approach

### Forecasting Features

The model uses eight features:

1. `lag_1`
2. `lag_24`
3. `lag_168`
4. `rolling_mean_24`
5. `hour`
6. `day_of_week`
7. `month`
8. `is_weekend`

The target is electricity demand at the next hourly timestamp.

### Model

- Algorithm: Ridge Regression
- Regularization parameter: `alpha = 0.01`
- Preprocessing: StandardScaler
- Training strategy: Train + Validation refit
- Evaluation: Chronological holdout test set

### Chronological Split

| Dataset | Records |
|---|---:|
| Training + Validation | 13,612 |
| Testing | 2,410 |
| Total | 16,022 |

The test period covers September-December 2021.

## Model Performance

The selected Ridge model was evaluated against two forecasting baselines.

| Metric | Ridge | Persistence | Seasonal Naive |
|---|---:|---:|---:|
| MAE | 44,031.48 | 50,153.11 | 59,713.13 |
| RMSE | 119,465.42 | 126,016.99 | 132,356.52 |
| MAPE | 2.8400% | 3.2128% | 3.9771% |
| WAPE | **3.0105%** | 3.4291% | 4.0827% |

Ridge achieved a lower test WAPE than both baseline models.

These results describe performance on the defined historical test period and do not guarantee future forecasting accuracy.

## Project Structure

```text
SmartEnergy_AI/
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ SEWA-1H.parquet
â”‚   â””â”€â”€ SmartEnergy_1H_Forecasting_Dataset_v2.parquet
â”‚
â”œâ”€â”€ models/
â”‚   â””â”€â”€ ridge_energy_forecasting_v1.joblib
â”‚
â”œâ”€â”€ reports/
â”‚   â””â”€â”€ ridge_model_test_evaluation.json
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ train_ridge_model.py
â”‚   â””â”€â”€ predict_energy_demand.py
â”‚
â”œâ”€â”€ venv/
â””â”€â”€ README.md
```

Additional data-quality audit scripts and reports are maintained in the project folders.

## Setup

Clone the repository:

```bash
git clone https://github.com/RadhaRani07/SmartEnergy_AI.git
cd SmartEnergy_AI
```

Replace `YOUR_USERNAME` with your GitHub username.

Create a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install pandas numpy scikit-learn pyarrow joblib
```

## Run Prediction

The prediction script accepts a CSV or Parquet file containing:

- `Timestamp`
- `Total`

The input must contain at least 168 continuous hourly observations.

Example:

```powershell
python .\src\predict_energy_demand.py --input .\data\prediction_test_input.csv
```

The script loads the saved Ridge model and generates a one-hour-ahead demand forecast.

## Data Quality and Limitations

Data-quality investigation identified missing readings, discrepancies, and a substantial distribution shift between 2020 and 2021.

Important limitations:

- The root cause of the distribution shift remains unresolved.
- Original missing readings have not been treated as verified observations.
- No unsupported source-data corrections are assumed.
- Forecasting accuracy may change under future distribution shifts.
- No electricity tariff data is available to calculate verified costs or savings.
- This is a forecasting prototype, not a deployed production energy-management system.

## Future Improvements

- Interactive forecasting dashboard
- Forecast visualization and historical comparisons
- Additional forecasting model evaluation
- Monitoring of prediction errors and distribution shifts
- Tariff-aware cost estimation when reliable tariff data becomes available

## Project Status

**Current status:** Working forecasting prototype.


## 🚀 Live Demo

Try the SmartEnergy AI application here:

🔗 **[Launch SmartEnergy AI](https://smartenergyai-wr6xhnm8n9k6dskdphueuv.streamlit.app/)**

The application allows users to:
- Upload hourly energy data in CSV or Parquet format.
- View historical energy demand.
- Generate a one-hour-ahead energy demand forecast.
- Download forecast results.

**Input Requirements:**
- File format: CSV or Parquet
- Required columns: `Timestamp` and `Total`
- Minimum history: 168 continuous hourly observations.

**Model:** Ridge Regression


The Ridge model has been trained, saved, reloaded, and tested through a one-hour-ahead prediction pipeline.

---

Developed as an AI/ML engineering portfolio project.
