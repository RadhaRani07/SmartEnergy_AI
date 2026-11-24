
from pathlib import Path
import sys

import streamlit as st
import pandas as pd
import numpy as np

# Make the src folder accessible
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict_energy_demand import load_model, build_features


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SmartEnergy AI",
    page_icon="⚡",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_model()


# --------------------------------------------------
# INPUT VALIDATION
# --------------------------------------------------

def prepare_uploaded_data(uploaded_file):

    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif uploaded_file.name.lower().endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)

    else:
        raise ValueError(
            "Please upload a CSV or Parquet file."
        )

    required = ["Timestamp", "Total"]

    if not all(col in df.columns for col in required):
        raise ValueError(
            "File must contain Timestamp and Total columns."
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
            "Missing or invalid Timestamp/Total values found."
        )

    df = df.sort_values("Timestamp").reset_index(drop=True)

    if df["Timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps detected."
        )

    if len(df) < 168:
        raise ValueError(
            "At least 168 hourly observations are required."
        )

    differences = df["Timestamp"].diff().dropna()

    if not (differences == pd.Timedelta(hours=1)).all():
        raise ValueError(
            "Data must contain continuous hourly timestamps."
        )

    if not np.isfinite(df["Total"]).all():
        raise ValueError(
            "Non-finite energy values detected."
        )

    if (df["Total"] <= 0).any():
        raise ValueError(
            "Energy demand values must be positive."
        )

    return df


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("⚡ SmartEnergy AI")

st.subheader(
    "AI-Powered Energy Demand Forecasting System"
)

st.write(
    "Forecast the next hour's energy demand "
    "using a trained Machine Learning model."
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("SmartEnergy AI")

st.sidebar.info(
    "Model: Ridge Regression\n\n"
    "Forecast Horizon: 1 Hour\n\n"
    "Required History: 168 Hours"
)

st.sidebar.markdown("---")

st.sidebar.write(
    "Upload hourly energy data to generate a forecast."
)


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

st.header("Upload Energy Dataset")

uploaded_file = st.file_uploader(
    "Choose CSV or Parquet file",
    type=["csv", "parquet"]
)


# --------------------------------------------------
# FORECASTING
# --------------------------------------------------

if uploaded_file is not None:

    try:

        with st.spinner("Validating uploaded data..."):

            history = prepare_uploaded_data(uploaded_file)

        st.success("Dataset successfully validated!")

        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Observations",
                f"{len(history):,}"
            )

        with col2:
            st.metric(
                "Latest Demand",
                f"{history['Total'].iloc[-1]:,.2f}"
            )

        with col3:
            st.metric(
                "Forecast Horizon",
                "1 Hour"
            )

        st.write(
            "**Latest Observation:**",
            history["Timestamp"].iloc[-1]
        )

        st.divider()

        st.subheader("Historical Energy Demand")

        chart_data = history.set_index("Timestamp")

        st.line_chart(
            chart_data["Total"],
            x_label="Timestamp",
            y_label="Energy Demand"
        )

        # Generate forecast
        artifact = get_model()

        X, target_timestamp = build_features(history)

        X = X[artifact["features"]]

        prediction = float(
            artifact["model"].predict(X)[0]
        )

        latest_demand = float(
            history["Total"].iloc[-1]
        )

        change = prediction - latest_demand

        percentage_change = (
            (change / latest_demand) * 100
        )

        st.divider()

        st.header("AI Forecast Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Predicted Energy Demand",
                f"{prediction:,.2f}",
                f"{change:+,.2f} ({percentage_change:+.2f}%)"
            )

        with result_col2:

            st.metric(
                "Forecast Timestamp",
                target_timestamp.strftime(
                    "%Y-%m-%d %H:%M"
                )
            )

        st.subheader("Forecast Summary")

        forecast_result = pd.DataFrame({
            "Metric": [
                "Model",
                "Forecast Horizon",
                "Latest Observed Timestamp",
                "Forecast Target Timestamp",
                "Latest Observed Demand",
                "Predicted Demand",
                "Input Observations",
                "Features Used"
            ],
            "Value": [
                artifact["model_name"],
                "1 Hour",
                str(history["Timestamp"].iloc[-1]),
                str(target_timestamp),
                round(latest_demand, 2),
                round(prediction, 2),
                len(history),
                len(artifact["features"])
            ]
        })

        st.dataframe(
            forecast_result,
            use_container_width=True,
            hide_index=True
        )

        csv_data = forecast_result.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Forecast Result",
            data=csv_data,
            file_name="smartenergy_forecast.csv",
            mime="text/csv"
        )

        st.info(
            "This is a model-based energy demand forecast. "
            "It is not a verified energy-cost or savings estimate."
        )

    except Exception as error:

        st.error(f"Unable to generate forecast: {error}")


else:

    st.info(
        "Please upload a CSV or Parquet file "
        "containing Timestamp and Total columns."
    )

    st.write(
        "At least 168 continuous hourly observations "
        "are required."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "SmartEnergy AI | Machine Learning Based "
    "Energy Demand Forecasting System"
)
