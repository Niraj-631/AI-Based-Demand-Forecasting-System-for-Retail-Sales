import os
import sys

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# Import the forecasting function.
# forecast.py now loads features.pkl only when a forecast is requested.
from src.forecast import forecast_demand


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Demand Forecasting",
    page_icon="🛒",
    layout="wide",
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load(
        os.path.join(
            PROJECT_ROOT,
            "models",
            "demand_model.pkl",
        )
    )

    family_mapping = joblib.load(
        os.path.join(
            PROJECT_ROOT,
            "models",
            "family_mapping.pkl",
        )
    )

    return model, family_mapping


# ============================================================
# LOAD MODEL PERFORMANCE
# ============================================================

@st.cache_data
def load_metrics():
    path = os.path.join(
        PROJECT_ROOT,
        "data",
        "final_model_comparison.csv",
    )

    return pd.read_csv(path)


# ============================================================
# LOAD SMALL RESOURCES ONLY
# ============================================================

model, family_mapping = load_model()
metrics = load_metrics()


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [
    "store_nbr",
    "family_code",
    "onpromotion",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "year",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
]


families = list(family_mapping.keys())

# Store Sales dataset contains 54 stores.
stores = list(range(1, 55))

# The feature dataset ends on 2017-08-15.
# Forecasting begins from the next day.
min_forecast_date = pd.Timestamp("2017-08-16")


# ============================================================
# HEADER
# ============================================================

st.title("🛒 :rainbow[AI-Based Demand Forecasting System]")

st.markdown(
    """
    ### Retail Sales Demand Prediction

    This application uses a trained **XGBoost machine learning model**
    to forecast retail demand using historical sales, promotions,
    store information, product family and calendar features.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ :blue[Forecast Settings]")

    store_nbr = st.selectbox(
        "🏪 Select Store",
        stores,
    )

    family = st.selectbox(
        "📦 Select Product Family",
        families,
    )

    forecast_date = st.date_input(
        "📅 Forecast Start Date",
        value=min_forecast_date.date(),
        min_value=min_forecast_date.date(),
    )

    horizon = st.selectbox(
        "🔮 Forecast Horizon",
        [7, 14, 30],
        index=0,
    )

    onpromotion = st.number_input(
        "📢 Items on Promotion",
        min_value=0,
        value=10,
        step=1,
    )

    st.divider()

    st.info(
        """
        **Historical data**

        Available through:

        **2017-08-15**

        Historical lag and rolling features are calculated
        automatically when the forecast is generated.
        """
    )


# ============================================================
# SELECTED DATE
# ============================================================

forecast_date = pd.Timestamp(forecast_date)


# ============================================================
# MAIN DASHBOARD
# ============================================================

st.subheader("📋 :green[Forecast Information]")

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.metric(
        "Store",
        store_nbr,
    )

with info2:
    st.metric(
        "Product Family",
        family,
    )

with info3:
    st.metric(
        "Forecast Start",
        forecast_date.strftime("%Y-%m-%d"),
    )

with info4:
    st.metric(
        "Horizon",
        f"{horizon} Days",
    )

st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🚀 Generate Demand Forecast",
    type="primary",
    use_container_width=True,
)


if predict_button:

    try:

        with st.spinner(
            "Generating forecast from historical sales..."
        ):

            forecast = forecast_demand(
                store_nbr=store_nbr,
                family=family,
                start_date=forecast_date,
                horizon=horizon,
                onpromotion=onpromotion,
            )

        st.success(
            "Demand forecast generated successfully!"
        )


        # ====================================================
        # FORECAST SUMMARY
        # ====================================================

        st.subheader("🎯 Forecast Summary")

        total_demand = forecast[
            "predicted_demand"
        ].sum()

        average_demand = forecast[
            "predicted_demand"
        ].mean()

        maximum_demand = forecast[
            "predicted_demand"
        ].max()

        minimum_demand = forecast[
            "predicted_demand"
        ].min()


        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.metric(
                "Total Forecast",
                f"{total_demand:,.2f}",
            )

        with s2:
            st.metric(
                "Average Daily Demand",
                f"{average_demand:,.2f}",
            )

        with s3:
            st.metric(
                "Maximum Demand",
                f"{maximum_demand:,.2f}",
            )

        with s4:
            st.metric(
                "Minimum Demand",
                f"{minimum_demand:,.2f}",
            )


        # ====================================================
        # FORECAST CHART
        # ====================================================

        st.subheader("📈 Future Demand Forecast")

        chart_data = forecast.copy()
        chart_data["date"] = pd.to_datetime(
            chart_data["date"]
        )

        chart_data = chart_data.set_index("date")

        st.line_chart(
            chart_data["predicted_demand"]
        )


        # ====================================================
        # FORECAST TABLE
        # ====================================================

        st.subheader("📅 Forecast Details")

        display_forecast = forecast.copy()

        display_forecast["predicted_demand"] = (
            display_forecast["predicted_demand"].round(2)
        )

        display_forecast.columns = [
            "Date",
            "Predicted Demand",
        ]

        st.dataframe(
            display_forecast,
            use_container_width=True,
            hide_index=True,
        )


        # ====================================================
        # DOWNLOAD FORECAST
        # ====================================================

        csv = forecast.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Forecast CSV",
            data=csv,
            file_name="demand_forecast.csv",
            mime="text/csv",
        )


    except Exception as e:

        st.error(
            f"Forecast generation failed: {e}"
        )

        st.exception(e)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.subheader("📊 :green[Model Performance]")


xgb_rows = metrics[
    metrics["model"] == "XGBoost"
]

if not xgb_rows.empty:

    xgb_metrics = xgb_rows.iloc[0]

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "MAE",
            f"{xgb_metrics['MAE']:.2f}",
        )

    with m2:
        st.metric(
            "RMSE",
            f"{xgb_metrics['RMSE']:.2f}",
        )

    with m3:
        st.metric(
            "MAPE",
            f"{xgb_metrics['MAPE']:.2f}%",
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader("⚖️ :green[Baseline vs XGBoost]")

comparison = metrics.set_index("model")

st.bar_chart(
    comparison[
        ["MAE", "RMSE"]
    ]
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader("⭐ :green[Feature Importance]")


importance_df = pd.DataFrame(
    {
        "Feature": FEATURES,
        "Importance": model.feature_importances_,
    }
)

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False,
)

st.bar_chart(
    importance_df.set_index("Feature")
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("ℹ️ :green[About This System]")


st.markdown(
    """
    **Machine Learning Model:** XGBoost

    **Dataset:** Store Sales - Time Series Forecasting

    **Forecasting Approach:** Recursive multi-step forecasting

    **Forecast Horizons:** 7, 14 and 30 days

    **Forecasting Features:**
    - Historical sales lags
    - Rolling averages
    - Day of week
    - Day of month
    - Month
    - Week of year
    - Year
    - Store
    - Product family
    - Promotion information

    **Evaluation Metrics:**
    - MAE
    - RMSE
    - MAPE
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Based Demand Forecasting System | "
    "XGBoost + Time-Series Feature Engineering"
)
