from pathlib import Path
import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "demand_model.pkl"
MAPPING_PATH = PROJECT_ROOT / "models" / "family_mapping.pkl"
FEATURES_PATH = PROJECT_ROOT / "data" / "features.pkl"


# ============================================================
# MODEL FEATURES
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


# ============================================================
# LOAD RESOURCES
# ============================================================

def load_model():
    """Load the trained XGBoost model."""
    return joblib.load(MODEL_PATH)


def load_family_mapping():
    """Load the product-family to integer-code mapping."""
    return joblib.load(MAPPING_PATH)


def load_data():
    """
    Load historical feature data only when forecasting is requested.

    Keeping this out of module import prevents pytest from loading
    the large features.pkl file just by importing this module.
    """
    data = pd.read_pickle(FEATURES_PATH)
    data["date"] = pd.to_datetime(data["date"])
    return data


# ============================================================
# FORECAST
# ============================================================

def forecast_demand(
    store_nbr,
    family,
    start_date,
    horizon=7,
    onpromotion=0,
):
    """
    Generate a recursive multi-day demand forecast.

    Parameters
    ----------
    store_nbr : int
        Store number.

    family : str
        Product family.

    start_date : str or datetime-like
        First date to forecast.

    horizon : int
        Number of future days.

    onpromotion : int or float
        Number of items on promotion for each forecast day.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - date
        - predicted_demand
    """

    if horizon <= 0:
        raise ValueError("horizon must be greater than 0")

    model = load_model()
    family_mapping = load_family_mapping()
    data = load_data()

    start_date = pd.Timestamp(start_date)

    # Check product family
    if family not in family_mapping:
        raise ValueError(f"Unknown product family: {family}")

    family_code = family_mapping[family]

    # --------------------------------------------------------
    # Get historical data
    # --------------------------------------------------------

    history = data[
        (data["store_nbr"] == store_nbr)
        & (data["family"] == family)
        & (data["date"] < start_date)
    ].sort_values("date")

    if len(history) < 30:
        raise ValueError(
            "At least 30 historical observations are required "
            "before the forecast start date."
        )

    # Keep historical sales as a list so predictions can be
    # appended recursively for later forecast days.
    sales_history = history["sales"].astype(float).tolist()

    forecasts = []

    # --------------------------------------------------------
    # Recursive forecasting
    # --------------------------------------------------------

    for step in range(horizon):

        current_date = start_date + pd.Timedelta(days=step)

        # Lag features
        lag_1 = sales_history[-1]
        lag_7 = sales_history[-7]
        lag_14 = sales_history[-14]

        # Rolling features
        rolling_mean_7 = sum(sales_history[-7:]) / 7
        rolling_mean_14 = sum(sales_history[-14:]) / 14
        rolling_mean_30 = sum(sales_history[-30:]) / 30

        # Calendar features
        day_of_week = current_date.dayofweek
        day_of_month = current_date.day
        month = current_date.month
        week_of_year = int(current_date.isocalendar().week)
        year = current_date.year

        # ----------------------------------------------------
        # Model input
        # ----------------------------------------------------

        X = pd.DataFrame(
            [
                {
                    "store_nbr": store_nbr,
                    "family_code": family_code,
                    "onpromotion": onpromotion,
                    "day_of_week": day_of_week,
                    "day_of_month": day_of_month,
                    "month": month,
                    "week_of_year": week_of_year,
                    "year": year,
                    "lag_1": lag_1,
                    "lag_7": lag_7,
                    "lag_14": lag_14,
                    "rolling_mean_7": rolling_mean_7,
                    "rolling_mean_14": rolling_mean_14,
                    "rolling_mean_30": rolling_mean_30,
                }
            ]
        )

        X = X[FEATURES]

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = float(model.predict(X)[0])

        # Demand cannot be negative.
        prediction = max(prediction, 0.0)

        # Add prediction to history so the next forecast day
        # can use this predicted value as a lag/rolling input.
        sales_history.append(prediction)

        forecasts.append(
            {
                "date": current_date,
                "predicted_demand": prediction,
            }
        )

    return pd.DataFrame(forecasts)


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MULTI-DAY DEMAND FORECAST TEST")
    print("=" * 60)

    forecast = forecast_demand(
        store_nbr=1,
        family="GROCERY I",
        start_date="2017-08-16",
        horizon=7,
        onpromotion=10,
    )

    print("\nForecast:")
    print("-" * 40)
    print(forecast.to_string(index=False))

    print("\n" + "=" * 60)
    print("FORECAST TEST COMPLETE")
    print("=" * 60)