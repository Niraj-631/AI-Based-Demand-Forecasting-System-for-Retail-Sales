import pandas as pd
import numpy as np
import joblib


MODEL_PATH = "models/demand_model.pkl"
MAPPING_PATH = "models/family_mapping.pkl"


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
    "rolling_mean_30"
]


def load_model():
    """
    Load trained XGBoost model and family mapping.
    """

    model = joblib.load(MODEL_PATH)
    family_mapping = joblib.load(MAPPING_PATH)

    return model, family_mapping


def prepare_features(
    store_nbr,
    family,
    date,
    onpromotion,
    lag_1,
    lag_7,
    lag_14,
    rolling_mean_7,
    rolling_mean_14,
    rolling_mean_30,
    family_mapping
):
    """
    Create model features from user inputs.
    """

    date = pd.to_datetime(date)

    if family not in family_mapping:
        raise ValueError(
            f"Unknown product family: {family}"
        )

    family_code = family_mapping[family]

    week_of_year = int(
        date.isocalendar().week
    )

    features = pd.DataFrame([{
        "store_nbr": store_nbr,
        "family_code": family_code,
        "onpromotion": onpromotion,
        "day_of_week": date.dayofweek,
        "day_of_month": date.day,
        "month": date.month,
        "week_of_year": week_of_year,
        "year": date.year,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "rolling_mean_7": rolling_mean_7,
        "rolling_mean_14": rolling_mean_14,
        "rolling_mean_30": rolling_mean_30
    }])

    return features[FEATURES]


def predict_demand(
    store_nbr,
    family,
    date,
    onpromotion,
    lag_1,
    lag_7,
    lag_14,
    rolling_mean_7,
    rolling_mean_14,
    rolling_mean_30
):
    """
    Predict demand for a given store, family and date.
    """

    model, family_mapping = load_model()

    X = prepare_features(
        store_nbr=store_nbr,
        family=family,
        date=date,
        onpromotion=onpromotion,
        lag_1=lag_1,
        lag_7=lag_7,
        lag_14=lag_14,
        rolling_mean_7=rolling_mean_7,
        rolling_mean_14=rolling_mean_14,
        rolling_mean_30=rolling_mean_30,
        family_mapping=family_mapping
    )

    prediction = model.predict(X)[0]

    prediction = max(float(prediction), 0)

    return prediction


if __name__ == "__main__":

    print("=" * 60)
    print("DEMAND FORECASTING AI - PREDICTION TEST")
    print("=" * 60)

    prediction = predict_demand(
        store_nbr=1,
        family="GROCERY I",
        date="2017-08-16",
        onpromotion=10,
        lag_1=150,
        lag_7=170,
        lag_14=165,
        rolling_mean_7=160,
        rolling_mean_14=158,
        rolling_mean_30=155
    )

    print("\nExample prediction")
    print("-" * 30)
    print(f"Store        : 1")
    print(f"Product      : GROCERY I")
    print(f"Date         : 2017-08-16")
    print(f"Promotion    : 10")
    print(f"Predicted demand: {prediction:.2f}")

    print("\n" + "=" * 60)
    print("PREDICTION TEST COMPLETE")
    print("=" * 60)