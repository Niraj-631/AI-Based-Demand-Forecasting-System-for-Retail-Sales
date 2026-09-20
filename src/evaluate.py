import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


print("=" * 60)
print("DEMAND FORECASTING AI - FINAL MODEL EVALUATION")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load test data
# ---------------------------------------------------------
print("\nLoading test data...")

test = pd.read_pickle(
    "data/test_features.pkl"
)

print(
    f"Test rows: {len(test):,}"
)


# ---------------------------------------------------------
# 2. Load trained model
# ---------------------------------------------------------
print("\nLoading trained XGBoost model...")

model = joblib.load(
    "models/demand_model.pkl"
)

family_mapping = joblib.load(
    "models/family_mapping.pkl"
)


# ---------------------------------------------------------
# 3. Encode family
# ---------------------------------------------------------
test["family_code"] = (
    test["family"]
    .map(family_mapping)
    .astype("int16")
)


# ---------------------------------------------------------
# 4. Define features
# ---------------------------------------------------------
features = [
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


# ---------------------------------------------------------
# 5. Prepare test data
# ---------------------------------------------------------
X_test = test[features]

y_test = test["sales"]


# ---------------------------------------------------------
# 6. XGBoost predictions
# ---------------------------------------------------------
print("\nGenerating XGBoost predictions...")

xgb_prediction = model.predict(
    X_test
)

# Demand cannot be negative
xgb_prediction = np.maximum(
    xgb_prediction,
    0
)


# ---------------------------------------------------------
# 7. Baseline predictions
# ---------------------------------------------------------
baseline_prediction = test["lag_7"].values


# ---------------------------------------------------------
# 8. Evaluation function
# ---------------------------------------------------------
def calculate_metrics(actual, prediction):

    mae = mean_absolute_error(
        actual,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            prediction
        )
    )

    non_zero = actual != 0

    mape = np.mean(
        np.abs(
            (
                actual[non_zero]
                - prediction[non_zero]
            )
            / actual[non_zero]
        )
    ) * 100

    return mae, rmse, mape


# ---------------------------------------------------------
# 9. Calculate metrics
# ---------------------------------------------------------
print("\nCalculating metrics...")


baseline_mae, baseline_rmse, baseline_mape = (
    calculate_metrics(
        y_test,
        baseline_prediction
    )
)


xgb_mae, xgb_rmse, xgb_mape = (
    calculate_metrics(
        y_test,
        xgb_prediction
    )
)


# ---------------------------------------------------------
# 10. Improvement
# ---------------------------------------------------------
mae_improvement = (
    (baseline_mae - xgb_mae)
    / baseline_mae
) * 100


rmse_improvement = (
    (baseline_rmse - xgb_rmse)
    / baseline_rmse
) * 100


mape_improvement = (
    (baseline_mape - xgb_mape)
    / baseline_mape
) * 100


# ---------------------------------------------------------
# 11. Display final results
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)


print("\n7-DAY BASELINE")
print("-" * 30)

print(f"MAE  : {baseline_mae:.2f}")
print(f"RMSE : {baseline_rmse:.2f}")
print(f"MAPE : {baseline_mape:.2f}%")


print("\nXGBOOST")
print("-" * 30)

print(f"MAE  : {xgb_mae:.2f}")
print(f"RMSE : {xgb_rmse:.2f}")
print(f"MAPE : {xgb_mape:.2f}%")


print("\nXGBOOST CHANGE VS BASELINE")
print("-" * 30)

print(
    f"MAE improvement  : "
    f"{mae_improvement:.2f}%"
)

print(
    f"RMSE improvement : "
    f"{rmse_improvement:.2f}%"
)

print(
    f"MAPE improvement : "
    f"{mape_improvement:.2f}%"
)


# ---------------------------------------------------------
# 12. Save predictions
# ---------------------------------------------------------
test_results = test[
    [
        "date",
        "store_nbr",
        "family",
        "sales"
    ]
].copy()

test_results["baseline_prediction"] = (
    baseline_prediction
)

test_results["xgb_prediction"] = (
    xgb_prediction
)

test_results.to_pickle(
    "data/test_predictions.pkl"
)


# ---------------------------------------------------------
# 13. Save metrics
# ---------------------------------------------------------
metrics = pd.DataFrame({
    "model": [
        "7-Day Baseline",
        "XGBoost"
    ],

    "MAE": [
        baseline_mae,
        xgb_mae
    ],

    "RMSE": [
        baseline_rmse,
        xgb_rmse
    ],

    "MAPE": [
        baseline_mape,
        xgb_mape
    ]
})


metrics.to_csv(
    "data/final_model_comparison.csv",
    index=False
)


print("\nSaved:")
print("  data/test_predictions.pkl")
print("  data/final_model_comparison.csv")


print("\n" + "=" * 60)
print("FINAL EVALUATION COMPLETE")
print("=" * 60)