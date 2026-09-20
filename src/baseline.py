import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


print("=" * 60)
print("DEMAND FORECASTING AI - BASELINE MODEL")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load test dataset
# ---------------------------------------------------------
print("\nLoading test data...")

test = pd.read_pickle("data/test_features.pkl")

print(f"Test rows: {len(test):,}")


# ---------------------------------------------------------
# 2. Create baseline predictions
# ---------------------------------------------------------
print("\nCreating 7-day lag predictions...")

test["prediction"] = test["lag_7"]


# ---------------------------------------------------------
# 3. Actual and predicted values
# ---------------------------------------------------------
actual = test["sales"]
predicted = test["prediction"]


# ---------------------------------------------------------
# 4. MAE
# ---------------------------------------------------------
mae = mean_absolute_error(
    actual,
    predicted
)


# ---------------------------------------------------------
# 5. RMSE
# ---------------------------------------------------------
rmse = np.sqrt(
    mean_squared_error(
        actual,
        predicted
    )
)


# ---------------------------------------------------------
# 6. MAPE
# ---------------------------------------------------------
# Avoid division by zero
non_zero = actual != 0

mape = np.mean(
    np.abs(
        (
            actual[non_zero]
            - predicted[non_zero]
        )
        / actual[non_zero]
    )
) * 100


# ---------------------------------------------------------
# 7. Results
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("BASELINE RESULTS")
print("=" * 60)

print(f"\nMAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")


# ---------------------------------------------------------
# 8. Sample predictions
# ---------------------------------------------------------
print("\nSample predictions:")

print(
    test[
        [
            "date",
            "store_nbr",
            "family",
            "sales",
            "lag_7",
            "prediction"
        ]
    ].head(20)
)


# ---------------------------------------------------------
# 9. Save results
# ---------------------------------------------------------
results = pd.DataFrame({
    "metric": [
        "MAE",
        "RMSE",
        "MAPE"
    ],
    "value": [
        mae,
        rmse,
        mape
    ]
})

results.to_csv(
    "data/baseline_results.csv",
    index=False
)

print("\nResults saved to:")
print("data/baseline_results.csv")


print("\n" + "=" * 60)
print("BASELINE MODEL COMPLETE")
print("=" * 60)