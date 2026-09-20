import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

print("=" * 60)
print("DEMAND FORECASTING AI - VISUALIZATION")
print("=" * 60)

# Load test predictions
print("\nLoading test predictions...")
results = pd.read_pickle("data/test_predictions.pkl")

results["date"] = pd.to_datetime(results["date"])

print(f"Rows loaded: {len(results):,}")

# ============================================================
# 1. ACTUAL VS XGBOOST
# ============================================================

print("\nCreating Actual vs XGBoost chart...")

daily = (
    results
    .groupby("date")
    .agg({
        "sales": "sum",
        "xgb_prediction": "sum"
    })
    .reset_index()
)

plt.figure(figsize=(14, 6))

plt.plot(
    daily["date"],
    daily["sales"],
    label="Actual Sales"
)

plt.plot(
    daily["date"],
    daily["xgb_prediction"],
    label="XGBoost Prediction"
)

plt.title("Actual vs XGBoost Predicted Demand")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "data/actual_vs_xgboost.png",
    dpi=150
)

plt.close()

print("Saved: data/actual_vs_xgboost.png")


# ============================================================
# 2. ACTUAL VS BASELINE VS XGBOOST
# ============================================================

print("\nCreating model comparison chart...")

plt.figure(figsize=(14, 6))

plt.plot(
    daily["date"],
    daily["sales"],
    label="Actual"
)

baseline_daily = (
    results
    .groupby("date")["baseline_prediction"]
    .sum()
    .reset_index()
)

plt.plot(
    baseline_daily["date"],
    baseline_daily["baseline_prediction"],
    label="7-Day Baseline"
)

plt.plot(
    daily["date"],
    daily["xgb_prediction"],
    label="XGBoost"
)

plt.title("Demand Forecast Comparison")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "data/model_comparison.png",
    dpi=150
)

plt.close()

print("Saved: data/model_comparison.png")


# ============================================================
# 3. RESIDUAL DISTRIBUTION
# ============================================================

print("\nCreating residual distribution...")

results["error"] = (
    results["sales"]
    - results["xgb_prediction"]
)

plt.figure(figsize=(10, 6))

plt.hist(
    results["error"],
    bins=100
)

plt.title("XGBoost Prediction Error Distribution")
plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "data/error_distribution.png",
    dpi=150
)

plt.close()

print("Saved: data/error_distribution.png")


# ============================================================
# 4. FEATURE IMPORTANCE
# ============================================================

print("\nCreating feature importance chart...")

model = joblib.load(
    "models/demand_model.pkl"
)

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

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    importance["feature"],
    importance["importance"]
)

plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "data/feature_importance.png",
    dpi=150
)

plt.close()

print("Saved: data/feature_importance.png")


# ============================================================
# 5. DAILY ACTUAL VS PREDICTED
# ============================================================

print("\nCreating daily prediction chart...")

plt.figure(figsize=(14, 6))

plt.scatter(
    daily["sales"],
    daily["xgb_prediction"],
    alpha=0.6
)

min_value = min(
    daily["sales"].min(),
    daily["xgb_prediction"].min()
)

max_value = max(
    daily["sales"].max(),
    daily["xgb_prediction"].max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title("Actual vs Predicted Daily Demand")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.tight_layout()

plt.savefig(
    "data/actual_vs_predicted_scatter.png",
    dpi=150
)

plt.close()

print("Saved: data/actual_vs_predicted_scatter.png")


print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)