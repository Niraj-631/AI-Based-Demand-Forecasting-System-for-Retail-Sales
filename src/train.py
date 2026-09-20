import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

print("=" * 60)
print("DEMAND FORECASTING AI - XGBOOST TRAINING")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load training data
# ---------------------------------------------------------
print("\nLoading training data...")

train = pd.read_pickle(
    "data/train_features.pkl"
)

print(f"Training rows: {len(train):,}")


# ---------------------------------------------------------
# 2. Load validation data
# ---------------------------------------------------------
print("\nLoading validation data...")

validation = pd.read_pickle(
    "data/validation_features.pkl"
)

print(
    f"Validation rows: "
    f"{len(validation):,}"
)


# ---------------------------------------------------------
# 3. Encode product family
# ---------------------------------------------------------
print("\nEncoding product family...")

# Convert category/string to numerical codes
all_families = pd.concat(
    [
        train["family"],
        validation["family"]
    ]
).astype("category").cat.categories

family_mapping = {
    family: index
    for index, family in enumerate(all_families)
}

train["family_code"] = (
    train["family"]
    .map(family_mapping)
    .astype("int16")
)

validation["family_code"] = (
    validation["family"]
    .map(family_mapping)
    .astype("int16")
)


# ---------------------------------------------------------
# 4. Select features
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

target = "sales"


# ---------------------------------------------------------
# 5. Create X and y
# ---------------------------------------------------------
print("\nPreparing training data...")

X_train = train[features]
y_train = train[target]

X_validation = validation[features]
y_validation = validation[target]

print(
    f"Training features: "
    f"{X_train.shape}"
)

print(
    f"Validation features: "
    f"{X_validation.shape}"
)


# ---------------------------------------------------------
# 6. Create XGBoost model
# ---------------------------------------------------------
print("\nCreating XGBoost model...")

model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    objective="reg:squarederror",

    eval_metric="mae",

    tree_method="hist",

    n_jobs=2,

    random_state=42
)


# ---------------------------------------------------------
# 7. Train model
# ---------------------------------------------------------
print("\nStarting training...")
print("This may take several minutes.\n")

model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_train, y_train),
        (X_validation, y_validation)
    ],

    verbose=50
)


# ---------------------------------------------------------
# 8. Validation predictions
# ---------------------------------------------------------
print("\nCreating validation predictions...")

validation_prediction = model.predict(
    X_validation
)

# Sales cannot be negative
validation_prediction = np.maximum(
    validation_prediction,
    0
)


# ---------------------------------------------------------
# 9. Calculate metrics
# ---------------------------------------------------------
mae = mean_absolute_error(
    y_validation,
    validation_prediction
)

rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        validation_prediction
    )
)


# MAPE excluding zero actual values
non_zero = y_validation != 0

mape = np.mean(
    np.abs(
        (
            y_validation[non_zero]
            - validation_prediction[non_zero]
        )
        / y_validation[non_zero]
    )
) * 100


# ---------------------------------------------------------
# 10. Display results
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("XGBOOST VALIDATION RESULTS")
print("=" * 60)

print(f"\nMAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")


# ---------------------------------------------------------
# 11. Feature importance
# ---------------------------------------------------------
print("\nTop feature importance:")

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(importance)


# ---------------------------------------------------------
# 12. Save model
# ---------------------------------------------------------
print("\nSaving model...")

joblib.dump(
    model,
    "models/demand_model.pkl"
)

# Save mapping too
joblib.dump(
    family_mapping,
    "models/family_mapping.pkl"
)

print("Saved:")
print("  models/demand_model.pkl")
print("  models/family_mapping.pkl")


# ---------------------------------------------------------
# 13. Save validation results
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
    "data/xgboost_validation_results.csv",
    index=False
)

print(
    "\nValidation results saved to:"
)

print(
    "data/xgboost_validation_results.csv"
)


print("\n" + "=" * 60)
print("XGBOOST TRAINING COMPLETE")
print("=" * 60)