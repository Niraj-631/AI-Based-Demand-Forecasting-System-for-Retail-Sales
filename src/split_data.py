import pandas as pd

print("=" * 60)
print("DEMAND FORECASTING AI - TIME SERIES DATA SPLIT")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load feature dataset
# ---------------------------------------------------------
print("\nLoading feature dataset...")

df = pd.read_pickle("data/features.pkl")

print(f"Rows loaded: {len(df):,}")


# ---------------------------------------------------------
# 2. Check date range
# ---------------------------------------------------------
print("\nDate range:")

print("Start:", df["date"].min())
print("End  :", df["date"].max())


# ---------------------------------------------------------
# 3. Define time-based splits
# ---------------------------------------------------------
train_end = "2016-12-31"

validation_end = "2017-06-30"

test_start = "2017-07-01"


# ---------------------------------------------------------
# 4. Create datasets
# ---------------------------------------------------------
train = df[df["date"] <= train_end].copy()

validation = df[
    (df["date"] > train_end)
    & (df["date"] <= validation_end)
].copy()

test = df[
    df["date"] >= test_start
].copy()


# ---------------------------------------------------------
# 5. Display sizes
# ---------------------------------------------------------
print("\nDataset sizes:")

print(f"Training   : {len(train):,} rows")
print(f"Validation : {len(validation):,} rows")
print(f"Test       : {len(test):,} rows")


# ---------------------------------------------------------
# 6. Display date ranges
# ---------------------------------------------------------
print("\nDate ranges:")

print(
    f"Training   : "
    f"{train['date'].min().date()} → "
    f"{train['date'].max().date()}"
)

print(
    f"Validation : "
    f"{validation['date'].min().date()} → "
    f"{validation['date'].max().date()}"
)

print(
    f"Test       : "
    f"{test['date'].min().date()} → "
    f"{test['date'].max().date()}"
)


# ---------------------------------------------------------
# 7. Save datasets
# ---------------------------------------------------------
print("\nSaving datasets...")

train.to_pickle("data/train_features.pkl")

validation.to_pickle(
    "data/validation_features.pkl"
)

test.to_pickle(
    "data/test_features.pkl"
)

print("Saved:")
print("  data/train_features.pkl")
print("  data/validation_features.pkl")
print("  data/test_features.pkl")


# ---------------------------------------------------------
# 8. Verify no overlap
# ---------------------------------------------------------
print("\nChecking chronological order...")

print(
    "Last training date:",
    train["date"].max().date()
)

print(
    "First validation date:",
    validation["date"].min().date()
)

print(
    "Last validation date:",
    validation["date"].max().date()
)

print(
    "First test date:",
    test["date"].min().date()
)


print("\n" + "=" * 60)
print("TIME SERIES SPLIT COMPLETE")
print("=" * 60)