import pandas as pd

print("=" * 60)
print("DEMAND FORECASTING AI - FEATURE ENGINEERING")
print("=" * 60)

# ---------------------------------------------------------
# 1. Load only required columns
# ---------------------------------------------------------
print("\nLoading dataset...")

columns = [
    "date",
    "store_nbr",
    "family",
    "sales",
    "onpromotion"
]

df = pd.read_csv(
    "data/train.csv",
    usecols=columns,
    low_memory=True
)

print(f"Dataset loaded: {len(df):,} rows")


# ---------------------------------------------------------
# 2. Convert date
# ---------------------------------------------------------
print("\nPreparing dates...")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["store_nbr", "family", "date"]
).reset_index(drop=True)


# ---------------------------------------------------------
# 3. Calendar Features
# ---------------------------------------------------------
print("Creating calendar features...")

df["day_of_week"] = df["date"].dt.dayofweek.astype("int8")
df["day_of_month"] = df["date"].dt.day.astype("int8")
df["month"] = df["date"].dt.month.astype("int8")
df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int8")
df["year"] = df["date"].dt.year.astype("int16")


# ---------------------------------------------------------
# 4. Lag Features
# ---------------------------------------------------------
print("Creating lag features...")

group = df.groupby(
    ["store_nbr", "family"],
    sort=False
)["sales"]

df["lag_1"] = group.shift(1)
df["lag_7"] = group.shift(7)
df["lag_14"] = group.shift(14)


# ---------------------------------------------------------
# 5. Rolling Features
# ---------------------------------------------------------
print("Creating rolling features...")

df["rolling_mean_7"] = (
    group.transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
)

df["rolling_mean_14"] = (
    group.transform(
        lambda x: x.shift(1).rolling(14).mean()
    )
)

df["rolling_mean_30"] = (
    group.transform(
        lambda x: x.shift(1).rolling(30).mean()
    )
)


# ---------------------------------------------------------
# 6. Remove rows without enough history
# ---------------------------------------------------------
print("\nRemoving rows without enough history...")

df = df.dropna().reset_index(drop=True)

print(f"Rows remaining: {len(df):,}")


# ---------------------------------------------------------
# 7. Reduce memory usage
# ---------------------------------------------------------
print("\nOptimizing memory usage...")

df["store_nbr"] = df["store_nbr"].astype("int8")
df["onpromotion"] = df["onpromotion"].astype("int16")

df["sales"] = df["sales"].astype("float32")

df["lag_1"] = df["lag_1"].astype("float32")
df["lag_7"] = df["lag_7"].astype("float32")
df["lag_14"] = df["lag_14"].astype("float32")

df["rolling_mean_7"] = df["rolling_mean_7"].astype("float32")
df["rolling_mean_14"] = df["rolling_mean_14"].astype("float32")
df["rolling_mean_30"] = df["rolling_mean_30"].astype("float32")

df["family"] = df["family"].astype("category")


# ---------------------------------------------------------
# 8. Display sample
# ---------------------------------------------------------
print("\nFeature sample:")

print(
    df[
        [
            "date",
            "store_nbr",
            "family",
            "sales",
            "onpromotion",
            "day_of_week",
            "month",
            "lag_1",
            "lag_7",
            "rolling_mean_7"
        ]
    ].head(10)
)


# ---------------------------------------------------------
# 9. Memory usage
# ---------------------------------------------------------
memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

print(f"\nFeature dataset memory usage: {memory_mb:.2f} MB")


# ---------------------------------------------------------
# 10. Save
# ---------------------------------------------------------
print("\nSaving feature dataset...")

df.to_pickle("data/features.pkl")

print("Saved: data/features.pkl")


print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)