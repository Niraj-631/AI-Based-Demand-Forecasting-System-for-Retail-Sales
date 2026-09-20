import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# DEMAND FORECASTING AI
# Exploratory Data Analysis
# ============================================================

DATA_PATH = "data/train.csv"

print("=" * 60)
print("DEMAND FORECASTING AI - EDA")
print("=" * 60)

# Load data
print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"])

print(f"Dataset loaded: {len(df):,} rows")

# ============================================================
# 1. DAILY TOTAL SALES
# ============================================================

daily_sales = (
    df.groupby("date")["sales"]
    .sum()
    .reset_index()
)

print("\nDaily sales summary:")
print(daily_sales.head())

# Plot daily sales
plt.figure(figsize=(14, 6))

plt.plot(
    daily_sales["date"],
    daily_sales["sales"]
)

plt.title("Total Daily Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("daily_sales.png", dpi=150)

plt.show()

# ============================================================
# 2. MONTHLY SALES
# ============================================================

df["year_month"] = df["date"].dt.to_period("M")

monthly_sales = (
    df.groupby("year_month")["sales"]
    .sum()
    .reset_index()
)

monthly_sales["year_month"] = (
    monthly_sales["year_month"].astype(str)
)

plt.figure(figsize=(14, 6))

plt.plot(
    monthly_sales["year_month"],
    monthly_sales["sales"]
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")

# Show fewer labels so the graph remains readable
plt.xticks(
    range(0, len(monthly_sales), 6),
    monthly_sales["year_month"].iloc[::6],
    rotation=45
)

plt.tight_layout()

plt.savefig("monthly_sales.png", dpi=150)

plt.show()

# ============================================================
# 3. SALES BY PRODUCT FAMILY
# ============================================================

family_sales = (
    df.groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 product families by total sales:")
print(family_sales.head(10))

plt.figure(figsize=(12, 7))

family_sales.head(10).sort_values().plot(kind="barh")

plt.title("Top 10 Product Families by Total Sales")
plt.xlabel("Total Sales")
plt.ylabel("Product Family")

plt.tight_layout()

plt.savefig("top_product_families.png", dpi=150)

plt.show()

# ============================================================
# 4. SALES BY STORE
# ============================================================

store_sales = (
    df.groupby("store_nbr")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 stores by total sales:")
print(store_sales.head(10))

plt.figure(figsize=(12, 6))

store_sales.head(10).sort_values().plot(kind="barh")

plt.title("Top 10 Stores by Total Sales")
plt.xlabel("Total Sales")
plt.ylabel("Store Number")

plt.tight_layout()

plt.savefig("top_stores.png", dpi=150)

plt.show()

# ============================================================
# 5. SALES BY DAY OF WEEK
# ============================================================

df["day_of_week"] = df["date"].dt.day_name()

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekly_sales = (
    df.groupby("day_of_week")["sales"]
    .sum()
    .reindex(day_order)
)

print("\nSales by day of week:")
print(weekly_sales)

plt.figure(figsize=(10, 6))

weekly_sales.plot(kind="bar")

plt.title("Sales by Day of Week")
plt.xlabel("Day")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("sales_by_day.png", dpi=150)

plt.show()

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)