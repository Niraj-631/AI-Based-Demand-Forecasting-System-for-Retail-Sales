import pandas as pd

# Path to our dataset
DATA_PATH = "data/train.csv"

print("=" * 60)
print("DEMAND FORECASTING AI - DATASET EXPLORATION")
print("=" * 60)

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

# Basic information
print("\n1. Dataset Shape")
print("-" * 30)
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

# Column names
print("\n2. Columns")
print("-" * 30)
for column in df.columns:
    print(f"- {column}")

# First 5 rows
print("\n3. First 5 Rows")
print("-" * 30)
print(df.head())

# Data types
print("\n4. Data Types")
print("-" * 30)
print(df.dtypes)

# Missing values
print("\n5. Missing Values")
print("-" * 30)
print(df.isnull().sum())

# Date range
print("\n6. Date Information")
print("-" * 30)

df["date"] = pd.to_datetime(df["date"])

print(f"Start date : {df['date'].min()}")
print(f"End date   : {df['date'].max()}")

# Number of unique stores
if "store_nbr" in df.columns:
    print(f"\nNumber of stores : {df['store_nbr'].nunique()}")

# Number of product families
if "family" in df.columns:
    print(f"Number of product families : {df['family'].nunique()}")

# Sales statistics
if "sales" in df.columns:
    print("\n7. Sales Statistics")
    print("-" * 30)
    print(df["sales"].describe())

print("\n" + "=" * 60)
print("EXPLORATION COMPLETE")
print("=" * 60)