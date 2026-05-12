# ============================================================
#   SALES DATA ANALYSIS — Mini Project
#   Fix: UTF-8 encoding for Windows terminals
# ============================================================

import sys
import io
import os



# Fix Windows cp1252 encoding issue — must be at the very top
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────
# Step 1: Generate Dummy Sales Data (with NumPy)
# ─────────────────────────────────────────────
print("=" * 60)
print("  SALES DATA ANALYSIS — Mini Project")
print("=" * 60)

n = 500  # 500 transactions

products   = ["Laptop", "Phone", "Tablet", "Headphone", "Charger"]
categories = ["Electronics", "Electronics", "Electronics", "Accessories", "Accessories"]
regions    = ["Dhaka", "Chittagong", "Sylhet", "Rajshahi", "Khulna"]
months     = pd.date_range("2024-01-01", periods=12, freq="MS")

product_idx = np.random.randint(0, 5, n)
region_idx  = np.random.randint(0, 5, n)
month_idx   = np.random.randint(0, 12, n)

base_prices = {"Laptop": 650, "Phone": 300, "Tablet": 250,
               "Headphone": 35, "Charger": 8}
price_noise = {"Laptop": 150, "Phone": 80, "Tablet": 60,
               "Headphone": 10, "Charger": 2}

product_names = np.array(products)[product_idx]
prices = np.array([
    round(base_prices[p] + np.random.uniform(-price_noise[p], price_noise[p]), 2)
    for p in product_names
])

quantity   = np.random.randint(1, 6, n)
discount   = np.random.choice([0, 5, 10, 15, 20], n, p=[0.4, 0.2, 0.2, 0.1, 0.1])
sale_dates = months[month_idx]

# Intentionally add dirty data for cleaning practice
prices_dirty = prices.copy().astype(float)
prices_dirty[np.random.choice(n, 15, replace=False)] = np.nan
prices_dirty[np.random.choice(n, 5,  replace=False)] = -999

df = pd.DataFrame({
    "date":     sale_dates,
    "product":  product_names,
    "category": np.array(categories)[product_idx],
    "region":   np.array(regions)[region_idx],
    "price":    prices_dirty,
    "quantity": quantity,
    "discount": discount,
})

print(f"\n[1] Raw Data Generated — {len(df)} rows, {df.shape[1]} columns")
print(df.head(6).to_string(index=False))


# ─────────────────────────────────────────────
# Step 2: Data Cleaning
# ─────────────────────────────────────────────
print("\n" + "-" * 60)
print("  Step 2 — Data Cleaning")
print("-" * 60)

print(f"\nNaN count (price): {df['price'].isna().sum()}")
print(f"Invalid prices (<0): {(df['price'] < 0).sum()}")

df['price'] = df['price'].where(df['price'] > 0, np.nan)
df['price'] = df.groupby('product')['price'].transform(
    lambda x: x.fillna(x.median())
)

df['product']  = df['product'].astype('category')
df['category'] = df['category'].astype('category')
df['region']   = df['region'].astype('category')
df['discount'] = df['discount'].astype('int8')
df['quantity'] = df['quantity'].astype('int8')
df['price']    = df['price'].astype('float32')

print(f"NaN count after cleaning: {df.isna().sum().sum()}")
print(f"Memory (after optimize): {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

df['revenue']      = (df['price'] * df['quantity']).astype('float32')
df['discount_amt'] = (df['revenue'] * df['discount'] / 100).astype('float32')
df['net_revenue']  = (df['revenue'] - df['discount_amt']).astype('float32')
df['month']        = df['date'].dt.month
df['month_name']   = df['date'].dt.strftime('%b')

print("\n[2] Cleaned DataFrame (first 5 rows):")
print(df[['product','region','price','quantity','discount','net_revenue']].head(5).to_string(index=False))
