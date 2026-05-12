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