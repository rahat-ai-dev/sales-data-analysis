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
    "price":     prices_dirty,
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



# ─────────────────────────────────────────────
# Step 3: Analysis and Insights
# ─────────────────────────────────────────────
print("\n" + "-" * 60)
print("  Step 3 — Analysis & Insights")
print("-" * 60)

total_rev=df["net_revenue"].sum()
total_orders=len(df)
avg_order_val=df["net_revenue"].mean()

print(f"\n[Q-1] Total net revenue: ${total_rev:>12,.2f}")
print(f"Total orders:{total_orders:>5}")
print(f"Average Order : ${avg_order_val:>8,.2f} ")


print("\n[Q-2] product wise revenue:")

prod_rev=(
    df.groupby("product",observed=True)["net_revenue"]
    .agg(total="sum", orders="count",avg="mean")
    .sort_values("total",ascending=False)
)

prod_rev_display=prod_rev.copy()
prod_rev_display["total"]=prod_rev_display["total"].map("${:,.2f}".format)
prod_rev_display["avg"]=prod_rev_display["avg"].map("${:,.2f}".format)
print(prod_rev_display.to_string())

print("\n[Q3] Region-wise Performance:")

region_stats=(
    df.groupby("region",observed=True)
    .agg(
      revenue=("net_revenue","sum"),
      orders=("net_revenue","count"),
      avg_sale=("net_revenue","mean"),
      avg_disc=("discount","mean"),
    )
    .sort_values("revenue",ascending=False)
)
region_display=region_stats.copy()
region_display["revenue"]=region_display["revenue"].map("${:,.2f}".format)
region_display["avg_sale"]=region_display["avg_sale"].map("${:,.2f}".format)
region_display["avg_disc"]=region_display["avg_disc"].map("${:,.1f}".format)
print(region_display.to_string())

print("\n[Q4] Monthly Revenue Trend:")

monthly=(
    df.groupby("month",observed=True)["net_revenue"]
    .sum().reset_index()
)
monthly.columns=["month","revenue"]
month_labels=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
monthly["month_name"]=monthly["month"].apply(lambda x:month_labels[x-1])

max_rev=monthly["revenue"].max()
for _, row in monthly.iterrows():
    bar_len=int(row["revenue"]/max_rev*30)
    bar="*"*bar_len
    print(f"{row["month_name"]:>3} {bar:<30} ${row["revenue"]:>9,.2f}")

print("\n[Q5] Discount vs Avg Net Revenue:")
disc_impact=(
    df.groupby("discount")["net_revenue"]
    .agg(avg="mean", count="count").reset_index()
)
for _, row in disc_impact.iterrows():
    print(f"Discount{int(row["discount"]):>2}% -> avg${row["avg"]:>9,.2f} ({int(row["count"])} orders)")


print("\n[Q6] Category-wise Share:")
cat_rev=df.groupby("category",observed=True)["net_revenue"].sum()
total=cat_rev.sum()

for cat,rev in cat_rev.sort_values(ascending=False).items():
    pct=rev/total*100
    print(f"{cat:<15} ${rev:>12,.2f} ({pct:.1f}%)")


# ─────────────────────────────────────────────
# Step 4: NumPy Statistics
# ─────────────────────────────────────────────
print("\n" + "-" * 60)
print("  Step 4 — NumPy Statistics")
print("-" * 60)

rev_arr=df["net_revenue"].values

stats=[
    ("Mean", np.mean(rev_arr)),
    ("Median", np.median(rev_arr)),
    ("Std Dev", np.std(rev_arr)),
    ("Min", np.min(rev_arr)),
    ("Max", np.max(rev_arr)),
    ("25th pct", np.percentile(rev_arr, 25)),
    ("75th pct", np.percentile(rev_arr, 75)),
    ("90th pct", np.percentile(rev_arr, 90)),
]
for label, value in stats:
    print(f" {label:<10}: ${value:>10,.2f}")


threshold = np.percentile(rev_arr,90)
top_orders=df[df["net_revenue"]>=threshold]
print(f"\n Top 10% orders(>=${threshold:,.2f}):")
print(f" Count :{len(top_orders)}")
print(f" Revenue: $ {top_orders["net_revenue"].sum():,.2f}({top_orders["net_revenue"].sum()/total_rev*100:.1f}% of total)")

corr=np.corrcoef(df["quantity"].values, df["net_revenue"].values)[0,1]
print(f"\n Quantity ↔ Revenue correlation:{corr:.3f}")






