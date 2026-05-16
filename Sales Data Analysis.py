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


# ─────────────────────────────────────────────
# Step 5: Visualization Dashboard
# ─────────────────────────────────────────────
print("\n" + "-" * 60)
print("  Step 5 — Building Visualization Dashboard")
print("-" * 60)

# ── Palette & Style ──────────────────────────
DARK_BG="#0D1117"
CARD_BG="#161B22"
ACCENT1="#58A6FF"
ACCENT2="#3FB950"
ACCENT3="#F78166"
ACCENT4="#D2A8FF"
ACCENT5="#FFA657"
GRID_COLOR="#21262D"
TEXT_MAIN="#E6EDF3"
TEXT_SUB="#8B949E"

PALETTE=[ACCENT1,ACCENT2,ACCENT3,ACCENT4,ACCENT5]

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_SUB,
    "axes.titlecolor": TEXT_MAIN,
    "xtick.color": TEXT_SUB,
    "ytick.color": TEXT_SUB,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
    "grid.alpha":1.0,
    "text.color": TEXT_MAIN,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})

# ── Figure Layout ────────────────────────────
fig=plt.figure(figsize=(22,26),facecolor=DARK_BG)
fig.subplots_adjust(hspace=0.45,wspace=0.35,
                    top=0.94,bottom=0.04,
                     left=0.05,right=0.97)

# Title banner
fig.text(0.5,0.965,"SALES PERFORMANCE DASHBOARD  -  2024",
         ha="center",va="center",
         fontsize=22,fontweight="bold",color=TEXT_MAIN,
         fontfamily="DejaVu Sans")
fig.text(0.5,0.952,f"500 Transections - 5 products - 5 Region - Total Revenue:${total_rev:,.2f}",
         ha="center",va="center",
         fontsize=11,color=TEXT_SUB)

# Decorative line under title
line=plt.Line2D([0.04,0.96],[0.945,0.945],
                transform=fig.transFigure,
                color=ACCENT1,linewidth=1.5,alpha=0.5)
fig.add_artist(line)

# ── KPI Cards ───────────────────────────────
best_product= df.groupby("product",observed=True)["net_revenue"].sum().idxmax()
best_region=df.groupby("region",observed=True)["net_revenue"].sum().idxmax()
best_month_n=monthly.loc[monthly["revenue"].idxmax(),"month_name"]
electronics_share=cat_rev.get("Electronics",0)/total*100
top_contrib=top_orders["net_revenue"].sum()/total_rev*100

kpis=[
    ("Total Revenue", f"${total_rev:,.2f}", ACCENT1),
    ("Total Orders", f"${total_orders:,}", ACCENT2),
    ("Average Order Value", f"${avg_order_val:,.2f}", ACCENT5),
    ("Best product", best_product, ACCENT3),
    ("Best Region", best_region, ACCENT4),
    ("Best Month", best_month_n, ACCENT2),
]

for i in range (6):
    left= 0.05+ i*(0.92/6)
    ax_kpi=fig.add_axes([left,0.895,0.138,0.048])
    ax_kpi.set_facecolor(CARD_BG)
    for spine in ax_kpi.spines.values():
        spine.set_visible(True)
        spine.set_color(kpis[i][2])
        spine.set_linewidth(1.5)
    ax_kpi.set_xticks([]); ax_kpi.set_yticks([])
    ax_kpi.text(0.5,0.78,kpis[i][0],ha="center",va="center",
                fontsize=8,color=TEXT_SUB,transform=ax_kpi.transAxes)
    ax_kpi.text(0.5,0.30,kpis[i][1],ha="center",va="center",
                fontsize=11,fontweight="bold",color=kpis[i][2],
                transform=ax_kpi.transAxes)

# ── Grid of 6 charts ─────────────────────────
gs = gridspec.GridSpec(3,3,
                       figure=fig,
                       top=0.875,bottom=0.04,
                       left=0.05,right=0.97,
                       hspace=0.48,wspace=0.32)

# ────── Chart 1: Monthly Revenue Bar + Line ──────
ax1=fig.add_subplot(gs[0,:2])
month_colors=[ACCENT1 if r < monthly["revenue"].max() else ACCENT2
              for r in monthly["revenue"]]
bars=ax1.bar(monthly["month_name"],monthly["revenue"],
             color=month_colors,width=0.6,
             zorder=3,edgecolor="none")
ax2_twin=ax1.twinx()
ax2_twin.set_facecolor(CARD_BG)
ax2_twin.plot(monthly["month_name"],monthly["revenue"],
              color=ACCENT2,linewidth=2.5,marker="o",
              markersize=6,markerfacecolor=DARK_BG,
              markeredgecolor=ACCENT2,markeredgewidth=2,zorder=4)
ax2_twin.tick_params(colors=TEXT_SUB)
ax2_twin.spines["right"].set_color(GRID_COLOR)
ax2_twin.yaxis.label.set_color(TEXT_SUB)

ax1.set_title("Monthly Revenue Trend", fontsize=13,fontweight="bold",
              color=TEXT_MAIN,pad=10)
ax1.set_ylabel("Revenue($)",color=TEXT_SUB,fontsize=9)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x:,.0f}"))
ax1.set_xticklabels(monthly["month_name"],fontsize=9)
ax1.yaxis.grid(True,zorder=0)
ax1.set_axisbelow(True)
max_idx=monthly["revenue"].idxmax()
bars[max_idx].set_color(ACCENT2)
bars[max_idx].set_edgecolor(ACCENT2)
ax1.text(max_idx,monthly["revenue"].max()*1.02,
         f"Peak\n${monthly['revenue'].max():,.0f}",
         ha="center",fontsize=7.5,color=ACCENT2,fontweight="bold")

# ── Chart 2: Product Revenue Horizontal Bar ──
ax2=fig.add_subplot(gs[0,2])
prod_data=df.groupby("product",observed=True)["net_revenue"].sum().sort_values()
colors_prod=[PALETTE[i%len(PALETTE)]for i in range(len(prod_data))]
h_bars=ax2.barh(prod_data.index,prod_data.values,
                color=colors_prod, height=0.55,edgecolor="none")
ax2.set_title("Revenue by product",fontsize=13,fontweight="bold",
              color=TEXT_MAIN,pad=10)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x:,.0f}"))
ax2.xaxis.grid(True,zorder=0)
ax2.set_axisbelow(True)
for bar, val in zip(h_bars, prod_data.values):
    ax2.text(val+prod_data.max()*0.02,bar.get_y()+ bar.get_height()/2,
             f"${val:,.0f}",va="center",fontsize=8,color=TEXT_SUB)

# ──Chart 3: Region Donut──
ax3=fig.add_subplot(gs[1,0])
region_rev=df.groupby("region",observed=True)["net_revenue"].sum().sort_values(ascending=False)
wedges,texts,autotexts=ax3.pie(
    region_rev.values,
    labels=None,
    autopct="%1.1f%%",
    colors=PALETTE,
    startangle=90,
    wedgeprops=dict(width=0.55,edgecolor=DARK_BG,linewidth=2),
    pctdistance=0.75,
)
for at in autotexts:
    at.set_fontsize(8.5)
    at.set_color(TEXT_MAIN)
    at.set_fontweight("bold")
ax3.set_title("Region Revenue Share", fontsize=13,fontweight="bold",
              color=TEXT_MAIN,pad=10)
ax3.legend(region_rev.index,loc="lower center",
           bbox_to_anchor=(0.5,-0.18),ncol=3,
           fontsize=8,frameon=False,
           labelcolor=TEXT_SUB)
ax3.text(0,0,f"${region_rev.sum():,.0f}\nTotal",
         ha="center",va="center",
         fontsize=9,fontweight="bold",color=TEXT_MAIN)

# ──chart 4: Discount Impact(Grouped bar) ──
ax4=fig.add_subplot(gs[1,1])
disc_product=(
     df.groupby(["discount","category"],observed=True)["net_revenue"]
    .mean().reset_index()
)
cats_unique=disc_product["category"].unique()
x_pos=np.arange(len(disc_impact))
width=0.38
for i, cat in enumerate(cats_unique):
    cat_data=disc_product[disc_product["category"]==cat]
    offset=(i-0.5)*width
    ax4.bar(x_pos+offset,
            cat_data.set_index("discount").reindex(disc_impact["discount"])["net_revenue"].values,
            width=width,label=cat,
            color=PALETTE[i],alpha=0.9,edgecolor="none")
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f'{int(d)}%' for d in disc_impact["discount"]],fontsize=9)
ax4.set_title("Discount vs Avg Revenue by Category",fontsize=12,
              fontweight="bold",color=TEXT_MAIN,pad=10)
ax4.set_xlabel("Discount Rate",color=TEXT_SUB)
ax4.set_ylabel("Avg Revenue($)",color=TEXT_SUB)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x:,.0f}"))
ax4.yaxis.grid(True,zorder=0)
ax4.set_axisbelow(True)
ax4.legend(fontsize=9,frameon=False,labelcolor=TEXT_SUB )

# ── Chart 5: Seaborn Heatmap – Region x Month ──
ax5 = fig.add_subplot(gs[1, 2])
heatmap_data = (
    df.groupby(['region', 'month_name'], observed=True)['net_revenue']
    .sum().unstack(fill_value=0)
)
month_order = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
heatmap_data = heatmap_data.reindex(columns=[m for m in month_order if m in heatmap_data.columns])
sns.heatmap(heatmap_data,
            ax=ax5,
            cmap=sns.light_palette(ACCENT1, as_cmap=True),
            linewidths=0.5,
            linecolor=DARK_BG,
            annot=True,
            fmt='.0f',
            annot_kws={'size': 6.5, 'color': TEXT_MAIN},
            cbar_kws={'shrink': 0.8})
ax5.set_title("Region x Month Revenue ($)", fontsize=12,
              fontweight='bold', color=TEXT_MAIN, pad=10)
ax5.set_xlabel("Month", color=TEXT_SUB, fontsize=9)
ax5.set_ylabel("Region", color=TEXT_SUB, fontsize=9)
ax5.tick_params(axis='x', rotation=45, labelsize=7.5)
ax5.tick_params(axis='y', rotation=0, labelsize=8)
cbar = ax5.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color=TEXT_SUB)
cbar.outline.set_edgecolor(GRID_COLOR)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_SUB, fontsize=7)

# ── Chart 6: Violin – Revenue Distribution by Product ──
ax6=fig.add_subplot(gs[2,:2])
violin_data= df[["product","net_revenue"]].copy()
violin_data["product"]=violin_data["product"].astype(str)
product_order=df.groupby("product",observed=True)["net_revenue"].median().sort_values(ascending=False).index.tolist()
sns.violinplot(data=violin_data,x="product",y="net_revenue",
               order=product_order,
               palette=PALETTE,
               inner="quartile",
               ax=ax6,
               linewidth=1.2,
               cut=0.5)
ax6.set_title("Revenue Distribution by Product (Violin Plot)",fontsize=13,
              fontweight="bold",color=TEXT_MAIN,pad=10)
ax6.set_xlabel("Product",color=TEXT_SUB)
ax6.set_ylabel("Net Revenue($)",color=TEXT_SUB)
ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x:,.0f}"))
ax6.yaxis.grid(True,zorder=0)
ax6.set_axisbelow(True)
for i, prod in enumerate(product_order):
    med= df[df["product"]==prod]["net_revenue"].median()
    ax6.text(i,med,f"${med:.0f}",va="center",
             fontsize=8,color=TEXT_MAIN,fontweight="bold")

# ── Chart 7: Scatter – Qty vs Revenue ──
ax7=fig.add_subplot(gs[2,2])
scatter_palette={p:PALETTE[i] for i, p in enumerate(products)}
for prod in products:
    subset=df[df["product"]==prod]
    ax7.scatter(subset["quantity"],subset["net_revenue"],
                color=scatter_palette[prod],alpha=0.55,
                s=35,edgecolor="none",label=prod)
m,b=np.polyfit(df["quantity"].values,df["net_revenue"].values,1)
x_line=np.linspace(df["quantity"].min(),df["quantity"].max(),100)
ax7.plot(x_line,m*x_line+b,color=TEXT_MAIN,linewidth=1.8,
         linestyle="--",alpha=0.6)
ax7.text(0.97,0.06,f"r={corr:.3f}",ha="right",va="bottom",
         transform=ax7.transAxes,fontsize=9,color=ACCENT2,
         fontweight="bold",
         bbox=dict(boxstyle="round, pad=0.3",facecolor=DARK_BG,edgecolor=ACCENT2,alpha=0.8))
ax7.set_title("Quantity vs Net Revenue",fontsize=13,
              fontweight="bold",color=TEXT_MAIN,pad=10)
ax7.set_xlabel("Quantity",color=TEXT_SUB)
ax7.set_ylabel("New Revenue($)",color=TEXT_SUB)
ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x:,.0f}"))
ax7.yaxis.grid(True,zorder=0)
ax7.set_axisbelow(True)
ax7.legend(fontsize=7.5,frameon=False,labelcolor=TEXT_SUB,
           loc="upper left",ncol=1)

#Footer
fig.text(0.5,0.012,
         "Sales Analysis Dashboard- Generated with Python(NumPy- Pandas - Matplotlib - Seaborn)  -  2024",
         ha="center",va="center",
         fontsize=8.5,color=TEXT_SUB,style="italic")
output_path="sales_dashboard.png"
plt.savefig(output_path,dpi=150,bbox_inches="tight",
            facecolor=DARK_BG,edgecolor="none")
plt.close()
print(f"\n Dashboard Saved->{output_path}")

# ─────────────────────────────────────────────
# Final Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL BUSINESS INSIGHTS")
print("=" * 60)

worst_region=df.groupby("region",observed=True)["net_revenue"].sum().idxmin()
print(f"""
1.Top-selling product:{best_product}
2.Best-performing region":{best_region}
3.Weakest region:{worst_region}
4.Best month:{best_month_n}(${monthly["revenue"].max():,.2f})
5.Top 10% order contribute:{top_contrib:.1f}% of revenue
6.Electronics market share:{electronics_share:.1f}% 

Total Net Revenue:${total_rev:,.2f}

""")
os.startfile("sales_dashboard.png")
