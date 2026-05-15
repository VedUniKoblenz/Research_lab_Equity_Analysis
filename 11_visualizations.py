import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
from math import pi
from pathlib import Path

from config_new import RESULTS_DIR

# =============================================================================
#  PATHS – CHANGE THIS TO YOUR ACTUAL DATA FOLDER
# =============================================================================
RESULTS_DIR = Path("results")  
PRICE_SUMMARY = RESULTS_DIR / "price_only_summary.csv"
DUAL_SUMMARY = RESULTS_DIR / "dual_fusion_summary.csv"
PRICE_PER_TICKER = RESULTS_DIR / "price_only_per_ticker.csv"
DUAL_PER_TICKER = RESULTS_DIR / "dual_fusion_per_ticker.csv"
PRICE_RETURNS    = RESULTS_DIR / "price_only_returns.csv"
DUAL_RETURNS     = RESULTS_DIR / "dual_fusion_returns.csv"
SECTOR_DELTA     = RESULTS_DIR / "sector_comparison_summary.csv"
MODEL_BY_SECTOR  = RESULTS_DIR / "model_comparison_by_sector.csv"
BENCHMARK_COMP   = RESULTS_DIR / "final_benchmark_comparison.csv"
OUTPUT_DIR = RESULTS_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
#  SECTOR MAPPING (complete from the first script)
# =============================================================================
SECTOR_MAPPING = {
    "AAPL":"Technology","MSFT":"Technology","GOOGL":"Technology","META":"Technology",
    "NVDA":"Technology","AMD":"Technology","AMZN":"Technology","INTC":"Technology",
    "CSCO":"Technology","ORCL":"Technology","IBM":"Technology","ADBE":"Technology",
    "CRM":"Technology","NFLX":"Technology","QCOM":"Technology","TXN":"Technology",
    "AVGO":"Technology","ACN":"Technology","NOW":"Technology","SNOW":"Technology",
    "JPM":"Finance","GS":"Finance","BAC":"Finance","C":"Finance","WFC":"Finance",
    "MS":"Finance","BLK":"Finance","V":"Finance","MA":"Finance","PYPL":"Finance",
    "AXP":"Finance","COF":"Finance","USB":"Finance","PNC":"Finance","SCHW":"Finance",
    "PFE":"Healthcare","JNJ":"Healthcare","MRK":"Healthcare","ABBV":"Healthcare",
    "AMGN":"Healthcare","GILD":"Healthcare","BIIB":"Healthcare","REGN":"Healthcare",
    "VRTX":"Healthcare","BMY":"Healthcare","LLY":"Healthcare","UNH":"Healthcare",
    "CVS":"Healthcare","CI":"Healthcare",
    "XOM":"Energy","CVX":"Energy","COP":"Energy","EOG":"Energy","SLB":"Energy",
    "PSX":"Energy","VLO":"Energy","MPC":"Energy","OXY":"Energy","HAL":"Energy",
    "KO":"Consumer","PG":"Consumer","COST":"Consumer","WMT":"Consumer","TGT":"Consumer",
    "HD":"Consumer","LOW":"Consumer","NKE":"Consumer","SBUX":"Consumer","MCD":"Consumer",
    "DIS":"Consumer","PEP":"Consumer","PM":"Consumer","MO":"Consumer","CL":"Consumer",
    "BA":"Industrials","CAT":"Industrials","GE":"Industrials","HON":"Industrials",
    "MMM":"Industrials","LMT":"Industrials","RTX":"Industrials","UPS":"Industrials",
    "FDX":"Industrials","DE":"Industrials",
    "T":"Telecommunications","VZ":"Telecommunications","TMUS":"Telecommunications",
    "CHTR":"Telecommunications","CMCSA":"Telecommunications",
    "NEE":"Utilities","DUK":"Utilities","SO":"Utilities","D":"Utilities","EXC":"Utilities",
    "AMT":"Real Estate","PLD":"Real Estate","CCI":"Real Estate","EQIX":"Real Estate","SPG":"Real Estate",
    "LIN":"Materials","APD":"Materials","SHW":"Materials","FCX":"Materials","NEM":"Materials",
    "UNP":"Transportation","CSX":"Transportation","NSC":"Transportation","ODFL":"Transportation","DAL":"Transportation",
}

SECTOR_COLORS = {
    "Technology":"#4C72B0","Finance":"#55A868","Healthcare":"#C44E52",
    "Energy":"#8172B2","Consumer":"#CCB974","Industrials":"#64B5CD",
    "Telecommunications":"#E07B54","Utilities":"#76C893","Real Estate":"#B07AA1",
    "Materials":"#FF9F4A","Transportation":"#D5BB67",
}

MODEL_COLORS = {"Price-Only": "#4C72B0", "Dual Fusion": "#DD8452"}
GOOD_COLOR   = "#2CA02C"
BAD_COLOR    = "#D62728"

# =============================================================================
#  DATA LOADING FUNCTIONS
# =============================================================================
def load_summary(path, model_name):
    df = pd.read_csv(path)
    df["Model"] = model_name
    return df

def load_per_ticker(path, model_name):
    df = pd.read_csv(path)
    df["Model"] = model_name
    df["Sector"] = df["ticker"].map(SECTOR_MAPPING)
    return df.dropna(subset=["Sector"])

def load_returns(path, model_name):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = ["portfolio_return"]
    df["Model"] = model_name
    # Convert infinite values to NaN to avoid seaborn/pandas warnings
    df["portfolio_return"] = df["portfolio_return"].replace([np.inf, -np.inf], np.nan)
    return df

# =============================================================================
#  LOAD ALL DATA
# =============================================================================
summary_price = load_summary(PRICE_SUMMARY, "Price-Only")
summary_dual  = load_summary(DUAL_SUMMARY,  "Dual Fusion")
summary_all   = pd.concat([summary_price, summary_dual], ignore_index=True)

per_price = load_per_ticker(PRICE_PER_TICKER, "Price-Only")
per_dual  = load_per_ticker(DUAL_PER_TICKER,  "Dual Fusion")
per_all   = pd.concat([per_price, per_dual], ignore_index=True)

ret_price = load_returns(PRICE_RETURNS, "Price-Only")
ret_dual  = load_returns(DUAL_RETURNS,  "Dual Fusion")

sector_delta = pd.read_csv(SECTOR_DELTA)
model_sector = pd.read_csv(MODEL_BY_SECTOR)
benchmark    = pd.read_csv(BENCHMARK_COMP)

# =============================================================================
#  STYLE SETUP
# =============================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "#F8F9FA",
    "axes.facecolor": "#F8F9FA",
    "axes.grid": True,
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.6,
})

# -----------------------------------------------------------------------------
# FIGURE 1 – Overall Benchmark Comparison (bar plot)
# -----------------------------------------------------------------------------
print("Generating Figure 1 – Overall Benchmark Comparison...")
metrics = ["Sharpe", "Mean IC", "Mean RankIC", "Hit Rate", "MaxDD"]
fig1_data = summary_all.melt(
    id_vars="Model",
    value_vars=metrics,
    var_name="Metric",
    value_name="Value"
)
plt.figure(figsize=(10, 5))
sns.barplot(data=fig1_data, x="Metric", y="Value", hue="Model", palette=MODEL_COLORS)
plt.title("Figure 1 – Overall Benchmark Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure1_overall_benchmark_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 2 – Sector-wise Performance Heatmap (Dual Fusion – Price-Only)
# -----------------------------------------------------------------------------
print("Generating Figure 2 – Sector-wise Performance Heatmap...")
sector_metrics = ["IC", "RankIC", "hit_rate", "Sharpe_annual", "MDD"]
sector_df = per_all.groupby(["Sector", "Model"])[sector_metrics].mean().reset_index()
heatmap_df = (
    sector_df[sector_df["Model"] == "Dual Fusion"].set_index("Sector")[sector_metrics] -
    sector_df[sector_df["Model"] == "Price-Only"].set_index("Sector")[sector_metrics]
)
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="coolwarm", center=0)
plt.title("Figure 2 – Sector-wise Performance (Dual Fusion – Price-Only)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure2_sectorwise_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 3 – Sector Delta Bar Chart (Sharpe Delta)
# -----------------------------------------------------------------------------
print("Generating Figure 3 – Sector Delta Bar Chart...")
sector_sharpe = sector_df.pivot(index="Sector", columns="Model", values="Sharpe_annual")
sector_sharpe["Delta"] = sector_sharpe["Dual Fusion"] - sector_sharpe["Price-Only"]
sector_sharpe = sector_sharpe.sort_values("Delta")
plt.figure(figsize=(8, 6))
plt.barh(
    sector_sharpe.index,
    sector_sharpe["Delta"],
    color=np.where(sector_sharpe["Delta"] >= 0, GOOD_COLOR, BAD_COLOR)
)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Figure 3 – Sector Delta Analysis (Sharpe Delta)", fontsize=14, fontweight="bold")
plt.xlabel("Dual Fusion – Price-Only")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure3_sector_delta.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 4 – Cumulative Return & Drawdown
# -----------------------------------------------------------------------------
print("Generating Figure 4 – Cumulative Returns & Drawdown...")
fig4, (ax_ret, ax_dd) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
fig4.suptitle("Figure 4 – Cumulative Return & Drawdown Comparison", fontsize=14, fontweight="bold")

for model, df_ret, color in [
    ("Price-Only", ret_price, MODEL_COLORS["Price-Only"]),
    ("Dual Fusion", ret_dual,  MODEL_COLORS["Dual Fusion"]),
]:
    cum = (1 + df_ret["portfolio_return"]).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    ax_ret.plot(cum.index, cum.values, label=model, color=color, linewidth=1.8)
    ax_dd.fill_between(dd.index, dd.values, 0, alpha=0.4, color=color)
    ax_dd.plot(dd.index, dd.values, color=color, linewidth=0.8)

ax_ret.axhline(1, color="#999", linewidth=0.8, linestyle="--")
ax_ret.set_ylabel("Cumulative Return (1 = start)", fontsize=10)
ax_ret.legend(fontsize=10)
ax_dd.set_ylabel("Drawdown", fontsize=10)
ax_dd.set_xlabel("Date", fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure4_cumulative_returns_drawdown.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 5 – Rolling Sharpe Ratio (60-day)
# -----------------------------------------------------------------------------
print("Generating Figure 5 – Rolling Sharpe Ratio...")
WINDOW = 60
fig5, ax5 = plt.subplots(figsize=(13, 5))
fig5.suptitle(f"Figure 5 – Rolling {WINDOW}-Day Sharpe Ratio", fontsize=14, fontweight="bold")

for model, df_ret, color in [
    ("Price-Only", ret_price, MODEL_COLORS["Price-Only"]),
    ("Dual Fusion", ret_dual,  MODEL_COLORS["Dual Fusion"]),
]:
    r = df_ret["portfolio_return"]
    roll_sharpe = r.rolling(WINDOW).mean() / r.rolling(WINDOW).std() * np.sqrt(252)
    ax5.plot(roll_sharpe.index, roll_sharpe.values, label=model, color=color, linewidth=1.5)

ax5.axhline(0, color="#999", linewidth=0.8, linestyle="--")
ax5.set_ylabel("Rolling Sharpe (annualised)", fontsize=10)
ax5.set_xlabel("Date", fontsize=10)
ax5.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure5_rolling_sharpe.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 6 – Per-Ticker Sharpe Scatter (Price-Only vs Dual Fusion)
# -----------------------------------------------------------------------------
print("Generating Figure 6 – Per-Ticker Sharpe Scatter...")
p_sharpe = per_price[["ticker", "Sharpe_annual", "Sector"]].rename(columns={"Sharpe_annual": "Price"})
d_sharpe = per_dual[["ticker",  "Sharpe_annual"]].rename(columns={"Sharpe_annual": "Dual"})
scatter_df = p_sharpe.merge(d_sharpe, on="ticker")

fig6, ax6 = plt.subplots(figsize=(9, 9))
fig6.suptitle("Figure 6 – Per-Ticker Sharpe: Price-Only vs Dual Fusion", fontsize=13, fontweight="bold")

for sector, grp in scatter_df.groupby("Sector"):
    ax6.scatter(grp["Price"], grp["Dual"],
                color=SECTOR_COLORS.get(sector, "#888"),
                label=sector, s=60, edgecolors="white", linewidth=0.4, zorder=3)

lim = max(abs(scatter_df[["Price","Dual"]].values.flatten())) * 1.1
ax6.axline((0, 0), slope=1, color="#999", linewidth=0.9, linestyle="--", label="y = x")
ax6.axhline(0, color="#bbb", linewidth=0.5)
ax6.axvline(0, color="#bbb", linewidth=0.5)
ax6.set_xlim(-lim, lim)
ax6.set_ylim(-lim, lim)
ax6.set_xlabel("Price-Only Sharpe", fontsize=10)
ax6.set_ylabel("Dual Fusion Sharpe", fontsize=10)
ax6.legend(fontsize=8, loc="upper left", framealpha=0.8)

for _, row in scatter_df.iterrows():
    delta = row["Dual"] - row["Price"]
    if abs(delta) > 1.5 or abs(row["Dual"]) > 2 or abs(row["Price"]) > 2:
        ax6.annotate(row["ticker"], (row["Price"], row["Dual"]),
                     fontsize=7, xytext=(4, 4), textcoords="offset points", color="#333")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure6_ticker_sharpe_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 7 – Sector Radar Chart (both models)
# -----------------------------------------------------------------------------
print("Generating Figure 7 – Sector Radar Chart...")
sector_metrics_radar = ["Mean IC", "Mean RankIC", "Hit Rate", "Sharpe Ratio"]
pivot_p = model_sector[model_sector["Model"] == "Price-Only"].groupby("Sector")[sector_metrics_radar].mean()
pivot_d = model_sector[model_sector["Model"] == "Dual Fusion"].groupby("Sector")[sector_metrics_radar].mean()

combined = pd.concat([pivot_p, pivot_d])
norm_min = combined.min()
norm_max = combined.max()
norm_p = (pivot_p - norm_min) / (norm_max - norm_min + 1e-9)
norm_d = (pivot_d - norm_min) / (norm_max - norm_min + 1e-9)

sectors = norm_p.index.tolist()
N = len(sector_metrics_radar)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig7, axes = plt.subplots(3, 4, subplot_kw=dict(polar=True), figsize=(16, 12))
fig7.suptitle("Figure 7 – Sector Performance Radar (normalised metrics)", fontsize=14, fontweight="bold")
axes_flat = axes.flatten()

for i, sector in enumerate(sectors):
    ax = axes_flat[i]
    vals_p = norm_p.loc[sector].tolist() + [norm_p.loc[sector].tolist()[0]]
    vals_d = norm_d.loc[sector].tolist() + [norm_d.loc[sector].tolist()[0]]
    ax.plot(angles, vals_p, color=MODEL_COLORS["Price-Only"], linewidth=1.5, label="Price-Only")
    ax.fill(angles, vals_p, color=MODEL_COLORS["Price-Only"], alpha=0.15)
    ax.plot(angles, vals_d, color=MODEL_COLORS["Dual Fusion"], linewidth=1.5, label="Dual Fusion")
    ax.fill(angles, vals_d, color=MODEL_COLORS["Dual Fusion"], alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["IC", "RankIC", "HitRate", "Sharpe"], fontsize=7)
    ax.set_yticks([])
    ax.set_title(sector, size=9, pad=10, fontweight="bold")

for j in range(len(sectors), len(axes_flat)):
    axes_flat[j].set_visible(False)

handles = [mpatches.Patch(color=MODEL_COLORS["Price-Only"], label="Price-Only"),
           mpatches.Patch(color=MODEL_COLORS["Dual Fusion"], label="Dual Fusion")]
fig7.legend(handles=handles, loc="lower right", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure7_sector_radar.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 8 – Return Distribution (KDE + histogram)
# -----------------------------------------------------------------------------
print("Generating Figure 8 – Return Distribution...")
fig8, axes8 = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig8.suptitle("Figure 8 – Daily Return Distribution", fontsize=14, fontweight="bold")

for ax, (model, df_ret, color) in zip(axes8, [
    ("Price-Only", ret_price, MODEL_COLORS["Price-Only"]),
    ("Dual Fusion", ret_dual,  MODEL_COLORS["Dual Fusion"]),
]):
    r = df_ret["portfolio_return"].dropna()
    mean_r = r.mean()
    std_r  = r.std()
    skew_r = r.skew()
    kurt_r = r.kurt()
    ax.hist(r, bins=50, color=color, alpha=0.4, density=True, label="Histogram")
    # seaborn uses a pandas option that is deprecated; suppress that specific FutureWarning here
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*use_inf_as_na.*", category=FutureWarning)
        sns.kdeplot(r, ax=ax, color=color, linewidth=2.5, label="KDE")
    ax.axvline(mean_r, color="#333", linewidth=1.2, linestyle="--", label=f"Mean={mean_r:.4f}")
    ax.axvline(0, color="#aaa", linewidth=0.8, linestyle=":")
    ax.set_title(model, fontsize=12, fontweight="bold")
    ax.set_xlabel("Daily Return", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    stats_txt = f"μ={mean_r:.4f}\nσ={std_r:.4f}\nSkew={skew_r:.2f}\nKurt={kurt_r:.2f}"
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ddd", alpha=0.8))
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure8_return_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 9 – Win/Loss Ticker Profile per Sector (stacked bar)
# -----------------------------------------------------------------------------
print("Generating Figure 9 – Win/Loss Ticker Profile per Sector...")
def win_loss(df, threshold=0):
    df = df.copy()
    df["win"]     = df["Sharpe_annual"] >  threshold
    df["loss"]    = df["Sharpe_annual"] < -threshold
    df["neutral"] = ~(df["win"] | df["loss"])
    return df.groupby("Sector")[["win","loss","neutral"]].sum()

wl_price = win_loss(per_price)
wl_dual  = win_loss(per_dual)

sectors_order = wl_price.index.tolist()
x = np.arange(len(sectors_order))
width = 0.35

fig9, axes9 = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
fig9.suptitle("Figure 9 – Win / Loss Ticker Profile per Sector (Sharpe > 0 = Win)", fontsize=13, fontweight="bold")

for ax, (model, wl_df, color_win) in zip(axes9, [
    ("Price-Only", wl_price, MODEL_COLORS["Price-Only"]),
    ("Dual Fusion", wl_dual,  MODEL_COLORS["Dual Fusion"]),
]):
    wins    = [wl_df.loc[s, "win"]     if s in wl_df.index else 0 for s in sectors_order]
    losses  = [wl_df.loc[s, "loss"]    if s in wl_df.index else 0 for s in sectors_order]
    neutral = [wl_df.loc[s, "neutral"] if s in wl_df.index else 0 for s in sectors_order]
    x_pos = np.arange(len(sectors_order))
    ax.bar(x_pos, wins,    color=GOOD_COLOR, label="Win (Sharpe > 0)", zorder=3)
    ax.bar(x_pos, neutral, bottom=wins, color="#CCCCCC", label="Neutral", zorder=3)
    ax.bar(x_pos, losses,  bottom=np.array(wins)+np.array(neutral),
           color=BAD_COLOR, label="Loss (Sharpe < 0)", zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sectors_order, rotation=45, ha="right", fontsize=8)
    ax.set_title(model, fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Tickers")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure9_win_loss_sector.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 10 – IC vs RankIC per Ticker (size ∝ |Sharpe|)
# -----------------------------------------------------------------------------
print("Generating Figure 10 – IC vs RankIC per Ticker...")
merged_ic = per_price[["ticker","IC","RankIC","Sharpe_annual","Sector"]].copy()
merged_ic["model"] = "Price-Only"
merged_ic2 = per_dual[["ticker","IC","RankIC","Sharpe_annual","Sector"]].copy()
merged_ic2["model"] = "Dual Fusion"
ic_all = pd.concat([merged_ic, merged_ic2])

fig10, axes10 = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
fig10.suptitle("Figure 10 – IC vs RankIC per Ticker (size ∝ |Sharpe|)", fontsize=13, fontweight="bold")

for ax, model in zip(axes10, ["Price-Only", "Dual Fusion"]):
    sub = ic_all[ic_all["model"] == model]
    for sector, grp in sub.groupby("Sector"):
        sizes = np.clip(np.abs(grp["Sharpe_annual"]) * 20, 10, 200)
        ax.scatter(grp["IC"], grp["RankIC"],
                   c=SECTOR_COLORS.get(sector, "#888"),
                   s=sizes, alpha=0.7, label=sector,
                   edgecolors="white", linewidth=0.3, zorder=3)
    ax.axhline(0, color="#bbb", linewidth=0.6)
    ax.axvline(0, color="#bbb", linewidth=0.6)
    ax.set_xlabel("IC", fontsize=10)
    ax.set_ylabel("RankIC", fontsize=10)
    ax.set_title(model, fontsize=11, fontweight="bold")

handles10 = [mpatches.Patch(color=SECTOR_COLORS[s], label=s)
             for s in SECTOR_COLORS if s in ic_all["Sector"].unique()]
fig10.legend(handles=handles10, loc="lower center", ncol=6, fontsize=8,
             bbox_to_anchor=(0.5, -0.08), frameon=False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure10_ic_rankic_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 11 – Summary Metrics Dashboard
# -----------------------------------------------------------------------------
print("Generating Figure 11 – Summary Metrics Dashboard...")
bench = pd.read_csv(BENCHMARK_COMP)
bench["Model"] = bench["Model"].str.replace(r"Model \d \((.+)\)", r"\1", regex=True)

metrics_map = {
    "Sharpe": "Sharpe Ratio",
    "MaxDD": "Max Drawdown",
    "Mean IC": "Mean IC",
    "IC IR": "IC IR",
    "Hit Rate": "Hit Rate",
    "Turnover": "Turnover",
}

fig11, axes11 = plt.subplots(2, 3, figsize=(14, 8))
fig11.suptitle("Figure 11 – Model Summary Metrics Dashboard", fontsize=14, fontweight="bold")
axes11 = axes11.flatten()

for i, (col, label) in enumerate(metrics_map.items()):
    ax = axes11[i]
    if col not in bench.columns:
        ax.set_visible(False)
        continue
    vals = bench[col].values
    models = bench["Model"].values
    colors = [MODEL_COLORS.get(m, "#888") for m in models]
    x_models = np.arange(len(models))
    bars = ax.bar(x_models, vals, color=colors, width=0.5, zorder=3)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_ylabel(label, fontsize=9)
    ax.set_xticks(x_models)
    ax.set_xticklabels(models, rotation=10, ha="right", fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(abs(vals))*0.03),
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="#999", linewidth=0.7, linestyle="--")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure11_summary_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 12 – Sector Full Metrics Heatmap (absolute values)
# -----------------------------------------------------------------------------
print("Generating Figure 12 – Sector Full Metrics Heatmap...")
sector_agg = model_sector.groupby(["Sector","Model"])[["Mean IC","Mean RankIC","Hit Rate","Sharpe Ratio","Max Drawdown"]].mean().reset_index()

fig12, axes12 = plt.subplots(1, 2, figsize=(14, 7))
fig12.suptitle("Figure 12 – Sector Avg Metrics Heatmap (absolute)", fontsize=13, fontweight="bold")

for ax, model in zip(axes12, ["Price-Only", "Dual Fusion"]):
    sub = sector_agg[sector_agg["Model"] == model].set_index("Sector")[["Mean IC","Mean RankIC","Hit Rate","Sharpe Ratio","Max Drawdown"]]
    sns.heatmap(sub, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
                ax=ax, linewidths=0.4, cbar_kws={"shrink":0.8})
    ax.set_title(model, fontsize=11, fontweight="bold")
    ax.set_ylabel("")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure12_sector_full_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 13 – Sector Delta Bubble Chart (ΔSharpe vs ΔIC, bubble ∝ |ΔHitRate|)
# -----------------------------------------------------------------------------
print("Generating Figure 13 – Sector Delta Bubble Chart...")
fig13, ax13 = plt.subplots(figsize=(11, 8))
fig13.suptitle("Figure 13 – Sector Delta: ΔSharpe vs ΔIC (bubble ∝ |ΔHitRate|)", fontsize=13, fontweight="bold")

for _, row in sector_delta.iterrows():
    sector = row["Sector"]
    x = row["Avg ΔIC"]
    y = row["Avg ΔSharpe"]
    size = max(abs(row["Avg ΔHitRate"]) * 5000, 80)
    color = SECTOR_COLORS.get(sector, "#888")
    ax13.scatter(x, y, s=size, color=color, alpha=0.75, edgecolors="white", linewidth=1, zorder=3)
    ax13.annotate(sector, (x, y), fontsize=9, ha="center", va="bottom", xytext=(0, 10), textcoords="offset points")

ax13.axhline(0, color="#999", linewidth=0.9, linestyle="--")
ax13.axvline(0, color="#999", linewidth=0.9, linestyle="--")
ax13.set_xlabel("Avg ΔIC (Dual Fusion − Price-Only)", fontsize=11)
ax13.set_ylabel("Avg ΔSharpe", fontsize=11)
ax13.text(0.97, 0.97, "Bubble size ∝ |ΔHitRate|", transform=ax13.transAxes, fontsize=9, ha="right", va="top", color="#555")

# Quadrant annotations
for text, xy, color_q in [
    ("Better IC\nBetter Sharpe", (0.98, 0.02), GOOD_COLOR),
    ("Worse IC\nBetter Sharpe", (0.02, 0.02), "#FF8C00"),
    ("Worse IC\nWorse Sharpe", (0.02, 0.98), BAD_COLOR),
    ("Better IC\nWorse Sharpe", (0.98, 0.98), "#888"),
]:
    ax13.text(*xy, text, transform=ax13.transAxes, fontsize=8,
              ha="right" if xy[0] > 0.5 else "left",
              va="bottom" if xy[1] < 0.5 else "top",
              color=color_q, alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure13_sector_delta_bubble.png", dpi=150, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------------
# FIGURE 14 – Top / Bottom 10 Tickers by Sharpe Delta
# -----------------------------------------------------------------------------
print("Generating Figure 14 – Top / Bottom 10 Tickers by Sharpe Delta...")
p_s = per_price[["ticker","Sharpe_annual","Sector"]].rename(columns={"Sharpe_annual":"Price"})
d_s = per_dual[["ticker","Sharpe_annual"]].rename(columns={"Sharpe_annual":"Dual"})
delta_df = p_s.merge(d_s, on="ticker")
delta_df["Delta"] = delta_df["Dual"] - delta_df["Price"]
delta_df = delta_df.sort_values("Delta")

top10  = delta_df.tail(10)
bot10  = delta_df.head(10)
plot_df = pd.concat([bot10, top10])

fig14, ax14 = plt.subplots(figsize=(10, 9))
fig14.suptitle("Figure 14 – Top / Bottom 10 Tickers: Sharpe Delta (Dual − Price)", fontsize=13, fontweight="bold")

colors14 = [GOOD_COLOR if v >= 0 else BAD_COLOR for v in plot_df["Delta"]]
bars14 = ax14.barh(
    [f"{r['ticker']} ({r['Sector'][:3].upper()})" for _, r in plot_df.iterrows()],
    plot_df["Delta"].values,
    color=colors14, edgecolor="white"
)
ax14.axvline(0, color="#555", linewidth=0.8)
ax14.set_xlabel("Sharpe Delta (Dual Fusion − Price-Only)", fontsize=10)
ax14.set_ylabel("Ticker (Sector abbrev.)", fontsize=10)

for bar, val in zip(bars14, plot_df["Delta"].values):
    pad = 0.02 if val >= 0 else -0.02
    ha  = "left" if val >= 0 else "right"
    ax14.text(val + pad, bar.get_y() + bar.get_height()/2,
              f"{val:+.2f}", va="center", ha=ha, fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure14_top_bottom_tickers.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nAll 14 figures saved to:", OUTPUT_DIR)