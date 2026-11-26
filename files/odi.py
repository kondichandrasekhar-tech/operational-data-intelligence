"""
Operational Data Intelligence (ODI) - base project
--------------------------------------------------
Pipeline:  Ingest -> Clean -> KPIs -> Anomaly detection -> Insights -> Report

Run:
    python odi.py                      # uses synthetic data (data/operations.csv)
    python odi.py --input my_file.csv  # your own data (same columns, see README)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
OUT_DIR = Path("output")
REQUIRED_COLS = ["date", "department", "orders", "completed", "avg_processing_min", "cost", "revenue"]


# ----------------------------------------------------------------- 1. INGEST
def generate_sample_data(path: Path, days: int = 120, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic operational dataset (with a few injected problems)."""
    rng = np.random.default_rng(seed)
    depts = {"Front Desk": 1.0, "Housekeeping": 1.4, "Kitchen": 1.8, "Maintenance": 0.7}
    dates = pd.date_range("2026-07-01", periods=days, freq="D")
    rows = []
    for d in dates:
        weekend = 1.25 if d.dayofweek >= 5 else 1.0
        for dept, scale in depts.items():
            orders = int(rng.normal(100 * scale * weekend, 8))
            completed = int(orders * rng.uniform(0.88, 0.99))
            proc = rng.normal(22 / scale + 8, 2.0)
            cost = orders * rng.uniform(4.0, 5.5)
            revenue = completed * rng.uniform(9.0, 11.0)
            rows.append([d, dept, orders, completed, round(proc, 2), round(cost, 2), round(revenue, 2)])
    df = pd.DataFrame(rows, columns=REQUIRED_COLS)

    # inject anomalies + dirty data so the pipeline has something to find/fix
    df.loc[df.sample(4, random_state=1).index, "avg_processing_min"] *= 3.2
    df.loc[df.sample(3, random_state=2).index, "cost"] *= 2.8
    df.loc[df.sample(5, random_state=3).index, "avg_processing_min"] = np.nan
    df = pd.concat([df, df.sample(6, random_state=4)])  # duplicates

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[ingest] {path} not found -> generating synthetic data")
        generate_sample_data(path)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing columns: {missing}")
    print(f"[ingest] loaded {len(df):,} rows from {path}")
    return df


# ------------------------------------------------------------------ 2. CLEAN
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "department"]).drop_duplicates()
    for col in ["orders", "completed", "avg_processing_min", "cost", "revenue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # fill missing numeric values with the department median
    num_cols = ["orders", "completed", "avg_processing_min", "cost", "revenue"]
    df[num_cols] = df.groupby("department")[num_cols].transform(lambda s: s.fillna(s.median()))
    df["completed"] = df[["completed", "orders"]].min(axis=1)  # cannot exceed orders
    df = df.sort_values(["date", "department"]).reset_index(drop=True)
    print(f"[clean] {n0:,} -> {len(df):,} rows (removed duplicates / invalid rows, filled gaps)")
    return df


# ------------------------------------------------------------------- 3. KPIs
def add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["completion_rate"] = df["completed"] / df["orders"]
    df["cost_per_order"] = df["cost"] / df["orders"]
    df["profit"] = df["revenue"] - df["cost"]
    df["margin"] = df["profit"] / df["revenue"]
    return df


def department_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("department").agg(
        orders=("orders", "sum"),
        completed=("completed", "sum"),
        avg_processing_min=("avg_processing_min", "mean"),
        cost=("cost", "sum"),
        revenue=("revenue", "sum"),
    )
    g["completion_rate"] = g["completed"] / g["orders"]
    g["cost_per_order"] = g["cost"] / g["orders"]
    g["profit"] = g["revenue"] - g["cost"]
    g["margin"] = g["profit"] / g["revenue"]
    return g.round(3).sort_values("profit", ascending=False)


# --------------------------------------------------------- 4. ANOMALY DETECT
def detect_anomalies(df: pd.DataFrame, metrics=("avg_processing_min", "cost_per_order", "completion_rate"),
                     window: int = 14, z_thresh: float = 3.0) -> pd.DataFrame:
    """Rolling z-score per department. Flags points far from their recent norm."""
    out = []
    for dept, g in df.groupby("department"):
        g = g.sort_values("date").copy()
        for m in metrics:
            roll_mean = g[m].rolling(window, min_periods=7).mean().shift(1)
            roll_std = g[m].rolling(window, min_periods=7).std().shift(1)
            z = (g[m] - roll_mean) / roll_std.replace(0, np.nan)
            flagged = g[z.abs() > z_thresh].copy()
            flagged["metric"] = m
            flagged["value"] = g.loc[flagged.index, m]
            flagged["z_score"] = z[flagged.index].round(2)
            out.append(flagged[["date", "department", "metric", "value", "z_score"]])
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return res.sort_values("z_score", key=abs, ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------- 5. INSIGHTS
def generate_insights(df: pd.DataFrame, summary: pd.DataFrame, anomalies: pd.DataFrame) -> list[str]:
    ins = []
    best, worst = summary["margin"].idxmax(), summary["margin"].idxmin()
    ins.append(f"Highest margin: {best} ({summary.loc[best, 'margin']:.1%}); lowest: {worst} ({summary.loc[worst, 'margin']:.1%}).")

    slow = summary["avg_processing_min"].idxmax()
    ins.append(f"Slowest processing: {slow} at {summary.loc[slow, 'avg_processing_min']:.1f} min/order on average.")

    low_comp = summary["completion_rate"].idxmin()
    ins.append(f"Lowest completion rate: {low_comp} ({summary.loc[low_comp, 'completion_rate']:.1%}).")

    # week-over-week trend on total profit
    weekly = df.set_index("date").resample("W")["profit"].sum()
    if len(weekly) >= 3:
        change = (weekly.iloc[-2] - weekly.iloc[-3]) / abs(weekly.iloc[-3])
        direction = "up" if change > 0 else "down"
        ins.append(f"Profit was {direction} {abs(change):.1%} in the last full week vs the week before.")

    # weekend effect
    df = df.assign(weekend=df["date"].dt.dayofweek >= 5)
    wk = df.groupby("weekend")["orders"].mean()
    if True in wk.index and False in wk.index:
        ins.append(f"Weekend order volume is {wk[True] / wk[False] - 1:+.1%} vs weekdays.")

    if len(anomalies):
        top = anomalies.iloc[0]
        ins.append(f"{len(anomalies)} anomalies flagged; strongest: {top['department']} on "
                   f"{top['date'].date()} ({top['metric']}, z={top['z_score']}).")
    else:
        ins.append("No anomalies detected.")
    return ins


# ------------------------------------------------------------------- 6. REPORT
def make_charts(df: pd.DataFrame, summary: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Operational Data Intelligence - Dashboard", fontsize=14, fontweight="bold")

    daily = df.groupby("date")[["orders", "profit"]].sum()
    axes[0, 0].plot(daily.index, daily["orders"])
    axes[0, 0].set_title("Total daily orders")

    axes[0, 1].plot(daily.index, daily["profit"], color="green")
    axes[0, 1].set_title("Total daily profit")

    summary["margin"].plot(kind="bar", ax=axes[1, 0], color="steelblue")
    axes[1, 0].set_title("Profit margin by department")
    axes[1, 0].tick_params(axis="x", rotation=20)

    for dept, g in df.groupby("department"):
        axes[1, 1].plot(g["date"], g["avg_processing_min"], label=dept, alpha=0.7)
    pa = anomalies[anomalies["metric"] == "avg_processing_min"]
    if len(pa):
        axes[1, 1].scatter(pa["date"], pa["value"], color="red", zorder=5, label="anomaly")
    axes[1, 1].set_title("Avg processing time (min)")
    axes[1, 1].legend(fontsize=7)

    for ax in axes.flat:
        ax.grid(alpha=0.3)
        if ax is not axes[1, 0]:
            ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dashboard.png", dpi=130)
    plt.close(fig)


def write_report(summary, anomalies, insights) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    summary.to_csv(OUT_DIR / "department_summary.csv")
    anomalies.to_csv(OUT_DIR / "anomalies.csv", index=False)
    lines = ["# Operational Data Intelligence Report", "", "## Key insights"]
    lines += [f"- {i}" for i in insights]
    lines += ["", "## Department summary", "", summary.to_markdown() if hasattr(summary, "to_markdown") and _has_tabulate() else summary.to_string(),
              "", "## Top anomalies", "", anomalies.head(10).to_string(index=False) if len(anomalies) else "None"]
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


# ------------------------------------------------------------------------ MAIN
def run(input_path: Path) -> None:
    df = load_data(input_path)
    df = clean_data(df)
    df = add_kpis(df)
    summary = department_summary(df)
    anomalies = detect_anomalies(df)
    insights = generate_insights(df, summary, anomalies)
    make_charts(df, summary, anomalies)
    write_report(summary, anomalies, insights)

    print("\n=== KEY INSIGHTS ===")
    for i in insights:
        print(" -", i)
    print("\n=== DEPARTMENT SUMMARY ===")
    print(summary.to_string())
    print(f"\nOutputs saved in ./{OUT_DIR}/  (dashboard.png, report.md, department_summary.csv, anomalies.csv)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Operational Data Intelligence pipeline")
    ap.add_argument("--input", type=Path, default=DATA_DIR / "operations.csv")
    run(ap.parse_args().input)
