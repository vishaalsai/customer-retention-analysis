"""
clv.py
------
Module for estimating Customer Lifetime Value (CLV) per customer and per
segment. Combines average order value, purchase frequency, and predicted
retention period to quantify the economic value of each customer group.

CLV Methodology (simple cohort model):
    CLV_basic    = AOV × purchases_per_year × time_horizon_years
                 = (Monetary / Frequency) × (Frequency / 2) × 3
                 = Monetary × 1.5

    Where the dataset spans ~2 years (Dec 2009 – Dec 2011), so
    frequency_per_year = Frequency / 2, and we project 3 years forward.

    CLV_adjusted = CLV_basic × (1 − churn_probability)
    Discounts the raw projection by the probability the customer will
    not generate any future revenue.

Pipeline:
    1. compute_basic_clv()        — Raw 3-year CLV per customer
    2. compute_risk_adjusted_clv()— Churn-discounted CLV
    3. compute_clv_segments()     — Segment-level CLV summary
    4. compute_retention_roi()    — ROI of a £10 retention intervention
    5. compute_clv_tiers()        — Platinum / Gold / Silver / Bronze tiers
    6. save_clv_results()         — Persist outputs

Usage:
    python src/clv.py
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_SPAN_YEARS = 2      # Dataset covers Dec 2009 – Dec 2011
TIME_HORIZON_YEARS = 3      # Forward projection window
RETENTION_CHURN_RATE = 0.05  # Assumed residual churn after intervention


# ---------------------------------------------------------------------------
# 1. BASIC CLV
# ---------------------------------------------------------------------------

def compute_basic_clv(rfm: pd.DataFrame, time_horizon_years: int = TIME_HORIZON_YEARS) -> pd.DataFrame:
    """
    Compute a simple historical-cohort CLV for each customer.

    Formula derivation:
        avg_order_value    = Monetary / Frequency
        purchases_per_year = Frequency / DATASET_SPAN_YEARS
        CLV_basic          = avg_order_value × purchases_per_year × time_horizon_years

    Algebraically this reduces to:
        CLV_basic = Monetary × (time_horizon_years / DATASET_SPAN_YEARS)

    With a 3-year horizon over a 2-year dataset: CLV_basic = Monetary × 1.5

    This is a zero-churn baseline — it assumes the customer continues
    purchasing at their historical rate indefinitely. Churn discounting
    is applied separately in compute_risk_adjusted_clv().

    Parameters
    ----------
    rfm : pd.DataFrame
        DataFrame with Recency, Frequency, Monetary, churn_probability columns.
    time_horizon_years : int
        Number of years to project CLV forward. Default: 3.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new column: CLV_basic.
    """
    multiplier = time_horizon_years / DATASET_SPAN_YEARS   # = 1.5

    rfm = rfm.copy()
    rfm["avg_order_value"] = rfm["Monetary"] / rfm["Frequency"]
    rfm["CLV_basic"]       = rfm["Monetary"] * multiplier

    print("\n" + "=" * 60)
    print(f"  BASIC CLV  (3-year projection, zero-churn baseline)")
    print("=" * 60)
    print(f"  Formula : CLV = Monetary × {multiplier:.1f}")
    print(f"  (Projects {time_horizon_years} yrs forward from {DATASET_SPAN_YEARS}-yr history)\n")
    print(f"  {'Metric':<25}  {'Value':>15}")
    print("  " + "-" * 43)
    print(f"  {'Mean CLV':<25}  £{rfm['CLV_basic'].mean():>14,.0f}")
    print(f"  {'Median CLV':<25}  £{rfm['CLV_basic'].median():>14,.0f}")
    print(f"  {'Std Dev CLV':<25}  £{rfm['CLV_basic'].std():>14,.0f}")
    print(f"  {'Min CLV':<25}  £{rfm['CLV_basic'].min():>14,.0f}")
    print(f"  {'Max CLV':<25}  £{rfm['CLV_basic'].max():>14,.0f}")
    print(f"  {'Total CLV (all customers)':<25}  £{rfm['CLV_basic'].sum():>14,.0f}")
    print("=" * 60)

    return rfm


# ---------------------------------------------------------------------------
# 2. RISK-ADJUSTED CLV
# ---------------------------------------------------------------------------

def compute_risk_adjusted_clv(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Discount basic CLV by the probability that the customer will churn.

    Formula:
        CLV_adjusted = CLV_basic × (1 − churn_probability)

    Interpretation:
        - A customer with CLV_basic = £5,000 and churn_probability = 0.90
          has an expected future value of £500 — only 10% of their
          theoretical maximum, because we only expect to realise that
          value 10% of the time.
        - A customer with CLV_basic = £5,000 and churn_probability = 0.02
          has CLV_adjusted ≈ £4,900 — nearly their full potential value.

    Revenue at risk = total CLV_basic − total CLV_adjusted
                    = the revenue we expect to lose to churn.

    Parameters
    ----------
    rfm : pd.DataFrame
        DataFrame with CLV_basic and churn_probability columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new column: CLV_adjusted.
    """
    rfm = rfm.copy()
    rfm["CLV_adjusted"] = rfm["CLV_basic"] * (1 - rfm["churn_probability"])

    total_basic    = rfm["CLV_basic"].sum()
    total_adjusted = rfm["CLV_adjusted"].sum()
    at_risk        = total_basic - total_adjusted
    at_risk_pct    = at_risk / total_basic * 100

    print("\n" + "=" * 60)
    print("  RISK-ADJUSTED CLV")
    print("=" * 60)
    print(f"  {'Metric':<30}  {'Value':>15}")
    print("  " + "-" * 48)
    print(f"  {'Mean CLV_adjusted':<30}  £{rfm['CLV_adjusted'].mean():>14,.0f}")
    print(f"  {'Median CLV_adjusted':<30}  £{rfm['CLV_adjusted'].median():>14,.0f}")
    print(f"  {'Total CLV_basic (full potential)':<30}  £{total_basic:>14,.0f}")
    print(f"  {'Total CLV_adjusted (expected)':<30}  £{total_adjusted:>14,.0f}")
    print(f"  {'Revenue at risk':<30}  £{at_risk:>14,.0f}  ({at_risk_pct:.1f}% of potential)")
    print()
    print(f"  Of the total projected £{total_basic:,.0f} in 3-year revenue,")
    print(f"  £{at_risk:,.0f} ({at_risk_pct:.1f}%) is at risk due to customer churn.")
    print("=" * 60)

    return rfm


# ---------------------------------------------------------------------------
# 3. CLV BY SEGMENT
# ---------------------------------------------------------------------------

def compute_clv_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate CLV metrics at the segment level.

    Produces a summary showing each segment's:
        - Customer count and share of total
        - Mean and total CLV (basic and adjusted)
        - Revenue at risk (basic − adjusted)
        - Share of total CLV_basic (to surface Pareto dynamics)

    Parameters
    ----------
    rfm : pd.DataFrame
        DataFrame with CLV_basic, CLV_adjusted, Segment columns.

    Returns
    -------
    pd.DataFrame
        Segment-level summary DataFrame.
    """
    summary = (
        rfm.groupby("Segment", as_index=True)
        .agg(
            Customers      =("CustomerID",    "count"),
            Mean_CLV_basic =("CLV_basic",      "mean"),
            Mean_CLV_adj   =("CLV_adjusted",   "mean"),
            Total_CLV_basic=("CLV_basic",      "sum"),
            Total_CLV_adj  =("CLV_adjusted",   "sum"),
        )
        .round(0)
        .astype(int)
    )

    total_customers = summary["Customers"].sum()
    total_clv       = summary["Total_CLV_basic"].sum()

    summary["Pct_Customers"]  = (summary["Customers"]       / total_customers * 100).round(1)
    summary["Pct_Total_CLV"]  = (summary["Total_CLV_basic"] / total_clv       * 100).round(1)
    summary["Revenue_at_Risk"]= summary["Total_CLV_basic"] - summary["Total_CLV_adj"]
    summary = summary.sort_values("Total_CLV_basic", ascending=False)

    print("\n" + "=" * 85)
    print("  CLV SEGMENT SUMMARY")
    print("=" * 85)
    print(summary.to_string())
    print("=" * 85)

    return summary.reset_index()


# ---------------------------------------------------------------------------
# 4. RETENTION ROI
# ---------------------------------------------------------------------------

def compute_retention_roi(
    rfm: pd.DataFrame,
    cost_per_customer: float = 10.0,
) -> pd.DataFrame:
    """
    Estimate the ROI of a single retention intervention per customer.

    Assumption: a retention intervention (e.g., win-back email + discount)
    costs £`cost_per_customer` and, if successful, reduces the customer's
    churn probability to RETENTION_CHURN_RATE (5%).

    Formula:
        CLV_if_retained = CLV_basic × (1 − RETENTION_CHURN_RATE)
                        = CLV_basic × 0.95
        ROI = (CLV_if_retained − CLV_adjusted) − cost_per_customer
            = CLV_basic × (churn_probability − 0.05) − cost_per_customer

    A positive ROI means the expected value gained from retaining the
    customer exceeds the cost of the intervention.

    Note: For Champions (churn_probability ≈ 0), the intervention would
    actually make things worse (increasing churn from ~0% to 5% post-campaign).
    The ROI will be strongly negative — correctly flagging them as
    "not worth spending retention budget on."

    Parameters
    ----------
    rfm : pd.DataFrame
        DataFrame with CLV_basic, CLV_adjusted, churn_probability.
    cost_per_customer : float
        Cost in £ of one retention outreach per customer. Default: £10.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new columns: Retention_ROI, Worth_Retaining.
    """
    rfm = rfm.copy()

    clv_if_retained      = rfm["CLV_basic"] * (1 - RETENTION_CHURN_RATE)
    rfm["Retention_ROI"] = (clv_if_retained - rfm["CLV_adjusted"]) - cost_per_customer
    rfm["Worth_Retaining"] = rfm["Retention_ROI"] > 0

    worth    = rfm["Worth_Retaining"].sum()
    not_worth = len(rfm) - worth
    total_roi = rfm.loc[rfm["Worth_Retaining"], "Retention_ROI"].sum()

    print("\n" + "=" * 60)
    print(f"  RETENTION ROI ANALYSIS  (cost = £{cost_per_customer:.0f} per customer)")
    print("=" * 60)
    print(f"  Assumption: intervention reduces churn to {RETENTION_CHURN_RATE * 100:.0f}%")
    print()
    print(f"  Customers worth retaining   : {worth:,}  ({worth / len(rfm) * 100:.1f}%)")
    print(f"  Customers NOT worth retaining: {not_worth:,}  ({not_worth / len(rfm) * 100:.1f}%)")
    print()
    print(f"  Total expected ROI (worth-retaining subset):  £{total_roi:,.0f}")
    print(f"  Total intervention cost (worth-retaining):    £{worth * cost_per_customer:,.0f}")
    print(f"  Net return on retention budget:               £{total_roi - worth * cost_per_customer:,.0f}")
    print()
    print("  Breakdown by segment:")
    seg_counts = rfm.groupby("Segment")["Worth_Retaining"].value_counts()
    print(seg_counts.to_string())
    print("=" * 60)

    return rfm


# ---------------------------------------------------------------------------
# 5. CLV TIERS
# ---------------------------------------------------------------------------

def compute_clv_tiers(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each customer to a CLV tier based on CLV_adjusted.

    Tier boundaries (percentile-based on CLV_adjusted):
        Platinum : top 10%  (>= 90th percentile)
        Gold     : 10–30%   (70th–90th percentile)
        Silver   : 30–60%   (40th–70th percentile)
        Bronze   : bottom 40% (< 40th percentile)

    Note: because CLV_adjusted is strongly bimodal (Champions ≈ CLV_basic,
    Dormant ≈ 0), many Dormant customers will land in Bronze regardless
    of their historical spend.

    Parameters
    ----------
    rfm : pd.DataFrame
        DataFrame with CLV_adjusted column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new column: CLV_tier.
    """
    p40 = rfm["CLV_adjusted"].quantile(0.40)
    p70 = rfm["CLV_adjusted"].quantile(0.70)
    p90 = rfm["CLV_adjusted"].quantile(0.90)

    def _assign_tier(val):
        if val >= p90:
            return "Platinum"
        elif val >= p70:
            return "Gold"
        elif val >= p40:
            return "Silver"
        else:
            return "Bronze"

    rfm = rfm.copy()
    rfm["CLV_tier"] = rfm["CLV_adjusted"].apply(_assign_tier)

    tier_order = ["Platinum", "Gold", "Silver", "Bronze"]
    total_clv  = rfm["CLV_adjusted"].sum()

    print("\n" + "=" * 65)
    print("  CLV TIER DISTRIBUTION")
    print("=" * 65)
    print(f"  {'Tier':<12}  {'Customers':>10}  {'% Base':>8}  {'Avg CLV':>12}  {'% Total CLV':>12}")
    print("  " + "-" * 58)
    for tier in tier_order:
        subset  = rfm[rfm["CLV_tier"] == tier]
        count   = len(subset)
        pct_cust = count / len(rfm) * 100
        avg_clv  = subset["CLV_adjusted"].mean()
        pct_clv  = subset["CLV_adjusted"].sum() / total_clv * 100
        print(f"  {tier:<12}  {count:>10,}  {pct_cust:>7.1f}%  £{avg_clv:>11,.0f}  {pct_clv:>11.1f}%")
    print("=" * 65)

    return rfm


# ---------------------------------------------------------------------------
# 6. SAVE CLV RESULTS
# ---------------------------------------------------------------------------

def save_clv_results(
    rfm: pd.DataFrame,
    segment_summary: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Persist the CLV-enriched customer DataFrame and segment summary to CSV.

    Parameters
    ----------
    rfm : pd.DataFrame
        Full customer-level DataFrame with all CLV columns.
    segment_summary : pd.DataFrame
        Segment-level summary from compute_clv_segments().
    output_path : str
        Destination path for the customer CSV
        (e.g. 'data/processed/clv_results.csv').
    """
    out_dir  = os.path.dirname(output_path)
    seg_path = os.path.join(out_dir, "clv_segment_summary.csv")

    os.makedirs(out_dir, exist_ok=True)
    rfm.to_csv(output_path, index=False)
    segment_summary.to_csv(seg_path, index=False)

    print(f"\n[save_clv_results] Customer data  -> {output_path}  ({len(rfm):,} rows)")
    print(f"[save_clv_results] Segment summary -> {seg_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rfm = pd.read_csv(
        "data/processed/churn_results.csv",
        index_col=None,   # CustomerID is a regular column
    )

    rfm              = compute_basic_clv(rfm)
    rfm              = compute_risk_adjusted_clv(rfm)
    segment_summary  = compute_clv_segments(rfm)
    rfm              = compute_retention_roi(rfm, cost_per_customer=10.0)
    rfm              = compute_clv_tiers(rfm)
    save_clv_results(rfm, segment_summary, "data/processed/clv_results.csv")
