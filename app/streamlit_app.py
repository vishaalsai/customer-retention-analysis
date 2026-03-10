"""
streamlit_app.py  ·  Phase 5
-----------------------------
Customer Retention Analytics — 5-page Streamlit dashboard.

Pages:
    1. Executive Summary   — KPIs, segment overview, model scores
    2. Customer Segments   — RFM distributions, 3D scatter, quadrant matrix
    3. Churn Prediction    — Model artefacts, risk tiers, customer lookup
    4. Customer Lifetime Value — CLV tiers, retention ROI
    5. Retention Strategy  — Playbooks, budget allocation, methodology

Run from project root:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"

# ── Page config  (must be the very first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="Customer Retention Analytics",
    page_icon="🛒",
    layout="wide",
)

# ── Colour palette ────────────────────────────────────────────────────────────
SEG_COLORS = {
    "Champions":        "#2ecc71",
    "Dormant / At-Risk": "#e74c3c",
}
TIER_COLORS = {
    "Platinum": "#FFD700",
    "Gold":     "#C0C0C0",
    "Silver":   "#CD7F32",
    "Bronze":   "#A9A9A9",
}
BLUE = "#3498db"

# ── Cached data loaders ───────────────────────────────────────────────────────
@st.cache_data
def load_clv() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "clv_results.csv")
    # Ensure boolean dtype regardless of how pandas read it
    if df["Worth_Retaining"].dtype == object:
        df["Worth_Retaining"] = df["Worth_Retaining"].str.strip() == "True"
    return df

@st.cache_data
def load_churn() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "churn_results.csv")

@st.cache_data
def load_segment_summary() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "clv_segment_summary.csv")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Customer Retention Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        options=[
            "🏠 Executive Summary",
            "👥 Customer Segments",
            "🔮 Churn Prediction",
            "💰 Customer Lifetime Value",
            "🎯 Retention Strategy",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "**Key Stats**\n\n"
        "| Metric | Value |\n"
        "|---|---|\n"
        "| Customers | **5,878** |\n"
        "| Revenue at Risk | **£3,359,853** |\n"
        "| Model AUC | **0.999** |"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Executive Summary
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Executive Summary":
    st.title("🏠 Executive Summary")
    st.markdown(
        "High-level overview of customer health, revenue risk, and model performance."
    )

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers",    "5,878")
    c2.metric("Historical Revenue", "£17,374,804")
    c3.metric(
        "Revenue at Risk", "£3,359,853",
        delta="-12.9% of potential", delta_color="inverse",
    )
    c4.metric(
        "Retention ROI", "£3,089,821",
        delta="80x return on £38,600 budget",
    )

    st.markdown("---")

    # ── Pie  +  CLV bar ───────────────────────────────────────────────────────
    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.subheader("Customer Segment Breakdown")
        seg_data = pd.DataFrame({
            "Segment":   ["Champions", "Dormant / At-Risk"],
            "Customers": [2312,         3566],
        })
        fig_pie = px.pie(
            seg_data, names="Segment", values="Customers",
            color="Segment", color_discrete_map=SEG_COLORS,
            hole=0.4,
        )
        fig_pie.update_traces(textinfo="label+percent+value")
        fig_pie.update_layout(
            margin=dict(t=10, b=10),
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        st.subheader("Mean CLV by Segment  (Basic vs Risk-Adjusted)")
        seg_summary = load_segment_summary()
        fig_clv = go.Figure([
            go.Bar(
                name="Mean CLV Basic",
                x=seg_summary["Segment"],
                y=seg_summary["Mean_CLV_basic"],
                marker_color=[SEG_COLORS.get(s, BLUE) for s in seg_summary["Segment"]],
            ),
            go.Bar(
                name="Mean CLV Adjusted",
                x=seg_summary["Segment"],
                y=seg_summary["Mean_CLV_adj"],
                marker_color=[SEG_COLORS.get(s, BLUE) for s in seg_summary["Segment"]],
                opacity=0.5,
            ),
        ])
        fig_clv.update_layout(
            barmode="group",
            yaxis_title="Mean CLV (£)",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_clv, use_container_width=True)

    # ── Business narrative ────────────────────────────────────────────────────
    st.info(
        "**Analysis of 5,878 customers reveals a critical revenue concentration risk:** "
        "39.3% of customers (Champions) generate 87.4% of projected revenue. Meanwhile, "
        "60.7% of customers are dormant or at-risk, representing £3.35M in revenue at risk. "
        "A targeted retention programme focusing on the 3,860 highest-ROI customers could "
        "yield £3.09M in recovered revenue — an 80x return on a £10-per-customer intervention budget."
    )

    # ── Model performance metrics ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Churn Prediction Model Performance")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  "98.3%")
    m2.metric("Precision", "98.7%")
    m3.metric("Recall",    "98.5%")
    m4.metric("F1 Score",  "98.6%")
    m5.metric("ROC-AUC",   "0.999")

    st.caption(
        "Dashboard built on Phase 1–4 pipeline outputs. "
        "All figures derived from UCI Online Retail II dataset (Dec 2009 – Dec 2011)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Customer Segments
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segments")
    st.markdown(
        "RFM-based segmentation: K-Means k=2 (silhouette=0.439). "
        "**Champions** = high recency, high frequency, high spend. "
        "**Dormant / At-Risk** = low engagement, high churn probability."
    )

    df = load_clv()

    # ── Segment overview table ────────────────────────────────────────────────
    st.subheader("Segment Overview")
    seg_tbl = (
        df.groupby("Segment", as_index=False)
        .agg(
            Customers     =("CustomerID",  "count"),
            Avg_Recency   =("Recency",      "mean"),
            Avg_Frequency =("Frequency",    "mean"),
            Avg_Monetary  =("Monetary",     "mean"),
            Avg_CLV       =("CLV_adjusted", "mean"),
        )
        .round(1)
    )
    seg_tbl.insert(
        2, "% of Total",
        (seg_tbl["Customers"] / seg_tbl["Customers"].sum() * 100)
        .round(1).astype(str) + "%",
    )
    seg_tbl.columns = [
        "Segment", "Customers", "% of Total",
        "Avg Recency (days)", "Avg Frequency", "Avg Monetary (£)", "Avg CLV (£)",
    ]
    st.dataframe(seg_tbl, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── RFM histograms ────────────────────────────────────────────────────────
    st.subheader("RFM Distributions by Segment")
    h1, h2, h3 = st.columns(3)
    for widget_col, rfm_col, title in [
        (h1, "Recency",   "Recency  (days since last purchase)"),
        (h2, "Frequency", "Frequency  (total orders)"),
        (h3, "Monetary",  "Monetary  (total spend £)"),
    ]:
        with widget_col:
            fig = px.histogram(
                df, x=rfm_col, color="Segment",
                color_discrete_map=SEG_COLORS,
                nbins=40, barmode="overlay", opacity=0.75,
                title=title,
            )
            fig.update_layout(margin=dict(t=40, b=10), showlegend=(rfm_col == "Recency"))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── 3D scatter  +  profile cards ─────────────────────────────────────────
    r3l, r3r = st.columns([3, 2])

    with r3l:
        st.subheader("3D RFM Customer Map")
        plot_df = df.sample(min(len(df), 3000), random_state=42)
        fig_3d = px.scatter_3d(
            plot_df,
            x="Recency", y="Frequency", z="Monetary",
            color="Segment", color_discrete_map=SEG_COLORS,
            opacity=0.55,
            labels={
                "Recency":   "Recency (days)",
                "Frequency": "Orders",
                "Monetary":  "Spend (£)",
            },
        )
        fig_3d.update_traces(marker=dict(size=3))
        fig_3d.update_layout(margin=dict(t=10, b=10), height=480)
        st.plotly_chart(fig_3d, use_container_width=True)

    with r3r:
        st.subheader("Segment Profiles")
        PROFILE_META = {
            "Champions": {
                "emoji":  "🏆",
                "churn":  "Low (< 3%)",
                "action": "Loyalty rewards, VIP access, early product releases, referral incentives.",
            },
            "Dormant / At-Risk": {
                "emoji":  "⚠️",
                "churn":  "High (> 95%)",
                "action": "Win-back email, personalised discount, re-engagement sequence.",
            },
        }
        for seg, meta in PROFILE_META.items():
            sub = df[df["Segment"] == seg]
            with st.expander(
                f"{meta['emoji']} {seg}  ({len(sub):,} customers)",
                expanded=(seg == "Champions"),
            ):
                st.markdown(f"**Customers:** {len(sub):,}  ({len(sub)/len(df)*100:.1f}%)")
                st.markdown(f"**Avg Recency:** {sub['Recency'].mean():.0f} days")
                st.markdown(f"**Avg Frequency:** {sub['Frequency'].mean():.1f} orders")
                st.markdown(f"**Avg Monetary:** £{sub['Monetary'].mean():,.0f}")
                st.markdown(f"**Avg CLV:** £{sub['CLV_adjusted'].mean():,.0f}")
                st.markdown(f"**Churn Risk:** {meta['churn']}")
                st.markdown(f"**Recommended Action:** {meta['action']}")

    st.markdown("---")

    # ── Quadrant scatter ──────────────────────────────────────────────────────
    st.subheader("Customer Risk vs Value Matrix")
    clv_mid = float(df["CLV_adjusted"].quantile(0.75))

    fig_quad = px.scatter(
        df,
        x="churn_probability", y="CLV_adjusted",
        color="Segment", size="Monetary", size_max=15,
        color_discrete_map=SEG_COLORS,
        opacity=0.55,
        labels={
            "churn_probability": "Churn Probability",
            "CLV_adjusted":      "Risk-Adjusted CLV (£)",
        },
    )
    fig_quad.add_vline(x=0.5,     line_dash="dash", line_color="gray", line_width=1)
    fig_quad.add_hline(y=clv_mid, line_dash="dash", line_color="gray", line_width=1)

    ann = dict(showarrow=False)
    fig_quad.add_annotation(x=0.12, y=clv_mid * 1.65, text="🏆 Protect: Champions",
                            **ann, font=dict(size=13, color="#27ae60"))
    fig_quad.add_annotation(x=0.82, y=clv_mid * 1.65, text="🚨 Urgent: Save These",
                            **ann, font=dict(size=13, color="#c0392b"))
    fig_quad.add_annotation(x=0.12, y=clv_mid * 0.07, text="🌱 Nurture: Grow These",
                            **ann, font=dict(size=13, color=BLUE))
    fig_quad.add_annotation(x=0.82, y=clv_mid * 0.07, text="⚠️ Deprioritize",
                            **ann, font=dict(size=13, color="#7f8c8d"))
    fig_quad.update_layout(height=520, margin=dict(t=20, b=20))
    st.plotly_chart(fig_quad, use_container_width=True)

    st.caption(
        "Quadrant lines: vertical at churn_probability = 0.5; "
        "horizontal at CLV 75th percentile (≈ £3,088). "
        "Point size proportional to historical spend (Monetary)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Churn Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Churn Prediction":
    st.title("🔮 Churn Prediction")
    st.markdown(
        "XGBoost binary classifier trained on Recency, Frequency, Monetary. "
        "Target: Champions (0) vs Dormant / At-Risk (1). "
        "Accuracy 98.3%  ·  AUC 0.999."
    )

    df_churn = load_churn()
    df_clv   = load_clv()   # includes CLV_tier, Worth_Retaining, CLV columns

    # ── Row 1: metrics  +  confusion matrix ──────────────────────────────────
    r1l, r1r = st.columns(2)

    with r1l:
        st.subheader("Model Performance Metrics")
        metrics_df = pd.DataFrame({
            "Metric":     ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            "Value":      ["98.3%",    "98.7%",     "98.5%",  "98.6%",    "0.999"],
            "Assessment": ["Excellent","Excellent",  "Excellent","Excellent","Near-perfect"],
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.markdown(
            "**Top churn driver: Frequency (53.3%)**  \n"
            "Customers who stop purchasing frequently are the earliest-warning churn signal."
        )

    with r1r:
        st.subheader("Confusion Matrix")
        cm_path = DATA_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.warning("confusion_matrix.png not found in data/processed/")

    st.markdown("---")

    # ── Row 2: ROC  +  Feature Importance ────────────────────────────────────
    r2l, r2r = st.columns(2)

    with r2l:
        st.subheader("ROC Curve  (AUC = 0.999)")
        roc_path = DATA_DIR / "roc_curve.png"
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        else:
            st.warning("roc_curve.png not found in data/processed/")

    with r2r:
        st.subheader("Feature Importance")
        fi_path = DATA_DIR / "feature_importance.png"
        if fi_path.exists():
            st.image(str(fi_path), use_container_width=True)
        else:
            st.warning("feature_importance.png not found in data/processed/")

    st.markdown("---")

    # ── Row 3: churn probability distribution ─────────────────────────────────
    st.subheader("Churn Probability Distribution by Segment")
    fig_dist = px.histogram(
        df_churn, x="churn_probability",
        color="Segment", color_discrete_map=SEG_COLORS,
        nbins=50, barmode="overlay", opacity=0.75,
        labels={"churn_probability": "Churn Probability", "count": "Customers"},
    )
    fig_dist.add_vline(
        x=0.5, line_dash="dash", line_color="gray",
        annotation_text="Decision boundary (0.5)",
        annotation_position="top right",
    )
    fig_dist.update_layout(margin=dict(t=20, b=10))
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    # ── Row 4: Risk Tier table ────────────────────────────────────────────────
    st.subheader("Risk Tier Breakdown")

    def assign_risk_tier(p: float) -> str:
        if   p >= 0.8: return "Critical"
        elif p >= 0.6: return "High"
        elif p >= 0.4: return "Medium"
        else:          return "Low"

    risk_df = df_clv.copy()
    risk_df["Risk_Tier"]       = risk_df["churn_probability"].apply(assign_risk_tier)
    risk_df["Revenue_at_Risk"] = risk_df["CLV_basic"] - risk_df["CLV_adjusted"]

    TIER_ORDER = ["Critical", "High", "Medium", "Low"]
    risk_agg = (
        risk_df.groupby("Risk_Tier")
        .agg(
            Customers      =("CustomerID",      "count"),
            Avg_CLV        =("CLV_adjusted",     "mean"),
            Revenue_at_Risk=("Revenue_at_Risk",  "sum"),
        )
        .reindex(TIER_ORDER)
        .reset_index()
    )
    risk_agg["Pct"] = (
        risk_agg["Customers"] / risk_agg["Customers"].sum() * 100
    ).round(1)

    risk_display = pd.DataFrame({
        "Risk Tier":       risk_agg["Risk_Tier"],
        "Customers":       risk_agg["Customers"],
        "% of Total":      risk_agg["Pct"].astype(str) + "%",
        "Avg CLV":         risk_agg["Avg_CLV"].apply(lambda x: f"£{x:,.0f}"),
        "Revenue at Risk": risk_agg["Revenue_at_Risk"].apply(lambda x: f"£{x:,.0f}"),
    })

    TIER_ROW_STYLE = {
        "Critical": "background-color: #f9d5d5",
        "High":     "background-color: #fce5c0",
        "Medium":   "background-color: #fef9c3",
        "Low":      "background-color: #d4edda",
    }

    def style_risk_row(row):
        return [TIER_ROW_STYLE.get(row["Risk Tier"], "")] * len(row)

    st.dataframe(
        risk_display.style.apply(style_risk_row, axis=1),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")

    # ── Row 5: Customer Lookup ────────────────────────────────────────────────
    st.subheader("🔍 Customer Risk Lookup")
    st.markdown("Enter a CustomerID to retrieve their full risk and CLV profile.")

    input_col, btn_col = st.columns([3, 1])
    with input_col:
        lookup_id = st.number_input(
            "CustomerID",
            min_value=0, max_value=999999, step=1, value=0,
            format="%d", label_visibility="collapsed",
        )
    with btn_col:
        do_lookup = st.button("🔍 Look Up", use_container_width=True)

    if do_lookup:
        if lookup_id == 0:
            st.warning("Please enter a valid CustomerID (e.g. 12347).")
        else:
            match = df_clv[df_clv["CustomerID"] == int(lookup_id)]
            if match.empty:
                st.error(
                    f"CustomerID **{int(lookup_id)}** was not found in the dataset. "
                    "Please check the ID and try again."
                )
            else:
                r = match.iloc[0]
                worth  = bool(r["Worth_Retaining"])
                action = (
                    "Loyalty rewards & VIP early access"
                    if r["Segment"] == "Champions"
                    else "Win-back campaign — personalised discount"
                )
                st.success(f"Found customer **{int(lookup_id)}**")
                la, lb, lc, ld = st.columns(4)
                la.metric("Segment",   r["Segment"])
                lb.metric("Recency",   f"{int(r['Recency'])} days")
                lc.metric("Frequency", f"{int(r['Frequency'])} orders")
                ld.metric("Monetary",  f"£{r['Monetary']:,.0f}")

                le, lf, lg, lh = st.columns(4)
                le.metric("Churn Probability", f"{r['churn_probability']:.1%}")
                lf.metric("CLV (Adjusted)",    f"£{r['CLV_adjusted']:,.0f}")
                lg.metric("CLV Tier",          r["CLV_tier"])
                lh.metric("Worth Retaining",   "Yes ✅" if worth else "No ❌")
                st.info(f"**Recommended Action:** {action}")

    st.caption(
        "Model: XGBoost (n_estimators=200, max_depth=4, learning_rate=0.05). "
        "MLflow run: 5716150ad84b41768d72d88bc0c3d282. "
        "Feature importance: Frequency 53.3% · Monetary 26.8% · Recency 19.9%."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Customer Lifetime Value
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Customer Lifetime Value":
    st.title("💰 Customer Lifetime Value")
    st.markdown(
        "3-year CLV projection.  "
        "`CLV_basic = Monetary × 1.5`  ·  "
        "`CLV_adjusted = CLV_basic × (1 − churn_probability)`"
    )

    df      = load_clv()
    seg_sum = load_segment_summary()

    # ── Row 1: KPIs ───────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric(
        "Total Projected CLV",
        f"£{df['CLV_basic'].sum():,.0f}",
        "3-year zero-churn baseline",
    )
    k2.metric(
        "Total Risk-Adjusted CLV",
        f"£{df['CLV_adjusted'].sum():,.0f}",
        "Expected value after churn discount",
    )
    k3.metric(
        "Revenue at Risk",
        f"£{(df['CLV_basic'] - df['CLV_adjusted']).sum():,.0f}",
        delta="-12.9% of potential", delta_color="inverse",
    )

    st.markdown("---")

    # ── Row 2: CLV distribution  +  tier pie ──────────────────────────────────
    r2l, r2r = st.columns(2)

    with r2l:
        st.subheader("CLV Distribution  (capped at 99th percentile)")
        p99     = float(df["CLV_basic"].quantile(0.99))
        df_clip = df[df["CLV_basic"] <= p99]
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df_clip["CLV_basic"],    name="CLV Basic",
            nbinsx=50, marker_color=BLUE,      opacity=0.7,
        ))
        fig_dist.add_trace(go.Histogram(
            x=df_clip["CLV_adjusted"], name="CLV Adjusted",
            nbinsx=50, marker_color="#e74c3c", opacity=0.65,
        ))
        fig_dist.update_layout(
            barmode="overlay",
            xaxis_title="CLV (£)", yaxis_title="Customers",
            margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with r2r:
        st.subheader("CLV Tier Distribution")
        tier_cnts = df["CLV_tier"].value_counts().reset_index()
        tier_cnts.columns = ["Tier", "Customers"]
        fig_tier_pie = px.pie(
            tier_cnts, names="Tier", values="Customers",
            color="Tier", color_discrete_map=TIER_COLORS,
            hole=0.4,
        )
        fig_tier_pie.update_traces(textinfo="label+percent+value")
        fig_tier_pie.update_layout(
            margin=dict(t=10, b=10),
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_tier_pie, use_container_width=True)

    st.markdown("---")

    # ── Row 3: CLV tier summary table ─────────────────────────────────────────
    st.subheader("CLV Tier Summary")
    CLV_TIER_ORDER = ["Platinum", "Gold", "Silver", "Bronze"]
    total_adj      = float(df["CLV_adjusted"].sum())
    tier_rows      = []
    for tier in CLV_TIER_ORDER:
        sub = df[df["CLV_tier"] == tier]
        tier_rows.append({
            "Tier":         tier,
            "Customers":    len(sub),
            "%":            f"{len(sub)/len(df)*100:.1f}%",
            "Mean CLV":     f"£{sub['CLV_adjusted'].mean():,.0f}",
            "Total CLV":    f"£{sub['CLV_adjusted'].sum():,.0f}",
            "% of Revenue": f"{sub['CLV_adjusted'].sum()/total_adj*100:.1f}%",
        })
    tier_df = pd.DataFrame(tier_rows)

    TIER_ROW_BG = {
        "Platinum": "background-color: #FFF9C4",
        "Gold":     "background-color: #F5F5F5",
        "Silver":   "background-color: #FFF3E0",
        "Bronze":   "background-color: #EEEEEE",
    }

    def style_tier_row(row):
        return [TIER_ROW_BG.get(row["Tier"], "")] * len(row)

    st.dataframe(
        tier_df.style.apply(style_tier_row, axis=1),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")

    # ── Row 4: CLV by segment  +  Revenue at Risk ─────────────────────────────
    r4l, r4r = st.columns(2)

    with r4l:
        st.subheader("Total CLV by Segment")
        fig_seg = px.bar(
            seg_sum, x="Segment", y="Total_CLV_adj",
            color="Segment", color_discrete_map=SEG_COLORS,
            labels={"Total_CLV_adj": "Total Risk-Adjusted CLV (£)", "Segment": ""},
        )
        fig_seg.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_seg, use_container_width=True)

    with r4r:
        st.subheader("Revenue at Risk by Segment")
        fig_risk = px.bar(
            seg_sum, x="Segment", y="Revenue_at_Risk",
            color="Segment", color_discrete_map=SEG_COLORS,
            labels={"Revenue_at_Risk": "Revenue at Risk (£)", "Segment": ""},
        )
        fig_risk.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("---")

    # ── Row 5: Retention ROI ──────────────────────────────────────────────────
    st.subheader("Retention ROI Analysis")
    worth_df = (
        df.groupby(["Segment", "Worth_Retaining"])
        .size()
        .reset_index(name="Customers")
    )
    worth_df["Category"] = worth_df["Worth_Retaining"].map(
        {True: "Worth Retaining", False: "Not Worth Retaining"}
    )
    fig_worth = px.bar(
        worth_df,
        x="Segment", y="Customers", color="Category",
        color_discrete_map={
            "Worth Retaining":     "#2ecc71",
            "Not Worth Retaining": "#e74c3c",
        },
        barmode="group",
        labels={"Customers": "Number of Customers", "Segment": ""},
    )
    fig_worth.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig_worth, use_container_width=True)

    st.success(
        "💡 Investing **£10 per customer** in retention for **3,860 high-ROI customers** "
        "yields an expected return of **£3,089,821** — an **80x return on investment**."
    )

    st.caption(
        "CLV formula: CLV_basic = Monetary × 1.5 (3-yr horizon / 2-yr dataset).  "
        "CLV_adjusted = CLV_basic × (1 − churn_probability).  "
        "Tier boundaries: Platinum ≥ 90th pct · Gold 70–90th · Silver 40–70th · Bronze < 40th."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Retention Strategy
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Retention Strategy":
    st.title("🎯 Retention Strategy")
    st.markdown(
        "### Translating model outputs into business decisions\n"
        "Segment-specific playbooks informed by churn probability, CLV tier, and retention ROI."
    )

    # ── Row 2: Strategy matrix ─────────────────────────────────────────────────
    st.subheader("Strategy Matrix")
    strategy_df = pd.DataFrame([
        ["Champions",        "Low",      "Platinum", "Loyalty rewards, early access",         "Maintain"],
        ["At-Risk High CLV", "Critical", "Gold",     "Personal outreach, 20% discount",       "Highest"],
        ["At-Risk Low CLV",  "Critical", "Bronze",   "Automated email, assess ROI",            "Low"],
        ["New Customers",    "Medium",   "Silver",   "Onboarding sequence, education",         "Medium"],
    ], columns=["Segment", "Risk", "CLV Tier", "Action", "Budget Priority"])

    STRATEGY_BG = {
        "Low":      "background-color: #d4edda",
        "Critical": "background-color: #f9d5d5",
        "Medium":   "background-color: #fef9c3",
    }

    def style_strategy_row(row):
        return [STRATEGY_BG.get(row["Risk"], "")] * len(row)

    st.dataframe(
        strategy_df.style.apply(style_strategy_row, axis=1),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")

    # ── Row 3: Budget pie  +  Expected outcomes ───────────────────────────────
    r3l, r3r = st.columns(2)

    with r3l:
        st.subheader("£50,000 Retention Budget Allocation")
        budget_df = pd.DataFrame({
            "Category": [
                "At-Risk High CLV",
                "Champions Maintenance",
                "New Customer Nurture",
                "At-Risk Low CLV",
            ],
            "Allocation": [60, 20, 15, 5],
        })
        fig_budget = px.pie(
            budget_df, names="Category", values="Allocation",
            color="Category",
            color_discrete_map={
                "At-Risk High CLV":      "#e74c3c",
                "Champions Maintenance": "#2ecc71",
                "New Customer Nurture":  BLUE,
                "At-Risk Low CLV":       "#95a5a6",
            },
            hole=0.4,
        )
        fig_budget.update_traces(textinfo="label+percent")
        fig_budget.update_layout(
            margin=dict(t=10, b=10),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_budget, use_container_width=True)

    with r3r:
        st.subheader("Expected Outcomes")
        st.metric("Customers Targeted",        "3,860")
        st.metric("Budget Required",            "£38,600")
        st.metric("Expected Revenue Recovered", "£3,089,821")
        st.metric("ROI",                        "80x",
                  delta="per £10 invested per customer")

    st.markdown("---")

    # ── Row 4: Model limitation ────────────────────────────────────────────────
    st.warning(
        "⚠️ **Model Limitation:** The churn label in this project is derived from "
        "RFM-based segmentation rather than actual subscription cancellation events. "
        "In a production environment, this would be replaced with verified churn events "
        "(e.g. account closure, 90-day inactivity) to eliminate label leakage between "
        "features and target."
    )

    # ── Row 5: Methodology expander ────────────────────────────────────────────
    with st.expander("📋 Methodology Notes"):
        st.markdown(
            "| Parameter | Detail |\n"
            "|---|---|\n"
            "| **Dataset** | UCI Online Retail II — 1,067,371 raw transactions |\n"
            "| **Data cleaning** | 287,946 rows removed (27%); cancellations, nulls, negative quantities |\n"
            "| **RFM window** | 2009-12-01 to 2011-12-09 (reference date: 2011-12-10) |\n"
            "| **Segmentation** | K-Means, k=2, silhouette score = 0.439 |\n"
            "| **Churn model** | XGBoost (n_estimators=200, max_depth=4, lr=0.05), 80/20 split, stratified |\n"
            "| **CLV horizon** | 3-year projection; cost_per_customer = £10 |\n"
            "| **MLflow run ID** | `5716150ad84b41768d72d88bc0c3d282` |\n"
            "| **Label note** | Churn label derived from RFM segment, not verified cancellation events |"
        )

    st.caption(
        "Phase 5 dashboard built with Streamlit & Plotly. "
        "All data from UCI Online Retail II (Dec 2009 – Dec 2011). "
        "Project: github.com/vishaalsai/customer-retention-analysis"
    )
