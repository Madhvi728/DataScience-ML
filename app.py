import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Bank Marketing and Customer Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",)

# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------

COLUMN_MAP = {
    "V1": "age",
    "V2": "job",
    "V3": "marital",
    "V4": "education",
    "V5": "default",
    "V6": "balance",
    "V7": "housing",
    "V8": "loan",
    "V9": "contact",
    "V10": "day",
    "V11": "month",
    "V12": "duration",
    "V13": "campaign",
    "V14": "pdays",
    "V15": "previous",
    "V16": "poutcome",
    "Class": "subscribed_raw" }

MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]


@st.cache_data
def load_data(path="Bank_Marketing_Dataset.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_MAP)

    # Class: 1 = no, 2 = yes (matches the dataset's known ~11.7% subscription rate)
    df["subscribed"] = df["subscribed_raw"].map({1: "no", 2: "yes"})
    df["subscribed_flag"] = df["subscribed_raw"].map({1: 0, 2: 1})

    df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)

    # Age groups
    bins = [17, 25, 35, 45, 55, 65, 100]
    labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

    # Previously contacted flag
    df["was_contacted_before"] = np.where(df["pdays"] == -1, "Never", "Previously Contacted")

    return df

df = load_data()

# ----------------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------------

PRIMARY = "#1f4e8c"
ACCENT = "#f2a900"
PALETTE = px.colors.sequential.Blues_r
CATEGORICAL_PALETTE = px.colors.qualitative.Set2

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f7f9fc;
        border: 1px solid #e3e8ef;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #1f4e8c;
    }
    .block-container {
        padding-top: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True,)

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION + FILTERS
# ----------------------------------------------------------------------------

with st.sidebar:
    st.title("🏦 Bank Marketing")
    selected = option_menu(
        menu_title=None,
        options=["Overview", "Preprocessing", "Customer Profile", "Financial Health",
                 "Campaign Performance", "Subscription Drivers"],
        icons=["gear", "speedometer2", "people", "cash-coin",
               "megaphone", "graph-up-arrow"],
        default_index=0,)

    st.markdown("---")
    st.subheader("Filters")

    job_filter = st.multiselect("Job", sorted(df["job"].unique()), default=None)
    marital_filter = st.multiselect("Marital Status", sorted(df["marital"].unique()), default=None)
    education_filter = st.multiselect("Education", sorted(df["education"].unique()), default=None)
    age_range = st.slider("Age Range", int(df["age"].min()), int(df["age"].max()),
                           (int(df["age"].min()), int(df["age"].max())))
    outcome_filter = st.multiselect("Subscribed (Target)", ["yes", "no"], default=None)

    st.markdown("---")
    st.caption("Data source: Bank Marketing Dataset (UCI) · 45,211 records")

# Apply filters
fdf = df.copy()
if job_filter:
    fdf = fdf[fdf["job"].isin(job_filter)]
if marital_filter:
    fdf = fdf[fdf["marital"].isin(marital_filter)]
if education_filter:
    fdf = fdf[fdf["education"].isin(education_filter)]
if outcome_filter:
    fdf = fdf[fdf["subscribed"].isin(outcome_filter)]
fdf = fdf[(fdf["age"] >= age_range[0]) & (fdf["age"] <= age_range[1])]

if fdf.empty:
    st.warning("No records match the current filter selection. Please adjust filters.")
    st.stop()

# ----------------------------------------------------------------------------
# SHARED KPI HEADER
# ----------------------------------------------------------------------------

def kpi_header(data):
    total = len(data)
    subs = data["subscribed_flag"].sum()
    rate = subs / total * 100 if total else 0
    avg_balance = data["balance"].mean()
    avg_duration = data["duration"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Subscribed", f"{subs:,}")
    c3.metric("Subscription Rate", f"{rate:.1f}%")
    c4.metric("Avg. Balance", f"€{avg_balance:,.0f}")
    c5.metric("Avg. Call Duration", f"{avg_duration:,.0f}s")

# ----------------------------------------------------------------------------
# PAGE: OVERVIEW
# ----------------------------------------------------------------------------

if selected == "Overview":
    st.title("Overview")
    st.caption("A snapshot of the bank's term deposit marketing campaign")
    kpi_header(fdf)
    st.markdown("---")

    col1, col2 = st.columns([1.3, 1])

    with col1:
        by_month = (fdf.groupby("month", observed=True)["subscribed_flag"]
                    .agg(["count", "sum"]).reset_index())
        by_month["rate"] = by_month["sum"] / by_month["count"] * 100
        fig = go.Figure()
        fig.add_bar(x=by_month["month"], y=by_month["count"], name="Total Contacts",
                    marker_color="#c9d6e8")
        fig.add_scatter(x=by_month["month"], y=by_month["rate"], name="Subscription Rate (%)",
                         yaxis="y2", line=dict(color=ACCENT, width=3))
        fig.update_layout(
            title="Campaign Volume & Subscription Rate by Month",
            yaxis=dict(title="Contacts"),
            yaxis2=dict(title="Subscription Rate (%)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.15),
            height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sub_counts = fdf["subscribed"].value_counts().reset_index()
        sub_counts.columns = ["subscribed", "count"]
        fig = px.pie(sub_counts, names="subscribed", values="count", hole=0.55,
                     color="subscribed",
                     color_discrete_map={"yes": PRIMARY, "no": "#d9dfe8"},
                     title="Subscription Outcome")
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.histogram(fdf, x="age", nbins=40, color="subscribed",
                            barmode="overlay", opacity=0.7,
                            color_discrete_map={"yes": PRIMARY, "no": "#d9dfe8"},
                            title="Age Distribution by Subscription")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        contact_counts = fdf["contact"].value_counts().reset_index()
        contact_counts.columns = ["contact", "count"]
        fig = px.bar(contact_counts, x="contact", y="count", color="contact",
                     color_discrete_sequence=CATEGORICAL_PALETTE,
                     title="Contact Method Distribution")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE: PREPROCESSING / ABOUT
# ----------------------------------------------------------------------------

elif selected == "Preprocessing":
    st.title("Preprocessing & About")
    st.caption("How the raw file was cleaned and prepared for this dashboard")

    st.markdown("""
    This dataset is the **UCI Bank Marketing dataset**, shipped with generic column
    names (`V1`–`V16`, `Class`). Before anything was visualized, the following
    preprocessing steps were applied.
    """)

    st.markdown("---")

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.subheader("1. Column Renaming")
        st.caption("Generic labels were mapped back to their real-world meaning")
        rename_table = pd.DataFrame({
            "Original": ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
                         "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "Class"],
            "Renamed To": ["age", "job", "marital", "education", "default", "balance",
                           "housing", "loan", "contact", "day", "month", "duration",
                           "campaign", "pdays", "previous", "poutcome", "subscribed_raw"],
            "Description": [
                "Customer's age in years", "Type of job", "Marital status",
                "Highest education level", "Has credit in default?",
                "Average yearly balance (EUR)", "Has a housing loan?",
                "Has a personal loan?", "Contact communication type",
                "Last contact day of the month", "Last contact month of year",
                "Last contact duration (seconds)", "Number of contacts this campaign",
                "Days since last contact from a previous campaign (-1 = never)",
                "Number of contacts before this campaign",
                "Outcome of the previous marketing campaign",
                "Raw target: 1 = no, 2 = yes",
            ],
        })
        st.dataframe(rename_table, use_container_width=True, hide_index=True, height=460)

    with col2:
        st.subheader("2. Data Quality Checks")
        st.metric("Total Rows", f"{len(df):,}")
        st.metric("Total Columns", f"{df.shape[1] - 2}")  # excluding raw/derived target
        missing = df.isnull().sum().sum()
        st.metric("Missing Values Found", f"{missing:,}")

        st.subheader("3. Target Variable Encoding")
        st.markdown("""
        - `Class` (1 / 2) → decoded into `subscribed` (**"no" / "yes"**)
        - A binary helper column `subscribed_flag` (0 / 1) was added for
          aggregations, correlations, and rate calculations used across every
          tab in this dashboard.
        """)
        target_counts = df["subscribed"].value_counts().rename_axis("subscribed").reset_index(name="count")
        target_counts["share (%)"] = (target_counts["count"] / target_counts["count"].sum() * 100).round(2)
        st.dataframe(target_counts, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("4. Derived Features")
    st.markdown("""
    | Feature | Logic |
    |---|---|
    | `age_group` | `age` binned into 18-25, 26-35, 36-45, 46-55, 56-65, 65+ |
    | `was_contacted_before` | `"Never"` if `pdays == -1`, else `"Previously Contacted"` |
    | `month` | Cast to an **ordered categorical** (Jan → Dec) for correct chart sorting |
    """)

    st.markdown("---")
    st.subheader("5. Before / After Preview")
    tab1, tab2, tab3 = st.tabs(["Original (raw)", "Processed (used in dashboard)", "Column Data Types"])
    with tab1:
        st.dataframe(pd.read_csv("Bank_Marketing_Dataset.csv").head(10), use_container_width=True)
    with tab2:
        st.dataframe(
            df.drop(columns=["subscribed_raw"]).head(10), use_container_width=True)

    with tab3:
        dtype_df = df.drop(columns=["subscribed_raw"]).dtypes.astype(str).rename_axis("column").reset_index(name="dtype")
        st.dataframe(dtype_df, use_container_width=True, hide_index=True, height=460)

    st.markdown("---")
    st.subheader("Explore & Export")
    st.caption(f"Showing {len(fdf):,} of {len(df):,} records after your sidebar filters")
    with st.expander("View filtered dataset"):
        st.dataframe(fdf.drop(columns=["subscribed_raw"]), use_container_width=True, height=400)

    csv = fdf.drop(columns=["subscribed_raw"]).to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered Data as CSV", csv, "filtered_bank_data.csv", "text/csv")

# ----------------------------------------------------------------------------
# PAGE: CUSTOMER PROFILE
# ----------------------------------------------------------------------------

elif selected == "Customer Profile":
    st.title("Customer Profile")
    st.caption("Demographic composition of the customer base")
    kpi_header(fdf)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        job_rate = (fdf.groupby("job")["subscribed_flag"].agg(["count", "mean"])
                    .reset_index().sort_values("count", ascending=True))
        job_rate["mean"] *= 100
        fig = px.bar(job_rate, x="count", y="job", orientation="h",
                     color="mean", color_continuous_scale="Blues",
                     labels={"count": "Customers", "mean": "Sub. Rate (%)"},
                     title="Customers by Job Type (colored by subscription rate)")
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.sunburst(fdf, path=["marital", "education"], color="marital",
                           color_discrete_sequence=CATEGORICAL_PALETTE,
                           title="Marital Status → Education Breakdown")
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        age_group_rate = (fdf.groupby("age_group", observed=True)["subscribed_flag"]
                           .agg(["count", "mean"]).reset_index())
        age_group_rate["mean"] *= 100
        fig = px.bar(age_group_rate, x="age_group", y="mean", text_auto=".1f",
                     color_discrete_sequence=[PRIMARY],
                     labels={"mean": "Subscription Rate (%)", "age_group": "Age Group"},
                     title="Subscription Rate by Age Group")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        edu_marital = pd.crosstab(fdf["education"], fdf["marital"])
        fig = px.imshow(edu_marital, text_auto=True, color_continuous_scale="Blues",
                         title="Education vs. Marital Status (Heatmap)")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE: FINANCIAL HEALTH
# ----------------------------------------------------------------------------

elif selected == "Financial Health":
    st.title("Financial Health")
    st.caption("Balance, loans, and credit profile of customers")
    kpi_header(fdf)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(fdf, x="job", y="balance", color="job",
                     color_discrete_sequence=CATEGORICAL_PALETTE,
                     title="Balance Distribution by Job")
        fig.update_layout(height=450, showlegend=False, yaxis_range=[-2000, 10000])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        loan_summary = fdf.melt(value_vars=["housing", "loan", "default"],
                                 var_name="loan_type", value_name="status")
        loan_summary = loan_summary.groupby(["loan_type", "status"]).size().reset_index(name="count")
        fig = px.bar(loan_summary, x="loan_type", y="count", color="status", barmode="group",
                     color_discrete_map={"yes": ACCENT, "no": PRIMARY},
                     title="Housing Loan, Personal Loan & Credit Default")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.scatter(fdf.sample(min(3000, len(fdf)), random_state=1),
                          x="age", y="balance", color="subscribed",
                          color_discrete_map={"yes": PRIMARY, "no": "#d9dfe8"},
                          opacity=0.6, title="Age vs. Balance (sampled)")
        fig.update_layout(height=400, yaxis_range=[-2000, 20000])
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        default_rate = (fdf.groupby("default")["subscribed_flag"].agg(["count", "mean"])
                         .reset_index())
        default_rate["mean"] *= 100
        fig = px.bar(default_rate, x="default", y="mean", text_auto=".1f",
                     color="default", color_discrete_sequence=[PRIMARY, ACCENT],
                     labels={"mean": "Subscription Rate (%)"},
                     title="Subscription Rate: Credit Default Status")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE: CAMPAIGN PERFORMANCE
# ----------------------------------------------------------------------------

elif selected == "Campaign Performance":
    st.title("Campaign Performance")
    st.caption("How outreach effort translates into subscriptions")
    kpi_header(fdf)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        camp_rate = (fdf[fdf["campaign"] <= 15].groupby("campaign")["subscribed_flag"]
                     .agg(["count", "mean"]).reset_index())
        camp_rate["mean"] *= 100
        fig = px.line(camp_rate, x="campaign", y="mean", markers=True,
                      color_discrete_sequence=[PRIMARY],
                      labels={"mean": "Subscription Rate (%)", "campaign": "Contacts This Campaign"},
                      title="Subscription Rate vs. Number of Contacts (this campaign)")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(fdf.groupby("was_contacted_before")["subscribed_flag"]
                     .agg(["count", "mean"]).reset_index().assign(mean=lambda d: d["mean"] * 100),
                     x="was_contacted_before", y="mean", text_auto=".1f",
                     color="was_contacted_before", color_discrete_sequence=[PRIMARY, ACCENT],
                     labels={"mean": "Subscription Rate (%)", "was_contacted_before": "Prior Contact"},
                     title="Subscription Rate: New vs. Previously Contacted")
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        poutcome_rate = (fdf.groupby("poutcome")["subscribed_flag"].agg(["count", "mean"])
                          .reset_index())
        poutcome_rate["mean"] *= 100
        fig = px.bar(poutcome_rate, x="poutcome", y="mean", text_auto=".1f",
                     color="poutcome", color_discrete_sequence=CATEGORICAL_PALETTE,
                     labels={"mean": "Subscription Rate (%)"},
                     title="Subscription Rate by Previous Campaign Outcome")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram(fdf, x="duration", color="subscribed", nbins=50,
                            barmode="overlay", opacity=0.7,
                            color_discrete_map={"yes": PRIMARY, "no": "#d9dfe8"},
                            title="Call Duration Distribution by Outcome")
        fig.update_layout(height=400, xaxis_range=[0, 1500])
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE: SUBSCRIPTION DRIVERS
# ----------------------------------------------------------------------------

elif selected == "Subscription Drivers":
    st.title("Subscription Drivers")
    st.caption("What separates a 'yes' from a 'no'")
    kpi_header(fdf)
    st.markdown("---")

    numeric_cols = ["age", "balance", "day", "duration", "campaign", "pdays",
                     "previous", "subscribed_flag"]
    corr = fdf[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1,
                     title="Correlation Matrix (Numeric Features)")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.violin(fdf, x="subscribed", y="duration", color="subscribed", box=True,
                        color_discrete_map={"yes": PRIMARY, "no": "#d9dfe8"},
                        title="Call Duration: Subscribed vs. Not")
        fig.update_layout(height=420, yaxis_range=[0, 1500], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        month_rate = (fdf.groupby("month", observed=True)["subscribed_flag"]
                      .agg(["count", "mean"]).reset_index())
        month_rate["mean"] *= 100
        fig = px.bar(month_rate, x="month", y="mean", text_auto=".1f",
                     color="mean", color_continuous_scale="Blues",
                     labels={"mean": "Subscription Rate (%)"},
                     title="Subscription Rate by Month")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top Feature Combinations")
    top_combo = (fdf.groupby(["job", "education"])["subscribed_flag"]
                 .agg(["count", "mean"]).reset_index())
    top_combo = top_combo[top_combo["count"] >= 50].sort_values("mean", ascending=False).head(10)
    top_combo["mean"] = (top_combo["mean"] * 100).round(1)
    top_combo.columns = ["Job", "Education", "Customers", "Subscription Rate (%)"]
    st.dataframe(top_combo, use_container_width=True, hide_index=True)