"""
Heart Disease Dataset — Interactive Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_MAP = {"No Disease": "#2E86AB", "Disease": "#E63946"}

# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
@st.cache_data
def load_raw(path="heart_disease.csv"):
    return pd.read_csv(path)


@st.cache_data
def load_data(path="heart_disease.csv"):
    df = pd.read_csv(path)

    # Human-readable mappings for categorical codes
    sex_map = {1: "Male", 0: "Female"}
    cp_map = {
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-anginal Pain",
        4: "Asymptomatic",
    }
    fbs_map = {1: "> 120 mg/dl", 0: "<= 120 mg/dl"}
    restecg_map = {
        0: "Normal",
        1: "ST-T Abnormality",
        2: "LV Hypertrophy",
    }
    exang_map = {1: "Yes", 0: "No"}
    slope_map = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
    thal_map = {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}
    target_map = {0: "No Disease", 1: "Disease"}

    df["sex_label"] = df["sex"].map(sex_map)
    df["cp_label"] = df["cp"].map(cp_map)
    df["fbs_label"] = df["fbs"].map(fbs_map)
    df["restecg_label"] = df["restecg"].map(restecg_map)
    df["exang_label"] = df["exang"].map(exang_map)
    df["slope_label"] = df["slope"].map(slope_map)
    df["thal_label"] = df["thal"].map(thal_map)
    df["target_label"] = df["target_binary"].map(target_map)

    age_bins = [0, 40, 50, 60, 70, 100]
    age_labels = ["<40", "40-49", "50-59", "60-69", "70+"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)

    return df


raw_df = load_raw()
df = load_data()

COLUMN_INFO = [
    ("age", "Numeric", "Patient age in years"),
    ("sex", "Binary", "1 = Male, 0 = Female"),
    ("cp", "Categorical (1-4)", "Chest pain type: 1=Typical Angina, 2=Atypical Angina, 3=Non-anginal Pain, 4=Asymptomatic"),
    ("trestbps", "Numeric", "Resting blood pressure (mm Hg) on admission"),
    ("chol", "Numeric", "Serum cholesterol (mg/dl)"),
    ("fbs", "Binary", "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)"),
    ("restecg", "Categorical (0-2)", "Resting ECG results: 0=Normal, 1=ST-T abnormality, 2=LV hypertrophy"),
    ("thalach", "Numeric", "Maximum heart rate achieved during exercise test"),
    ("exang", "Binary", "Exercise-induced angina (1 = yes, 0 = no)"),
    ("oldpeak", "Numeric", "ST depression induced by exercise relative to rest"),
    ("slope", "Categorical (1-3)", "Slope of peak exercise ST segment: 1=Upsloping, 2=Flat, 3=Downsloping"),
    ("ca", "Numeric (0-3)", "Number of major vessels colored by fluoroscopy"),
    ("thal", "Categorical", "Thalassemia: 3=Normal, 6=Fixed defect, 7=Reversible defect"),
    ("target_binary", "Binary (target)", "0 = No heart disease, 1 = Heart disease present"),
]

# ------------------------------------------------------------------
# SIDEBAR — FILTERS (apply to Home & Risk Factor Analysis tabs)
# ------------------------------------------------------------------
st.sidebar.title("❤️ Filters")
st.sidebar.markdown("Filters apply to the **Home** and **Risk Factor Analysis** tabs.")

age_range = st.sidebar.slider(
    "Age range",
    int(df["age"].min()),
    int(df["age"].max()),
    (int(df["age"].min()), int(df["age"].max())),
)

sex_filter = st.sidebar.multiselect(
    "Sex", options=sorted(df["sex_label"].unique()), default=sorted(df["sex_label"].unique())
)

cp_filter = st.sidebar.multiselect(
    "Chest Pain Type",
    options=sorted(df["cp_label"].unique()),
    default=sorted(df["cp_label"].unique()),
)

target_filter = st.sidebar.multiselect(
    "Diagnosis",
    options=sorted(df["target_label"].unique()),
    default=sorted(df["target_label"].unique()),
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dataset: Cleveland Heart Disease dataset (UCI). "
    f"{df.shape[0]} patients, {df.shape[1] - 8} clinical features."
)

filtered_df = df[
    (df["age"] >= age_range[0])
    & (df["age"] <= age_range[1])
    & (df["sex_label"].isin(sex_filter))
    & (df["cp_label"].isin(cp_filter))
    & (df["target_label"].isin(target_filter))
]

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.title("❤️ Heart Disease Analysis Dashboard")
st.markdown(
    "Interactive exploration of patient clinical data and its relationship "
    "to heart disease diagnosis."
)

tab_home, tab_about, tab_processing, tab_risk = st.tabs(
    ["🏠 Home", "📋 About Dataset", "🔧 Data Processing & Analysis", "⚠️ Risk Factor Analysis"]
)

# ====================================================================
# TAB 1 — HOME
# ====================================================================
with tab_home:
    if filtered_df.empty:
        st.warning("No records match the current filter selection. Please broaden your filters.")
    else:
        # KPI ROW
        col1, col2, col3, col4, col5 = st.columns(5)

        total_patients = len(filtered_df)
        disease_count = int((filtered_df["target_binary"] == 1).sum())
        disease_rate = disease_count / total_patients * 100
        avg_age = filtered_df["age"].mean()
        avg_chol = filtered_df["chol"].mean()
        avg_thalach = filtered_df["thalach"].mean()

        col1.metric("Total Patients", f"{total_patients:,}")
        col2.metric("Disease Cases", f"{disease_count:,}", f"{disease_rate:.1f}% of selection")
        col3.metric("Avg. Age", f"{avg_age:.1f} yrs")
        col4.metric("Avg. Cholesterol", f"{avg_chol:.0f} mg/dl")
        col5.metric("Avg. Max Heart Rate", f"{avg_thalach:.0f} bpm")

        st.markdown("---")

        # ROW 1 — Diagnosis distribution & Age distribution
        row1_col1, row1_col2 = st.columns([1, 2])

        with row1_col1:
            st.subheader("Diagnosis Breakdown")
            diag_counts = filtered_df["target_label"].value_counts().reset_index()
            diag_counts.columns = ["Diagnosis", "Count"]
            fig_pie = px.pie(
                diag_counts,
                names="Diagnosis",
                values="Count",
                color="Diagnosis",
                color_discrete_map=COLOR_MAP,
                hole=0.45,
            )
            fig_pie.update_traces(textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        with row1_col2:
            st.subheader("Age Distribution by Diagnosis")
            fig_age = px.histogram(
                filtered_df,
                x="age",
                color="target_label",
                color_discrete_map=COLOR_MAP,
                barmode="overlay",
                nbins=25,
                opacity=0.7,
                labels={"age": "Age", "target_label": "Diagnosis"},
            )
            fig_age.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
            st.plotly_chart(fig_age, use_container_width=True)

        # ROW 2 — Max heart rate scatter & Sex comparison
        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            st.subheader("Max Heart Rate vs. Age")
            fig_scatter = px.scatter(
                filtered_df,
                x="age",
                y="thalach",
                color="target_label",
                color_discrete_map=COLOR_MAP,
                size="chol",
                hover_data=["cp_label", "sex_label", "chol", "trestbps"],
                labels={"age": "Age", "thalach": "Max Heart Rate", "target_label": "Diagnosis"},
            )
            fig_scatter.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
            st.plotly_chart(fig_scatter, use_container_width=True)

        with row2_col2:
            st.subheader("Disease Rate by Sex & Age Group")
            grp = (
                filtered_df.groupby(["age_group", "sex_label"], observed=True)["target_binary"]
                .mean()
                .mul(100)
                .reset_index()
            )
            fig_sex = px.bar(
                grp,
                x="age_group",
                y="target_binary",
                color="sex_label",
                barmode="group",
                labels={"age_group": "Age Group", "target_binary": "Disease Rate (%)", "sex_label": "Sex"},
            )
            fig_sex.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
            st.plotly_chart(fig_sex, use_container_width=True)

        # ROW 3 — Fasting Blood Sugar & Resting ECG
        row3_col1, row3_col2 = st.columns(2)

        with row3_col1:
            st.subheader("Disease Rate by Fasting Blood Sugar")
            fbs_group = (
                filtered_df.groupby("fbs_label")["target_binary"]
                .mean()
                .mul(100)
                .reset_index()
                .rename(columns={"target_binary": "disease_rate"})
            )
            fig_fbs = px.bar(
                fbs_group,
                x="fbs_label",
                y="disease_rate",
                color="fbs_label",
                labels={"fbs_label": "Fasting Blood Sugar", "disease_rate": "Disease Rate (%)"},
                text_auto=".1f",
            )
            fig_fbs.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
            st.plotly_chart(fig_fbs, use_container_width=True)

        with row3_col2:
            st.subheader("Disease Rate by Resting ECG Result")
            ecg_group = (
                filtered_df.groupby("restecg_label")["target_binary"]
                .mean()
                .mul(100)
                .reset_index()
                .rename(columns={"target_binary": "disease_rate"})
                .sort_values("disease_rate", ascending=False)
            )
            fig_ecg = px.bar(
                ecg_group,
                x="restecg_label",
                y="disease_rate",
                color="restecg_label",
                labels={"restecg_label": "Resting ECG Result", "disease_rate": "Disease Rate (%)"},
                text_auto=".1f",
            )
            fig_ecg.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
            st.plotly_chart(fig_ecg, use_container_width=True)

        st.markdown("---")
        row4_col1, row4_col2 = st.columns(2)

        with row4_col1:
            st.subheader("Cholesterol vs. Resting Blood Pressure")
            fig_chol_bp = px.scatter(
                filtered_df,
                x="trestbps",
                y="chol",
                color="target_label",
                color_discrete_map=COLOR_MAP,
                trendline="ols",
                labels={"trestbps": "Resting BP (mm Hg)", "chol": "Cholesterol (mg/dl)", "target_label": "Diagnosis"},
            )
            fig_chol_bp.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
            st.plotly_chart(fig_chol_bp, use_container_width=True)

        with row4_col2:
            st.subheader("Avg. ST Depression (oldpeak) by Age Group")
            oldpeak_trend = (
                filtered_df.groupby(["age_group", "target_label"], observed=True)["oldpeak"]
                .mean()
                .reset_index()
            )
            fig_oldpeak_trend = px.line(
                oldpeak_trend,
                x="age_group",
                y="oldpeak",
                color="target_label",
                color_discrete_map=COLOR_MAP,
                markers=True,
                labels={"age_group": "Age Group", "oldpeak": "Avg. ST Depression", "target_label": "Diagnosis"},
            )
            fig_oldpeak_trend.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
            st.plotly_chart(fig_oldpeak_trend, use_container_width=True)

        st.markdown("---")
        st.subheader("Filtered Data")

        show_table = st.checkbox("Show raw data table", value=False)
        if show_table:
            st.dataframe(
                filtered_df.drop(
                    columns=["sex_label", "cp_label", "fbs_label", "restecg_label", "exang_label",
                             "slope_label", "thal_label", "target_label", "age_group"]
                ),
                use_container_width=True,
            )

        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download filtered data as CSV",
            data=csv,
            file_name="heart_disease_filtered.csv",
            mime="text/csv",
        )

        st.caption(
            "Data source: Cleveland Heart Disease dataset (UCI Machine Learning Repository). "
            "target_binary: 0 = no disease, 1 = disease present."
        )

# ====================================================================
# TAB 2 — ABOUT DATASET
# ====================================================================
with tab_about:
    st.header("About the Dataset")

    st.markdown(
        """
This dashboard uses the **Cleveland Heart Disease dataset**, one of the most
widely studied datasets in the UCI Machine Learning Repository. It contains
clinical, demographic, and diagnostic-test measurements for patients
evaluated for heart disease, along with a binary label indicating whether
disease was ultimately diagnosed.
"""
    )

    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("Rows (Patients)", f"{raw_df.shape[0]:,}")
    ov2.metric("Columns", f"{raw_df.shape[1]}")
    ov3.metric("Missing Values", f"{int(raw_df.isna().sum().sum())}")
    ov4.metric("Duplicate Rows", f"{int(raw_df.duplicated().sum())}")

    st.markdown("### Column Dictionary")
    col_info_df = pd.DataFrame(COLUMN_INFO, columns=["Column", "Type", "Description"])
    st.dataframe(col_info_df, use_container_width=True, hide_index=True)

    st.markdown("### Sample Rows")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.markdown("### Target Variable")
    st.markdown(
        "`target_binary` is the diagnosis label: **0 = no heart disease**, "
        "**1 = heart disease present**. This is the outcome all other "
        "features in this dashboard are analyzed against."
    )

    target_counts = df["target_label"].value_counts().reset_index()
    target_counts.columns = ["Diagnosis", "Count"]
    fig_target = px.bar(
        target_counts,
        x="Diagnosis",
        y="Count",
        color="Diagnosis",
        color_discrete_map=COLOR_MAP,
        text_auto=True,
    )
    fig_target.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig_target, use_container_width=True)

    st.markdown("### Feature Distributions")
    st.markdown("A quick look at how patients are distributed across the main categorical fields.")

    dist_col1, dist_col2, dist_col3 = st.columns(3)

    with dist_col1:
        st.markdown("**Sex**")
        sex_counts = df["sex_label"].value_counts().reset_index()
        sex_counts.columns = ["Sex", "Count"]
        fig_sex_pie = px.pie(sex_counts, names="Sex", values="Count", hole=0.4)
        fig_sex_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True,
                                   legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_sex_pie, use_container_width=True)

    with dist_col2:
        st.markdown("**Chest Pain Type**")
        cp_counts = df["cp_label"].value_counts().reset_index()
        cp_counts.columns = ["Chest Pain Type", "Count"]
        fig_cp_pie = px.pie(cp_counts, names="Chest Pain Type", values="Count", hole=0.4)
        fig_cp_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True,
                                  legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig_cp_pie, use_container_width=True)

    with dist_col3:
        st.markdown("**Thalassemia**")
        thal_counts = df["thal_label"].value_counts().reset_index()
        thal_counts.columns = ["Thalassemia", "Count"]
        fig_thal_pie = px.pie(thal_counts, names="Thalassemia", values="Count", hole=0.4)
        fig_thal_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True,
                                    legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig_thal_pie, use_container_width=True)

    dist_col4, dist_col5 = st.columns(2)

    with dist_col4:
        st.markdown("**Resting ECG Result**")
        ecg_counts = df["restecg_label"].value_counts().reset_index()
        ecg_counts.columns = ["Resting ECG", "Count"]
        fig_ecg_bar = px.bar(ecg_counts, x="Resting ECG", y="Count", text_auto=True, color="Resting ECG")
        fig_ecg_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_ecg_bar, use_container_width=True)

    with dist_col5:
        st.markdown("**ST Segment Slope**")
        slope_counts = df["slope_label"].value_counts().reset_index()
        slope_counts.columns = ["Slope", "Count"]
        fig_slope_bar = px.bar(slope_counts, x="Slope", y="Count", text_auto=True, color="Slope")
        fig_slope_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_slope_bar, use_container_width=True)

    st.markdown("**Age Distribution (all patients)**")
    fig_age_all = px.histogram(df, x="age", nbins=25, color_discrete_sequence=["#2E86AB"])
    fig_age_all.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_age_all, use_container_width=True)

    dist_col6, dist_col7 = st.columns(2)

    with dist_col6:
        st.markdown("**Fasting Blood Sugar**")
        fbs_counts = df["fbs_label"].value_counts().reset_index()
        fbs_counts.columns = ["Fasting Blood Sugar", "Count"]
        fig_fbs_pie = px.pie(fbs_counts, names="Fasting Blood Sugar", values="Count", hole=0.4)
        fig_fbs_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True,
                                   legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_fbs_pie, use_container_width=True)

    with dist_col7:
        st.markdown("**Exercise-Induced Angina**")
        exang_counts = df["exang_label"].value_counts().reset_index()
        exang_counts.columns = ["Exercise-Induced Angina", "Count"]
        fig_exang_pie = px.pie(exang_counts, names="Exercise-Induced Angina", values="Count", hole=0.4)
        fig_exang_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True,
                                     legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_exang_pie, use_container_width=True)

    st.markdown("**Cholesterol & Max Heart Rate Distributions**")
    dist_col8, dist_col9 = st.columns(2)
    with dist_col8:
        fig_chol_all = px.histogram(df, x="chol", nbins=25, color_discrete_sequence=["#E63946"])
        fig_chol_all.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis_title="Cholesterol (mg/dl)")
        st.plotly_chart(fig_chol_all, use_container_width=True)
    with dist_col9:
        fig_thalach_all = px.histogram(df, x="thalach", nbins=25, color_discrete_sequence=["#F4A261"])
        fig_thalach_all.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis_title="Max Heart Rate")
        st.plotly_chart(fig_thalach_all, use_container_width=True)

    st.caption(
        "Source: UCI Machine Learning Repository — Heart Disease Data Set "
        "(Cleveland Clinic Foundation)."
    )

# ====================================================================
# TAB 3 — DATA PROCESSING & ANALYSIS
# ====================================================================
with tab_processing:
    st.header("Data Processing & Analysis")

    st.markdown("### 1. Data Quality Checks")
    dq1, dq2 = st.columns(2)

    with dq1:
        st.markdown("**Missing values per column**")
        missing = raw_df.isna().sum()
        missing = missing[missing.index]
        st.dataframe(
            missing.reset_index().rename(columns={"index": "Column", 0: "Missing Count"}),
            use_container_width=True,
            hide_index=True,
        )
        if missing.sum() == 0:
            st.success("No missing values found in any column.")

    with dq2:
        st.markdown("**Data types**")
        dtypes_df = raw_df.dtypes.astype(str).reset_index()
        dtypes_df.columns = ["Column", "Dtype"]
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    st.markdown("### 2. Processing Steps Applied")
    st.markdown(
        """
1. **Loaded** the raw CSV (`heart_disease.csv`) with pandas.
2. **Verified** there are no missing values or duplicate rows requiring imputation/removal.
3. **Decoded categorical codes** (e.g. `cp`, `restecg`, `slope`, `thal`, `sex`) into
   human-readable labels for charting, while keeping the original numeric codes for modeling.
4. **Binned age** into 5 groups (`<40`, `40-49`, `50-59`, `60-69`, `70+`) to support
   group-level comparisons.
5. **Cached** the loading/transform step with `st.cache_data` so the dashboard stays fast
   as filters change.
"""
    )

    st.markdown("### 3. Descriptive Statistics (Numeric Features)")
    numeric_cols_all = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    st.dataframe(df[numeric_cols_all].describe().T.round(2), use_container_width=True)

    st.markdown("### 3b. Skewness, Kurtosis & Outlier Counts (IQR method)")
    stats_rows = []
    for c in numeric_cols_all:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = int(((df[c] < lower) | (df[c] > upper)).sum())
        stats_rows.append(
            {
                "Feature": c,
                "Skewness": round(df[c].skew(), 2),
                "Kurtosis": round(df[c].kurtosis(), 2),
                "Outliers (IQR)": outliers,
                "% Outliers": round(outliers / len(df) * 100, 1),
            }
        )
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    st.markdown("### 3c. Numeric Feature Distributions")
    hist_col1, hist_col2, hist_col3 = st.columns(3)
    with hist_col1:
        fig_h1 = px.histogram(df, x="thalach", nbins=25, marginal="box", color_discrete_sequence=["#2E86AB"])
        fig_h1.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis_title="Max Heart Rate")
        st.plotly_chart(fig_h1, use_container_width=True)
    with hist_col2:
        fig_h2 = px.histogram(df, x="oldpeak", nbins=25, marginal="box", color_discrete_sequence=["#E63946"])
        fig_h2.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis_title="ST Depression (oldpeak)")
        st.plotly_chart(fig_h2, use_container_width=True)
    with hist_col3:
        fig_h3 = px.histogram(df, x="trestbps", nbins=25, marginal="box", color_discrete_sequence=["#F4A261"])
        fig_h3.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis_title="Resting BP")
        st.plotly_chart(fig_h3, use_container_width=True)

    st.markdown("### 4. Feature Correlation Heatmap")
    st.markdown(
        "Pearson correlation between numeric clinical features and the diagnosis target. "
        "Values closer to ±1 indicate a stronger linear relationship."
    )
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "target_binary"]
    corr = df[numeric_cols].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig_corr.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### 5. Distributions & Outliers")
    proc_col1, proc_col2 = st.columns(2)

    with proc_col1:
        st.subheader("Cholesterol by Diagnosis")
        fig_chol = px.box(
            df,
            x="target_label",
            y="chol",
            color="target_label",
            color_discrete_map=COLOR_MAP,
            points="outliers",
            labels={"target_label": "Diagnosis", "chol": "Cholesterol (mg/dl)"},
        )
        fig_chol.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_chol, use_container_width=True)

    with proc_col2:
        st.subheader("Resting Blood Pressure by Diagnosis")
        fig_bp = px.box(
            df,
            x="target_label",
            y="trestbps",
            color="target_label",
            color_discrete_map=COLOR_MAP,
            points="outliers",
            labels={"target_label": "Diagnosis", "trestbps": "Resting BP (mm Hg)"},
        )
        fig_bp.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_bp, use_container_width=True)

    st.markdown("### 6. Pairwise Feature Relationships")
    st.markdown(
        "Scatter matrix of key numeric features, colored by diagnosis. Useful for spotting "
        "clusters or separability between disease/no-disease patients."
    )
    scatter_dims = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig_matrix = px.scatter_matrix(
        df,
        dimensions=scatter_dims,
        color="target_label",
        color_discrete_map=COLOR_MAP,
        labels={d: d for d in scatter_dims},
    )
    fig_matrix.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="", height=700)
    fig_matrix.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=4, opacity=0.6))
    st.plotly_chart(fig_matrix, use_container_width=True)

    st.markdown("### 7. Categorical Feature vs. Diagnosis (Crosstab Heatmap)")
    cat_feature = st.selectbox(
        "Choose a categorical feature to cross-tabulate against diagnosis:",
        options=["cp_label", "sex_label", "restecg_label", "slope_label", "thal_label", "fbs_label", "exang_label"],
        format_func=lambda x: x.replace("_label", "").replace("_", " ").title(),
    )
    crosstab = pd.crosstab(df[cat_feature], df["target_label"], normalize="index").mul(100).round(1)
    fig_crosstab = px.imshow(
        crosstab,
        text_auto=".1f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        labels=dict(color="% of row"),
    )
    fig_crosstab.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_crosstab, use_container_width=True)

    st.caption(
        "Note: this section always uses the **full dataset** (unaffected by sidebar filters) "
        "so it reflects the true underlying data quality and distributions."
    )

# TAB 4 — RISK FACTOR ANALYSIS
with tab_risk:
    st.header("Risk Factor Analysis")
    st.markdown(
        "Which clinical and demographic factors are most associated with a heart "
        "disease diagnosis? Charts below use the current sidebar filters."
    )

    if filtered_df.empty:
        st.warning("No records match the current filter selection. Please broaden your filters.")
    else:
        # ROW 1 — Chest pain type & Exercise-induced angina
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            st.subheader("Disease Rate by Chest Pain Type")
            cp_group = (
                filtered_df.groupby("cp_label")["target_binary"]
                .mean()
                .mul(100)
                .reset_index()
                .rename(columns={"target_binary": "disease_rate"})
                .sort_values("disease_rate", ascending=False)
            )
            fig_cp = px.bar(
                cp_group,
                x="cp_label",
                y="disease_rate",
                color="disease_rate",
                color_continuous_scale="Reds",
                labels={"cp_label": "Chest Pain Type", "disease_rate": "Disease Rate (%)"},
                text_auto=".1f",
            )
            fig_cp.update_layout(margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
            st.plotly_chart(fig_cp, use_container_width=True)
            st.caption("Asymptomatic chest pain is classically the strongest predictor of disease in this dataset.")

        with r1c2:
            st.subheader("Exercise-Induced Angina vs. Diagnosis")
            exang_group = (
                filtered_df.groupby(["exang_label", "target_label"]).size().reset_index(name="count")
            )
            fig_exang = px.bar(
                exang_group,
                x="exang_label",
                y="count",
                color="target_label",
                color_discrete_map=COLOR_MAP,
                barmode="group",
                labels={"exang_label": "Exercise-Induced Angina", "count": "Patients", "target_label": "Diagnosis"},
            )
            fig_exang.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
            st.plotly_chart(fig_exang, use_container_width=True)

        # ROW 2 — ST depression & Number of major vessels
        r2c1, r2c2 = st.columns(2)

        with r2c1:
            st.subheader("ST Depression (oldpeak) by Diagnosis")
            fig_oldpeak = px.violin(
                filtered_df,
                x="target_label",
                y="oldpeak",
                color="target_label",
                color_discrete_map=COLOR_MAP,
                box=True,
                points="outliers",
                labels={"target_label": "Diagnosis", "oldpeak": "ST Depression (oldpeak)"},
            )
            fig_oldpeak.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
            st.plotly_chart(fig_oldpeak, use_container_width=True)

        with r2c2:
            st.subheader("Disease Rate by # Major Vessels (ca)")
            ca_group = (
                filtered_df.groupby("ca")["target_binary"]
                .mean()
                .mul(100)
                .reset_index()
                .rename(columns={"target_binary": "disease_rate"})
            )
            fig_ca = px.bar(
                ca_group,
                x="ca",
                y="disease_rate",
                color="disease_rate",
                color_continuous_scale="Reds",
                labels={"ca": "Major Vessels Colored (ca)", "disease_rate": "Disease Rate (%)"},
                text_auto=".1f",
            )
            fig_ca.update_layout(margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
            st.plotly_chart(fig_ca, use_container_width=True)

        # ROW 3 — Thalassemia & Slope
        r3c1, r3c2 = st.columns(2)

        with r3c1:
            st.subheader("Disease Rate by Thalassemia Result")
            thal_group = (
                filtered_df.dropna(subset=["thal_label"])
                .groupby("thal_label")["target_binary"]
                .mean()
                .mul(100)
                .reset_index()
                .rename(columns={"target_binary": "disease_rate"})
                .sort_values("disease_rate", ascending=False)
            )
            fig_thal = px.bar(
                thal_group,
                x="thal_label",
                y="disease_rate",
                color="disease_rate",
                color_continuous_scale="Reds",
                labels={"thal_label": "Thalassemia Result", "disease_rate": "Disease Rate (%)"},
                text_auto=".1f",
            )
            fig_thal.update_layout(margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
            st.plotly_chart(fig_thal, use_container_width=True)

        with r3c2:
            st.subheader("Disease Rate by ST Segment Slope")
            slope_group = (
                filtered_df.groupby("slope_label")["target_binary"]
                .mean()
                .mul(100)
                .reset_index()
                .rename(columns={"target_binary": "disease_rate"})
                .sort_values("disease_rate", ascending=False)
            )
            fig_slope = px.bar(
                slope_group,
                x="slope_label",
                y="disease_rate",
                color="disease_rate",
                color_continuous_scale="Reds",
                labels={"slope_label": "ST Segment Slope", "disease_rate": "Disease Rate (%)"},
                text_auto=".1f",
            )
            fig_slope.update_layout(margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
            st.plotly_chart(fig_slope, use_container_width=True)

        # ROW 4 — Radar profile & Sunburst breakdown
        r4c1, r4c2 = st.columns(2)

        with r4c1:
            st.subheader("Average Clinical Profile: Disease vs. No Disease")
            radar_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
            radar_data = filtered_df.groupby("target_label")[radar_features].mean()
            # Normalize each feature 0-1 across the two groups so they're comparable on one radar
            radar_norm = (radar_data - df[radar_features].min()) / (
                df[radar_features].max() - df[radar_features].min()
            )

            fig_radar = go.Figure()
            for label in radar_norm.index:
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=radar_norm.loc[label].values.tolist() + [radar_norm.loc[label].values[0]],
                        theta=radar_features + [radar_features[0]],
                        fill="toself",
                        name=label,
                        line_color=COLOR_MAP.get(label, "#888888"),
                    )
                )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                margin=dict(t=30, b=10, l=40, r=40),
                showlegend=True,
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption("Values normalized 0–1 across the full dataset range so features are comparable on one chart.")

        with r4c2:
            st.subheader("Chest Pain × Sex × Diagnosis Breakdown")
            fig_sunburst = px.sunburst(
                filtered_df,
                path=["cp_label", "sex_label", "target_label"],
                color="target_label",
                color_discrete_map={**COLOR_MAP, "(?)": "#cccccc"},
            )
            fig_sunburst.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_sunburst, use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Correlation with Diagnosis")
        st.markdown(
            "How strongly each numeric feature correlates with disease diagnosis "
            "(absolute Pearson correlation, computed on the full dataset)."
        )
        imp_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
        importance = df[imp_features + ["target_binary"]].corr()["target_binary"].drop("target_binary")
        importance_df = importance.reset_index()
        importance_df.columns = ["Feature", "Correlation"]
        importance_df["Abs Correlation"] = importance_df["Correlation"].abs()
        importance_df = importance_df.sort_values("Abs Correlation", ascending=True)
        fig_importance = px.bar(
            importance_df,
            x="Correlation",
            y="Feature",
            orientation="h",
            color="Correlation",
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
            text_auto=".2f",
        )
        fig_importance.update_layout(margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
        st.plotly_chart(fig_importance, use_container_width=True)

        st.subheader("Disease Rate Trend by Age Group")
        age_trend = (
            filtered_df.groupby("age_group", observed=True)["target_binary"].mean().mul(100).reset_index()
        )
        fig_age_trend = px.line(
            age_trend,
            x="age_group",
            y="target_binary",
            markers=True,
            labels={"age_group": "Age Group", "target_binary": "Disease Rate (%)"},
        )
        fig_age_trend.update_traces(line_color="#E63946")
        fig_age_trend.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_age_trend, use_container_width=True)

        st.markdown("---")
        st.markdown("### Combined Risk Factor Summary")
        st.markdown(
            """
Based on the current filter selection, disease rate tends to be higher among patients with:
- **Asymptomatic** chest pain type
- **Exercise-induced angina** present
- Higher **ST depression (oldpeak)** values
- A greater **number of major vessels** colored by fluoroscopy
- **Reversible defect** or **fixed defect** thalassemia results
- A **flat or downsloping** ST segment slope

These patterns are consistent with established cardiology risk indicators, though this
dataset is observational and does not establish causation.
"""
        )