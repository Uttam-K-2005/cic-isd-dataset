"""Interactive exploratory dashboard for the cleaned CICIDS2017 dataset.

Run with: streamlit run dashboard.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATA_PATH = Path(__file__).with_name("cicids2017_cleaned.csv")
LABEL_COLUMN = "Label"
PREFERRED_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Flow Bytes/s",
    "Flow Packets/s", "Fwd Packet Length Mean", "Bwd Packet Length Mean",
    "Packet Length Mean", "Packet Length Std", "Average Packet Size",
]
PLOT_SAMPLE_SIZE = 75_000

st.set_page_config(page_title="CICIDS2017 Intrusion Detection Dashboard", page_icon="🛡️", layout="wide")


@st.cache_data(show_spinner="Loading the cleaned CICIDS2017 dataset…")
def load_data(path: str) -> pd.DataFrame:
    """Load once per cache lifetime; source data is never changed."""
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    if LABEL_COLUMN in frame.columns:
        frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype("string").str.strip()
    return frame


def available_features(frame: pd.DataFrame) -> list[str]:
    return [f for f in PREFERRED_FEATURES if f in frame and pd.api.types.is_numeric_dtype(frame[f])]


def traffic_categories(frame: pd.DataFrame) -> pd.Series:
    """Derived analysis-only category: all non-BENIGN labels are ATTACK."""
    return pd.Series(np.where(frame[LABEL_COLUMN].eq("BENIGN"), "BENIGN", "ATTACK"), index=frame.index, name="Traffic Category")


def sample_values(values: pd.Series, maximum: int = PLOT_SAMPLE_SIZE) -> pd.Series:
    values = values.dropna()
    return values.sample(maximum, random_state=42) if len(values) > maximum else values


def display_limits(values: pd.Series, limit_extremes: bool) -> tuple[float, float] | None:
    """Display-only percentile limits; values are not deleted or altered."""
    if not limit_extremes:
        return None
    values = values.dropna()
    low, high = values.quantile([0.01, 0.99])
    return None if pd.isna(low) or pd.isna(high) or low == high else (float(low), float(high))


def feature_label(feature: str) -> str:
    return "Flow Duration (microseconds)" if feature == "Flow Duration" else feature


def format_feature_axis(fig: go.Figure, values: pd.Series, feature: str, limit_extremes: bool) -> None:
    fig.update_xaxes(title=feature_label(feature))
    limits = display_limits(values, limit_extremes)
    if limits:
        fig.update_xaxes(range=list(limits))


def histogram(values: pd.Series, feature: str, title: str, limit_extremes: bool) -> go.Figure:
    fig = px.histogram(sample_values(values).to_frame(name=feature), x=feature, nbins=60, title=title,
                       color_discrete_sequence=["#2F6B9A"])
    fig.update_yaxes(title="Number of sampled flows")
    fig.update_layout(bargap=0.03)
    format_feature_axis(fig, values, feature, limit_extremes)
    return fig


def comparison_histogram(frame: pd.DataFrame, categories: pd.Series, feature: str, limit_extremes: bool) -> go.Figure:
    parts = []
    for group in ("BENIGN", "ATTACK"):
        parts.append(pd.DataFrame({feature: sample_values(frame.loc[categories.eq(group), feature]), "Traffic category": group}))
    plot_data = pd.concat(parts, ignore_index=True)
    fig = px.histogram(plot_data, x=feature, color="Traffic category", barmode="overlay", opacity=0.62, nbins=60,
                       title=f"{feature}: benign and attack traffic comparison",
                       color_discrete_map={"BENIGN": "#2E8B57", "ATTACK": "#C94C4C"})
    fig.update_yaxes(title="Number of sampled flows")
    format_feature_axis(fig, frame[feature], feature, limit_extremes)
    return fig


def statistics(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    results = []
    for feature in features:
        values = frame[feature]
        results.append({"Feature": feature, "Count": int(values.count()), "Mean": values.mean(),
                        "Median": values.median(), "Standard deviation": values.std(),
                        "Minimum": values.min(), "Maximum": values.max()})
    return pd.DataFrame(results).set_index("Feature")


st.title("Adaptive Machine Learning-Based Network Intrusion Detection System")
st.caption("CICIDS2017 exploratory dashboard — Stage 1 data analysis")
if not DATA_PATH.exists():
    st.error(f"Cleaned dataset not found: {DATA_PATH.name}")
    st.stop()

df = load_data(str(DATA_PATH))
if LABEL_COLUMN not in df.columns:
    st.error("The cleaned dataset does not contain the required 'Label' column.")
    st.stop()

features = available_features(df)
labels = df[LABEL_COLUMN].value_counts(dropna=False)
attack_labels = [label for label in labels.index.tolist() if label != "BENIGN"]
categories = traffic_categories(df)
total_flows = len(df)
benign_flows = int(categories.eq("BENIGN").sum())
attack_flows = int(categories.eq("ATTACK").sum())

with st.sidebar:
    st.header("Analysis controls")
    st.caption(f"Dataset: {DATA_PATH.name}")
    chosen_category = st.selectbox("Traffic category", ["All traffic", "BENIGN", "ATTACK"])
    chosen_attack = st.selectbox("Attack type", attack_labels or ["No attack labels available"])
    chosen_feature = st.selectbox("Feature", features or ["No requested numeric features available"])
    limit_extremes = st.checkbox("Display 1st–99th percentile axis range", value=True,
                                 help="This affects chart axes only; it does not remove observations.")
    st.divider()
    st.caption("Charts use deterministic samples of up to 75,000 flows per group.")

st.header("Dataset overview")
cols = st.columns(5)
cols[0].metric("Total network flows", f"{total_flows:,}")
cols[1].metric("Features", len(df.columns) - 1)
cols[2].metric("Benign flows", f"{benign_flows:,}")
cols[3].metric("Attack flows", f"{attack_flows:,}")
cols[4].metric("Attack classes", len(attack_labels))
st.caption("Preview of the cleaned source data. This dashboard does not modify it.")
st.dataframe(df.head(10), use_container_width=True, hide_index=True)

st.header("Traffic / attack distribution")
st.write("Flow counts expose CICIDS2017 class imbalance and the relative prevalence of each attack label.")
distribution = labels.rename_axis("Label").reset_index(name="Flows")
distribution["Percentage"] = distribution["Flows"] / total_flows * 100
distribution["Traffic group"] = np.where(distribution["Label"].eq("BENIGN"), "BENIGN", "Attack class")
bar = px.bar(distribution, x="Label", y="Flows", color="Traffic group", text="Percentage",
             title="Network flows by label (sorted by frequency)",
             color_discrete_map={"BENIGN": "#2E8B57", "Attack class": "#C94C4C"})
bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
bar.update_xaxes(title="CICIDS2017 label", categoryorder="array", categoryarray=distribution["Label"].tolist())
bar.update_yaxes(title="Number of flows")
bar.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
st.plotly_chart(bar, use_container_width=True)
st.dataframe(distribution[["Label", "Flows", "Percentage"]].style.format({"Flows": "{:,}", "Percentage": "{:.2f}%"}),
             use_container_width=True, hide_index=True)

st.header("Benign vs malicious traffic")
benign_pct = benign_flows / total_flows * 100 if total_flows else 0
attack_pct = attack_flows / total_flows * 100 if total_flows else 0
cols = st.columns(4)
cols[0].metric("Benign flow count", f"{benign_flows:,}")
cols[1].metric("Attack flow count", f"{attack_flows:,}")
cols[2].metric("Benign percentage", f"{benign_pct:.2f}%")
cols[3].metric("Attack percentage", f"{attack_pct:.2f}%")
donut = px.pie(names=["BENIGN", "ATTACK"], values=[benign_flows, attack_flows], hole=0.55,
               title="Traffic category composition", color=["BENIGN", "ATTACK"],
               color_discrete_map={"BENIGN": "#2E8B57", "ATTACK": "#C94C4C"})
donut.update_traces(textinfo="label+percent")
st.plotly_chart(donut, use_container_width=True)

st.header("RQ1: Network-flow characteristics")
st.write("Compare flow properties to investigate how benign and malicious traffic differ. Flow Duration is recorded in microseconds.")
if features:
    if chosen_category == "All traffic":
        st.plotly_chart(comparison_histogram(df, categories, chosen_feature, limit_extremes), use_container_width=True)
        box_values = sample_values(df[chosen_feature])
        plot_rows = pd.DataFrame({chosen_feature: box_values,
                                  "Traffic category": categories.loc[box_values.index]})
        box = px.box(plot_rows, x="Traffic category", y=chosen_feature, color="Traffic category", points=False,
                     title=f"{chosen_feature} by traffic category",
                     color_discrete_map={"BENIGN": "#2E8B57", "ATTACK": "#C94C4C"})
        box.update_xaxes(title="Traffic category")
        box.update_yaxes(title=feature_label(chosen_feature))
        st.plotly_chart(box, use_container_width=True)
    else:
        selected_values = df.loc[categories.eq(chosen_category), chosen_feature]
        st.plotly_chart(histogram(selected_values, chosen_feature, f"{chosen_feature} — {chosen_category} traffic", limit_extremes), use_container_width=True)
        st.caption(f"Histogram based on up to {PLOT_SAMPLE_SIZE:,} flows; percentile axis limits are display-only.")

st.header("Attack-type analysis")
st.write("Inspect an individual attack label against BENIGN traffic using the selected feature.")
if attack_labels and features:
    attack_values = df.loc[df[LABEL_COLUMN].eq(chosen_attack), chosen_feature]
    cols = st.columns(2)
    cols[0].metric(f"{chosen_attack} flows", f"{len(attack_values):,}")
    cols[1].metric("Percentage of dataset", f"{len(attack_values) / total_flows * 100:.3f}%")
    compare = pd.concat([
        pd.DataFrame({chosen_feature: sample_values(df.loc[df[LABEL_COLUMN].eq("BENIGN"), chosen_feature]), "Traffic": "BENIGN"}),
        pd.DataFrame({chosen_feature: sample_values(attack_values), "Traffic": chosen_attack}),
    ], ignore_index=True)
    attack_fig = px.histogram(compare, x=chosen_feature, color="Traffic", barmode="overlay", opacity=0.62, nbins=60,
                              title=f"{chosen_feature}: {chosen_attack} compared with BENIGN")
    attack_fig.update_yaxes(title="Number of sampled flows")
    format_feature_axis(attack_fig, pd.concat([df.loc[df[LABEL_COLUMN].eq("BENIGN"), chosen_feature], attack_values]), chosen_feature, limit_extremes)
    st.plotly_chart(attack_fig, use_container_width=True)
else:
    st.info("No individual attack labels or requested numeric features are available.")

st.header("Feature statistics")
st.write("Descriptive statistics use all cleaned flows, rather than the visualization samples.")
selected_stats = st.multiselect("Features for statistics", features, default=features[:4])
if selected_stats:
    st.dataframe(statistics(df, selected_stats).style.format("{:,.4g}"), use_container_width=True)
else:
    st.info("Select at least one numeric feature to view descriptive statistics.")

st.header("Data quality")
st.caption("Known post-preprocessing checks are displayed without rerunning costly full-data duplicate scans.")
quality = pd.DataFrame({
    "Measure": ["Dataset shape", "Missing values", "Infinite values", "Duplicate rows", "Number of classes"],
    "Value": [f"{df.shape[0]:,} rows × {df.shape[1]:,} columns", "0", "0", "0", str(df[LABEL_COLUMN].nunique())],
    "Note": ["Cleaned dataset", "Removed during preprocessing", "Converted and removed during preprocessing", "Exact duplicates removed during preprocessing", "Unique values in Label"],
})
st.dataframe(quality, use_container_width=True, hide_index=True)

st.header("Machine Learning Performance")
st.info("Machine-learning results will be displayed here after model training.")
st.caption("Reserved for RQ2: accuracy, precision, recall, F1-score, macro F1-score, confusion matrices, and model comparison.")

st.header("Adaptive Intrusion Detection")
st.info("Adaptive IDS results will be displayed here after the changing-distribution experiments.")
st.caption("Reserved for RQ3: performance across distributions, metrics over time or batches, baseline-versus-adaptive comparison, and concept-drift analysis.")
