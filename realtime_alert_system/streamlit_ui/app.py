import streamlit as st
import pandas as pd
import numpy as np
import os
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="ICU Alert Triage System", layout="wide")

st.markdown("""
### Real-Time ICU Alert System

This dashboard simulates a real-time ICU monitoring system. Patient vitals are streamed, scored for risk, and triaged using clinician feedback. Alerts are ranked by severity to support early intervention and reduce alarm fatigue.
""")

# Load base dataset
DATA_PATH = "mixed_focus.csv"
if not os.path.exists(DATA_PATH):
    st.error("mixed_focus.csv not found. Check the path.")
    st.stop()

df = pd.read_csv(DATA_PATH)
df.rename(columns=lambda x: x.strip().title(), inplace=True)
df["Pulse"] = df["Hr"].fillna(80)
df["SysBP"] = 110
df["Emergency"] = df["Mortality"].apply(lambda x: 1 if x == 1 else 0)

# --- Severity Scoring Functions ---
def calculate_severity(row, feedback_bias=1.0):
    score = 0
    if row['Pulse'] > 130:
        score += 2
    if row['SysBP'] < 90:
        score += 2
    if row['Emergency'] == 1:
        score += 3
    return score * feedback_bias

def generate_diagnosis(row):
    if row['Pulse'] > 130 and row['SysBP'] < 90:
        return "Likely shock or severe instability"
    elif row['Pulse'] > 130:
        return "Possible tachycardia"
    elif row['SysBP'] < 90:
        return "Hypotension suspected"
    elif row['Emergency'] == 1:
        return "Emergency admission — prioritize triage"
    else:
        return "Stable vitals"

# Sidebar - clinician feedback
st.sidebar.header("Clinician Feedback Settings")
feedback_importance = st.sidebar.slider(
    "Adjust alert sensitivity", 0.5, 2.0, 1.0, 0.1,
    help="Simulates how clinicians prioritize alerts. Higher = more sensitive."
)

# Apply scoring
df['SeverityScore'] = df.apply(lambda row: calculate_severity(row, feedback_importance), axis=1)
df['Diagnosis'] = df.apply(generate_diagnosis, axis=1)
anomalies = df[df['SeverityScore'] > 0].copy()

# --- Log alerts to CSV ---
alert_log_path = "data/alert_log.csv"
os.makedirs("data", exist_ok=True)

top_alerts = anomalies.sort_values(by='SeverityScore', ascending=False).reset_index(drop=True)

with open(alert_log_path, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["alert_id", "score", "rank", "is_critical", "feedback_applied"])
    writer.writeheader()
    for idx, row in top_alerts.iterrows():
        base_score = calculate_severity(row, feedback_bias=1.0)
        feedback_score = calculate_severity(row, feedback_bias=feedback_importance)

        # Log baseline (feedback_applied = False)
        writer.writerow({
            "alert_id": row["Patient_Id"],
            "score": base_score,
            "rank": idx + 1,
            "is_critical": int(base_score >= 4),
            "feedback_applied": False
    })

        # Log adjusted (feedback_applied = True)
        writer.writerow({
            "alert_id": row["Patient_Id"],
            "score": feedback_score,
            "rank": idx + 1,
            "is_critical": int(feedback_score >= 4),
            "feedback_applied": True
    })

# === Create Tabs ===
tab1, tab2, tab3 = st.tabs(["Vitals & Alerts", "Patient Clustering", "Validation & Impact"])

# --- Tab 1: Vitals & Alerts ---
with tab1:
    st.subheader("Incoming Patient Vitals")
    st.dataframe(df[['Patient_Id', 'Pulse', 'SysBP', 'Emergency']].head(10))

    st.subheader("Detected Alerts (Ranked by Severity)")
    st.metric("Total Alerts", len(anomalies))
    st.dataframe(anomalies[['Patient_Id', 'Pulse', 'SysBP', 'Emergency', 'SeverityScore', 'Diagnosis']]
                 .sort_values(by='SeverityScore', ascending=False).head(10))

    st.subheader("Severity Score Distribution")
    st.bar_chart(anomalies['SeverityScore'].value_counts().sort_index())

# --- Tab 2: Patient Clustering ---
with tab2:
    st.subheader("Patient Similarity Recommender")
    features = ['Pulse', 'SysBP']
    X = StandardScaler().fit_transform(df[features])
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    df['Cluster'] = kmeans.labels_

    cluster_labels = {
        0: "Stable Vitals",
        1: "Moderate Risk",
        2: "High Risk"
    }
    df['ClusterLabel'] = df['Cluster'].map(cluster_labels)

    selected_index = st.selectbox("Select a patient index to find similar profiles", anomalies.index.tolist())
    selected_cluster = df.loc[selected_index, 'Cluster']
    similar_patients = df[df['Cluster'] == selected_cluster]

    st.write(f"Patients similar to index {selected_index} — **{cluster_labels[selected_cluster]}** profile")
    st.dataframe(similar_patients[['Patient_Id', 'Pulse', 'SysBP', 'Emergency', 'SeverityScore', 'Diagnosis']].head(10))

# --- Tab 3: Validation & Impact ---
with tab3:
    st.subheader("Validation & Impact of Feedback Loop")

    if not os.path.exists(alert_log_path):
        st.warning("alert_log.csv not found.")
    else:
        df_feedback = pd.read_csv(alert_log_path)

        if 'feedback_applied' not in df_feedback.columns or 'is_critical' not in df_feedback.columns:
            st.warning("Log must include 'feedback_applied' and 'is_critical'.")
        else:
            k = st.slider("Select Top-k Alerts for Evaluation", min_value=1, max_value=10, value=5)

            df_before = df_feedback[df_feedback['feedback_applied'] == False]
            df_after = df_feedback[df_feedback['feedback_applied'] == True]

            def compute_metrics(df_subset, k):
                top_k = df_subset[df_subset['rank'] <= k]
                precision = top_k['is_critical'].mean()
                false_pos = 1 - precision
                return precision, false_pos

            precision_before, false_before = compute_metrics(df_before, k)
            precision_after, false_after = compute_metrics(df_after, k)

            col1, col2 = st.columns(2)
            col1.metric(f"Precision@{k} (Before)", f"{precision_before:.2f}")
            col1.metric(f"False Positives@{k} (Before)", f"{false_before:.2f}")
            col2.metric(f"Precision@{k} (After)", f"{precision_after:.2f}")
            col2.metric(f"False Positives@{k} (After)", f"{false_after:.2f}")

            st.subheader("Metric Comparison")
            metrics_df = pd.DataFrame({
                'Precision@k': [precision_before, precision_after],
                'False Positives@k': [false_before, false_after]
            }, index=['Before', 'After'])
            st.bar_chart(metrics_df)
