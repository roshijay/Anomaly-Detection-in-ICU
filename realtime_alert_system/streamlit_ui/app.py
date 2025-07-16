import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="ICU Alert Triage System", layout="wide")

st.markdown("""
### Real-Time ICU Alert System

This dashboard simulates a real-time ICU monitoring system. Patient vitals are streamed, scored for risk, and triaged using clinician feedback. Alerts are ranked by severity to support early intervention and reduce alarm fatigue.
""")

# Load dataset
DATA_PATH = "mixed_focus.csv"
if not os.path.exists(DATA_PATH):
    st.error("mixed_focus.csv not found. Check the path.")
    st.stop()

df = pd.read_csv(DATA_PATH)
df.rename(columns=lambda x: x.strip().title(), inplace=True)
df["Pulse"] = df["Hr"].fillna(80)
df["SysBP"] = 110
df["Emergency"] = df["Mortality"].apply(lambda x: 1 if x == 1 else 0)

# --- Severity Scoring ---
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

st.sidebar.header("Clinician Feedback Settings")
feedback_importance = st.sidebar.slider(
    "Adjust alert sensitivity", 0.5, 2.0, 1.0, 0.1,
    help="Simulates how clinicians prioritize alerts. Higher = more sensitive."
)

df['SeverityScore'] = df.apply(lambda row: calculate_severity(row, feedback_importance), axis=1)
df['Diagnosis'] = df.apply(generate_diagnosis, axis=1)
anomalies = df[df['SeverityScore'] > 0].copy()

# --- UI Sections ---
st.subheader("Incoming Patient Vitals")
st.dataframe(df[['Patient_Id', 'Pulse', 'SysBP', 'Emergency']].head(10))

st.subheader("Detected Alerts (Ranked by Severity)")
st.caption("These are patients whose vitals triggered critical thresholds (e.g., Pulse > 130, SysBP < 90). Higher severity scores indicate greater risk.")
st.metric("Total Alerts", len(anomalies))
st.dataframe(anomalies[['Patient_Id', 'Pulse', 'SysBP', 'Emergency', 'SeverityScore', 'Diagnosis']].sort_values(by='SeverityScore', ascending=False).head(10))

st.subheader("Severity Score Distribution")
st.bar_chart(anomalies['SeverityScore'].value_counts().sort_index())

# --- Patient Similarity Recommender ---
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
