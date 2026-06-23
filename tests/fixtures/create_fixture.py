"""
Creates a small synthetic dataset with the same schema as the real
PhysioNet 2019 data. Used in CI/CD to train a minimal test model
without requiring the full 540MB feature dataset.
"""
import sys
import os
import pandas as pd
import numpy as np

# Add src/ to path BEFORE importing from it
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, src_path)

from feature_engineering import add_rolling_features, add_lab_features

np.random.seed(42)

N_PATIENTS = 50
HOURS_PER_PATIENT = 20
N_SEPSIS_PATIENTS = 8  # guarantee at least 8 sepsis patients

# Explicitly assign which patients get sepsis
sepsis_patient_ids = set(range(1, N_SEPSIS_PATIENTS + 1))

rows = []
for patient_id in range(1, N_PATIENTS + 1):
    is_sepsis_patient = patient_id in sepsis_patient_ids
    sepsis_onset_hour = np.random.randint(10, 18) if is_sepsis_patient else None

    for hour in range(HOURS_PER_PATIENT):
        row = {
            'Patient_ID': patient_id,
            'Hour': hour,
            'Age': np.random.randint(40, 85),
            'Gender': np.random.randint(0, 2),
            'HR': np.random.normal(100 if is_sepsis_patient else 80, 10),
            'O2Sat': np.random.normal(94 if is_sepsis_patient else 98, 2),
            'Temp': np.random.normal(38.5 if is_sepsis_patient else 37.0, 0.5),
            'SBP': np.random.normal(95 if is_sepsis_patient else 120, 15),
            'MAP': np.random.normal(65 if is_sepsis_patient else 85, 10),
            'DBP': np.random.normal(55 if is_sepsis_patient else 75, 10),
            'Resp': np.random.normal(22 if is_sepsis_patient else 16, 3),
            'Lactate': np.random.normal(3.5, 1.0) if np.random.random() < 0.1 else np.nan,
            'Creatinine': np.random.normal(1.8, 0.5) if np.random.random() < 0.1 else np.nan,
            'WBC': np.random.normal(14, 3) if np.random.random() < 0.1 else np.nan,
            'BUN': np.random.normal(25, 8) if np.random.random() < 0.1 else np.nan,
            'Platelets': np.random.normal(180, 50) if np.random.random() < 0.1 else np.nan,
            'Bilirubin_total': np.random.normal(1.5, 0.5) if np.random.random() < 0.05 else np.nan,
            'Glucose': np.random.normal(140, 30) if np.random.random() < 0.15 else np.nan,
            'SepsisLabel': 1 if (sepsis_onset_hour and hour >= sepsis_onset_hour) else 0,
        }
        rows.append(row)

df = pd.DataFrame(rows)

vitals = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
labs = ['Lactate', 'Creatinine', 'WBC', 'BUN', 'Platelets', 'Bilirubin_total', 'Glucose']

df = add_rolling_features(df, vitals, window=6)
df = add_lab_features(df, labs)

output_path = os.path.join(os.path.dirname(__file__), 'sample_data.csv')
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows to {output_path}")
print(f"Columns: {len(df.columns)}")
print(f"Sepsis rate: {df['SepsisLabel'].mean()*100:.1f}%")
print(f"Sepsis hours: {df['SepsisLabel'].sum()}")
