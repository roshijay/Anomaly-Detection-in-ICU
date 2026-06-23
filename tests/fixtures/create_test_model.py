"""
Trains a minimal XGBoost model on the synthetic fixture data.
Used in CI/CD so pytest has a real model to load without
requiring the full 540MB production dataset.
"""
import sys
import os
import pandas as pd
import joblib

# Add src/ to path BEFORE importing from it
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, src_path)

from train_model import prepare_data, train_xgboost

# Paths
fixture_path = os.path.join(os.path.dirname(__file__), 'sample_data.csv')
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
os.makedirs(models_dir, exist_ok=True)

print("Loading fixture data...")
df = pd.read_csv(fixture_path)
print(f"Shape: {df.shape}")

print("Preparing features...")
X, y = prepare_data(df)
print(f"Feature matrix: {X.shape}")
print(f"Sepsis rate: {y.mean()*100:.1f}%")

print("Training minimal XGBoost model...")
model = train_xgboost(X, y)

model_path = os.path.join(models_dir, 'xgb_model.joblib')
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
