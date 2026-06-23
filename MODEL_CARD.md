# Model Card: ICU Sepsis Early Warning System

## Model Details
- **Model type:** XGBoost binary classifier (primary), Isolation Forest anomaly detector (secondary)
- **Version:** 0.1.0
- **Developed by:** Roshini Jayasankar
- **Date:** June 2026
- **Contact:** roshinijayasankar@gmail.com
- **Repository:** https://github.com/roshijay/Anomaly-Detection-in-ICU

---

## Intended Use

### Primary intended use
Clinical decision support tool for early sepsis detection in ICU settings. 
The system flags patients whose vitals and lab values suggest elevated sepsis 
risk, surfacing them for clinician review.

### Intended users
- ICU nurses and physicians reviewing patient status dashboards
- Clinical informatics teams evaluating early warning system prototypes
- Researchers studying sepsis prediction on the PhysioNet 2019 dataset

### Out-of-scope uses
- **This model is NOT a diagnostic tool.** It must not be used as a standalone 
  basis for clinical decisions without physician review.
- Not validated for use outside ICU settings (e.g. emergency department, 
  general ward).
- Not validated on pediatric patients.
- Not intended for deployment in production clinical environments without 
  prospective validation and regulatory review.

---

## Training Data

- **Dataset:** PhysioNet Computing in Cardiology Challenge 2019
- **Source:** https://physionet.org/content/challenge-2019/1.0.0/
- **Size:** 1,552,210 patient-hours across 40,336 ICU patients
- **Label definition:** SepsisLabel=1 defined using Sepsis-3 criteria — 
  suspected infection plus SOFA score increase ≥ 2 points
- **Class distribution:** 7.27% of patients developed sepsis (patient-level); 
  1.8% of patient-hours labeled sepsis (hour-level)
- **Hospital systems:** Two anonymized US hospital systems (Set A and Set B)

### Data preprocessing
- Rolling window features (6-hour lookback) computed for 7 vital signs: 
  mean, std, min, max, trend
- Lab recency features for 7 key labs: forward-filled last known value, 
  binary "was measured" flag, hours since last measurement
- Time-artifact features excluded: ICULOS, HospAdmTime, Hour 
  (correlated with label by construction, not genuine physiological signal)
- Raw sparse lab columns dropped in favor of engineered versions

---

## Evaluation

### Train/test split
- **Method:** Patient-level split (entire patients assigned to train or test, 
  never both) — prevents data leakage from the same patient appearing in both sets
- **Split:** 80% train, 20% test, stratified by patient-level sepsis status

### Performance metrics

| Metric | XGBoost | Isolation Forest |
|--------|---------|-----------------|
| AUC-ROC | 0.8133 | 0.6290 |
| Recall @ threshold | 0.85 | 0.11 |
| Precision @ threshold | 0.037 | 0.03 |
| Threshold used | 0.2635 | N/A |

### Threshold selection rationale
The operating threshold (0.2635) was selected to achieve recall ≥ 0.85, 
prioritizing sensitivity over specificity. In clinical sepsis detection, 
a missed case (false negative) carries far higher cost than a false alarm 
(false positive). This results in a high false positive rate (~96%), which 
in deployment would require a triage layer combining model output with 
clinical judgment.

### SHAP feature importance (top clinical drivers)
1. Lactate_hours_since_measured
2. Bilirubin_total_last_known
3. Bilirubin_total_hours_since_measured
4. Lactate_last_known
5. Temp_rolling_max
6. BUN_last_known
7. WBC_last_known

These align with Sepsis-3 SOFA score components, validating that the model 
learns genuine physiological signal rather than spurious correlations.

---

## Limitations and Known Issues

1. **Single time-point API simplification:** The `/predict` endpoint accepts 
   a single snapshot of vitals. Rolling window features (6-hour trends) are 
   approximated using current values. Risk scores from the API underestimate 
   true risk compared to the full time-series model. A production version 
   would maintain a stateful patient history buffer.

2. **Lab feature sparsity:** Only 7 of 33 available lab variables received 
   full feature engineering. Remaining labs are zero-filled, a known 
   simplification that may affect performance for patients with unusual 
   lab profiles.

3. **No prospective validation:** The model has been evaluated only on 
   held-out data from the same two hospital systems used for training. 
   Performance on data from different institutions, patient populations, 
   or EHR systems is unknown.

4. **Class imbalance:** Despite class-weighted training, precision at the 
   clinical threshold remains low (~4%). The model is designed as a 
   screening tool, not a confirmatory one.

5. **Sepsis-3 definition dependency:** Labels reflect Sepsis-3 criteria. 
   Performance under alternative sepsis definitions (e.g. Sepsis-2) is 
   not evaluated.

---

## Ethical Considerations

### Fairness
- The dataset includes demographic variables (Age, Gender) as features. 
  Subgroup performance across age, gender, and hospital system has not 
  been evaluated. Differential performance across demographic groups is 
  a known risk in clinical AI and should be assessed before any deployment.

### Human oversight
- This system is designed as decision support, not autonomous decision-making. 
  All alerts must be reviewed by a qualified clinician before any clinical 
  action is taken.

### Privacy
- Training data is de-identified per PhysioNet data use agreement. 
  No patient identifiers are used as model features.

### Transparency
- Model predictions are explained using SHAP values, identifying the 
  top contributing features per prediction. This supports clinician 
  understanding of why a patient was flagged.

---

## How to Use

### API endpoint
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "HR": 110, "Temp": 38.9, "SBP": 88,
    "Lactate": 4.2, "WBC": 18.5, "Creatinine": 2.1
  }'
```

### Response
```json
{
  "sepsis_risk_score": 0.1869,
  "flagged": false,
  "threshold_used": 0.2635
}
```

---

## Citation

If referencing the training data:
> Reyna, M., Josef, C., Seyedi, S., Jeter, R., Shashikumar, S. P., 
> Westover, M. B., ... & Clifford, G. D. (2019). Early prediction of sepsis 
> from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. 
> Critical Care Medicine.