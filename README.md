# Telecom Churn Prediction Pipeline

## Problem
Predict whether a telecom customer will leave (churn).

## Target
- 1 → Churn
- 0 → Stay

## Steps Implemented

### Rule 1: Preprocessing
- Missing value handling
- Label encoding

### Rule 2: Feature Engineering
- avg_monthly_spend
- charges_tenure_ratio

### Rule 3: Model Training
- RandomForestClassifier
- Accuracy + classification report printed

### Rule 4: Model Saving
- Saved as `model.pkl`

### Rule 5: Prediction Function
- `predict(input_dict)` returns:
  - prediction
  - confidence score

### Rule 6: README Docstring
- Included at top of pipeline.py

## How to Run

### Step 1: Install dependencies