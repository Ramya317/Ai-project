"""
README (REQUIRED RULE #6)

Project: Telecom Customer Churn Prediction

What this model does:
Predicts whether a telecom customer will churn (leave the service).

Target:
1 → Customer will leave
0 → Customer will stay

Input format (for predict function):
{
    "tenure": int,
    "monthly_charges": float,
    "total_charges": float,
    "contract": "Month-to-month" | "One year" | "Two year"
}

Output:
{
    "prediction": "Human readable label",
    "confidence": probability score
}

Known limitation:
This model uses a small synthetic dataset and may not generalize well to real-world telecom data.
"""

# ===============================
# IMPORTS
# ===============================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ===============================
# RULE #1: DATA PREPROCESSING
# ===============================
def preprocess(df):
    df = df.copy()
    df.fillna(0, inplace=True)

    le = LabelEncoder()
    df["contract"] = le.fit_transform(df["contract"])

    return df


# ===============================
# RULE #2: FEATURE ENGINEERING
# ===============================
def engineer_features(df):
    df = df.copy()

    df["avg_monthly_spend"] = df["total_charges"] / (df["tenure"] + 1)
    df["charges_tenure_ratio"] = df["monthly_charges"] / (df["tenure"] + 1)

    return df


# ===============================
# RULE #3: TRAIN + EVALUATE MODEL
# ===============================
def train_model():
    df = pd.read_csv("sample_data.csv")

    df = preprocess(df)
    df = engineer_features(df)

    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== MODEL EVALUATION ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "model.pkl")

    return model


# ===============================
# RULE #5: PREDICT FUNCTION (UPDATED OUTPUT FORMAT)
# ===============================
def predict(input_dict):
    """
    Loads model and predicts with human-readable output
    """
    model = joblib.load("model.pkl")

    df = pd.DataFrame([input_dict])

    df = preprocess(df)
    df = engineer_features(df)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][prediction]

    # ✅ HUMAN READABLE OUTPUT
    if prediction == 1:
        label = "1 → Customer will leave"
    else:
        label = "0 → Customer will stay"

    return {
        "prediction": label,
        "confidence": round(float(probability), 3)
    }


# ===============================
# RUN PIPELINE
# ===============================
if __name__ == "__main__":
    train_model()

    sample_input = {
        "tenure": 5,
        "monthly_charges": 300,
        "total_charges": 1500,
        "contract": "Month-to-month",
    }

    result = predict(sample_input)

    print("\n=== SAMPLE PREDICTION ===")
    print(result)