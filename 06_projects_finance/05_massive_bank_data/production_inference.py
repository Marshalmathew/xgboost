"""
Microservice Simulation: Real-Time Fraud Detection API
------------------------------------------------------
This script acts as the backend inference endpoint for our SOTA Fraud Model.
It receives a JSON payload (representing a live credit card swipe),
loads the serialized Universal JSON XGBoost model, and returns a sub-millisecond decision.
"""
import json
import logging
import os
import time

import joblib
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)

# Global state to keep the model in memory across API calls
_MODEL = None
_ISO_FOREST = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pre-defined categorical levels learned during training to prevent single-row categorical code corruption
CATEGORICAL_CATEGORIES = {
    "payment_type": ["AA", "AB", "AC", "AD", "AE"],
    "employment_status": ["CA", "CB", "CC", "CD", "CE", "CF", "CG"],
    "housing_status": ["BA", "BB", "BC", "BD", "BE", "BF", "BG"],
    "source": ["INTERNET", "TELEAPP"],
    "device_os": ["windows", "macintosh", "linux", "x11", "other"],
}

# Expected model features in order
EXPECTED_FEATURES = [
    "income", "name_email_similarity", "prev_address_months_count",
    "current_address_months_count", "customer_age", "days_since_request",
    "intended_balcon_amount", "payment_type", "zip_count_4w",
    "velocity_6h", "velocity_24h", "velocity_4w", "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w", "employment_status", "credit_risk_score",
    "email_is_free", "housing_status", "phone_home_valid", "phone_mobile_valid",
    "bank_months_count", "has_other_cards", "proposed_credit_limit",
    "foreign_request", "source", "session_length_in_minutes", "device_os",
    "keep_alive_session", "device_distinct_emails_8w", "device_fraud_count",
    "Anomaly_Score"
]


def load_models():
    """Loads the serialized model into memory (runs once on server startup)."""
    global _MODEL, _ISO_FOREST

    model_path = os.path.join(BASE_DIR, "fraud_model.json")
    iso_path = os.path.join(BASE_DIR, "isolation_forest.joblib")

    if not os.path.exists(model_path):
        print(f"WARNING: '{model_path}' not found. Please run the 02_fraud_detection notebook first.")
        return False

    print("Loading Universal XGBoost Model (JSON)...")
    _MODEL = xgb.Booster()
    _MODEL.load_model(model_path)

    if os.path.exists(iso_path):
        print("Loading Unsupervised Anomaly Engine (Joblib)...")
        _ISO_FOREST = joblib.load(iso_path)
    else:
        print("WARNING: Anomaly engine not found, will mock anomaly scores for demo.")

    return True


def score_transaction(payload: dict) -> dict:
    """
    Simulates a production-grade inference API endpoint.
    Expects a dictionary of transaction features. Returns a decision payload with measured latency.
    """
    if _MODEL is None:
        return {"error": "Model not loaded"}

    start_time = time.perf_counter()

    # 1. Convert JSON payload to DataFrame
    df = pd.DataFrame([payload])

    # 2. Extract transaction metadata
    transaction_id = df.pop("transaction_id").iloc[0] if "transaction_id" in df else "UNKNOWN_ID"

    # Schema drift check: identify unexpected fields
    valid_input_fields = set(EXPECTED_FEATURES) | set(CATEGORICAL_CATEGORIES.keys()) | {"transaction_id"}
    extra_fields = [c for c in df.columns if c not in valid_input_fields]
    if extra_fields:
        logger.warning("Payload contains unmodeled columns that will be ignored: %s", extra_fields)
        df = df.drop(columns=extra_fields)

    # 3. Handle Categoricals with fixed schemas matching training categories (Pandas 3/4 compatible)
    for col, categories in CATEGORICAL_CATEGORIES.items():
        if col in df.columns:
            cleaned = df[col].where(df[col].isin(categories))
            df[col] = pd.Categorical(cleaned, categories=categories)

    # 4. Inject Unsupervised Anomaly Score
    if _ISO_FOREST is not None:
        numeric_features = df.select_dtypes(include=["number"]).fillna(0)
        df["Anomaly_Score"] = -1 * _ISO_FOREST.score_samples(numeric_features)
    else:
        # Mock score if Isolation Forest wasn't serialized
        df["Anomaly_Score"] = 0.5

    # Align columns exactly to the model schema
    for feat in EXPECTED_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    df_model = df[EXPECTED_FEATURES]

    # 5. Execute Optimized Inference (In-place if supported, fallback to DMatrix)
    try:
        raw_pred = _MODEL.inplace_predict(df_model)
        fraud_prob = float(raw_pred[0])
    except Exception:
        dtest = xgb.DMatrix(df_model, enable_categorical=True)
        fraud_prob = float(_MODEL.predict(dtest)[0])

    # 6. Apply Business Logic Threshold
    BLOCK_THRESHOLD = 0.35
    decision = "BLOCK" if fraud_prob > BLOCK_THRESHOLD else "APPROVE"

    latency_ms = (time.perf_counter() - start_time) * 1000

    # 7. Return Backend API Response with genuine measured latency
    return {
        "transaction_id": transaction_id,
        "fraud_probability": round(fraud_prob, 4),
        "decision": decision,
        "anomaly_flag": bool(df["Anomaly_Score"].iloc[0] > 0.8),
        "latency_ms": f"{latency_ms:.2f}ms"
    }


if __name__ == "__main__":
    print("--- FRAUD INFERENCE MICROSERVICE ---")
    if load_models():
        simulated_payload = {
            "transaction_id": "TXN_987654321",
            "income": 0.8,
            "name_email_similarity": 0.1,
            "prev_address_months_count": -1,
            "current_address_months_count": 2,
            "customer_age": 45,
            "days_since_request": 0,
            "intended_balcon_amount": 10500,
            "payment_type": "AA",
            "zip_count_4w": 3,
            "velocity_6h": 12,
            "velocity_24h": 15,
            "velocity_4w": 25,
            "bank_branch_count_8w": 5,
            "date_of_birth_distinct_emails_4w": 3,
            "employment_status": "CE",
            "credit_risk_score": 150,
            "email_is_free": 1,
            "housing_status": "BB",
            "phone_home_valid": 0,
            "phone_mobile_valid": 1,
            "bank_months_count": 1,
            "has_other_cards": 0,
            "proposed_credit_limit": 10000,
            "foreign_request": 1,
            "source": "INTERNET",
            "session_length_in_minutes": 2.5,
            "device_os": "linux",
            "keep_alive_session": 1,
            "device_distinct_emails_8w": 1,
            "device_fraud_count": 0
        }

        print(f"\nProcessing Incoming Transaction: {simulated_payload['transaction_id']}")
        response = score_transaction(simulated_payload)

        print("\n--- API RESPONSE ---")
        print(json.dumps(response, indent=2))
