import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture
def sample_classification_data():
    """Generates synthetic binary classification data."""
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        random_state=42
    )
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="target")


@pytest.fixture
def sample_regression_data():
    """Generates synthetic continuous regression data."""
    X, y = make_regression(
        n_samples=150,
        n_features=4,
        noise=0.1,
        random_state=42
    )
    feature_names = [f"num_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="target")


@pytest.fixture
def sample_transaction_payload():
    """Returns a realistic simulated incoming transaction dictionary for fraud inference."""
    return {
        "transaction_id": "TXN_TEST_001",
        "income": 0.85,
        "name_email_similarity": 0.15,
        "prev_address_months_count": -1,
        "current_address_months_count": 4,
        "customer_age": 42,
        "days_since_request": 0,
        "intended_balcon_amount": 9500,
        "payment_type": "AA",
        "zip_count_4w": 2,
        "velocity_6h": 8,
        "velocity_24h": 12,
        "velocity_4w": 20,
        "bank_branch_count_8w": 4,
        "date_of_birth_distinct_emails_4w": 2,
        "employment_status": "CE",
        "credit_risk_score": 140,
        "email_is_free": 1,
        "housing_status": "BB",
        "phone_home_valid": 0,
        "phone_mobile_valid": 1,
        "bank_months_count": 2,
        "has_other_cards": 0,
        "proposed_credit_limit": 8000,
        "foreign_request": 1,
        "source": "INTERNET",
        "session_length_in_minutes": 3.0,
        "device_os": "linux",
        "keep_alive_session": 1,
        "device_distinct_emails_8w": 1,
        "device_fraud_count": 0
    }
