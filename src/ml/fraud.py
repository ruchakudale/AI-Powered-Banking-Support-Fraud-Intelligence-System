from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_fraud_model.pkl"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "models"
    / "fraud_config.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Fraud model not found:\n{MODEL_PATH}"
    )

final_fraud_model = joblib.load(
    MODEL_PATH
)


# ============================================================
# LOAD CONFIGURATION
# ============================================================

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Fraud configuration not found:\n{CONFIG_PATH}"
    )

fraud_config = joblib.load(
    CONFIG_PATH
)


# ============================================================
# CONFIG VALUES
# ============================================================

FINAL_THRESHOLD = fraud_config[
    "fraud_threshold"
]

LOW_THRESHOLD = fraud_config[
    "risk_thresholds"
]["low"]

MEDIUM_THRESHOLD = fraud_config[
    "risk_thresholds"
]["medium"]


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(
    fraud_probability: float
) -> str:
    """
    Convert fraud probability into a risk level.
    """

    if fraud_probability >= MEDIUM_THRESHOLD:
        return "High"

    elif fraud_probability >= LOW_THRESHOLD:
        return "Medium"

    return "Low"


# ============================================================
# FRAUD ACTION
# ============================================================

def get_fraud_action(
    risk_level: str
) -> dict:
    """
    Return business actions associated with a
    fraud risk level.
    """

    risk_actions = fraud_config[
        "risk_actions"
    ]

    if risk_level not in risk_actions:
        raise ValueError(
            f"Unknown risk level: {risk_level}"
        )

    return risk_actions[
        risk_level
    ]


# ============================================================
# FRAUD PREDICTION
# ============================================================

def predict_fraud_risk(
    transaction: dict
) -> dict:
    """
    Predict fraud probability, prediction,
    risk level, actions and escalation.

    Required fields:
        amount_inr
        hour_of_day
        day_of_week
    """

    required_features = [
        "amount_inr",
        "hour_of_day",
        "day_of_week"
    ]

    if not isinstance(transaction, dict):
        raise TypeError(
            "Transaction must be a dictionary."
        )

    missing_features = [
        feature
        for feature in required_features
        if feature not in transaction
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: "
            f"{missing_features}"
        )

    model_input = pd.DataFrame(
        [transaction]
    )

    model_input = model_input[
        required_features
    ]

    # --------------------------------------------------------
    # Fraud probability
    # --------------------------------------------------------

    fraud_probability = (
        final_fraud_model.predict_proba(
            model_input
        )[0, 1]
    )

    fraud_probability = float(
        fraud_probability
    )

    # --------------------------------------------------------
    # Fraud prediction
    # --------------------------------------------------------

    fraud_prediction = int(
        fraud_probability >= FINAL_THRESHOLD
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    risk_level = get_risk_level(
        fraud_probability
    )

    # --------------------------------------------------------
    # Business action
    # --------------------------------------------------------

    action_info = get_fraud_action(
        risk_level
    )

    return {
        "fraud_probability": round(
            fraud_probability,
            4
        ),

        "fraud_prediction": fraud_prediction,

        "risk_level": risk_level,

        "suggested_action": action_info[
            "suggested_action"
        ],

        "escalate": action_info[
            "escalate"
        ]
    }


# ============================================================
# MODEL CONFIGURATION
# ============================================================

def get_fraud_threshold() -> float:
    """
    Return the final fraud classification threshold.
    """

    return float(
        FINAL_THRESHOLD
    )