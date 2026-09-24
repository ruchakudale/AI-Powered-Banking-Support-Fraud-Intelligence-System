from pathlib import Path
import re
import joblib


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "intent_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Intent model not found:\n{MODEL_PATH}"
    )

intent_model = joblib.load(
    MODEL_PATH
)


# ============================================================
# BASIC INTENT PREDICTION
# ============================================================

def predict_intent(query: str) -> str:
    """
    Predict the banking intent for a customer query.
    """

    if not isinstance(query, str):
        raise TypeError(
            "Query must be a string."
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    prediction = intent_model.predict(
        [query]
    )[0]

    return str(prediction)


# ============================================================
# INTENT + DECISION SCORES
# ============================================================

def predict_intent_with_scores(query: str):
    """
    Predict intent and return the SVM decision score
    for every intent class.
    """

    if not isinstance(query, str):
        raise TypeError(
            "Query must be a string."
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    prediction = intent_model.predict(
        [query]
    )[0]

    scores = intent_model.decision_function(
        [query]
    )[0]

    score_map = dict(
        zip(
            intent_model.classes_,
            scores
        )
    )

    return {
        "intent": str(prediction),
        "scores": {
            label: round(
                float(score),
                4
            )
            for label, score in score_map.items()
        }
    }


# ============================================================
# AVAILABLE INTENT CLASSES
# ============================================================

def get_intent_classes():
    """
    Return all intent classes known by the model.
    """

    return list(
        intent_model.classes_
    )

def fraud_safety_override(query: str) -> bool:
    """
    Detect explicit unauthorized/fraud language that should
    safely route the query to the fraud workflow.

    This is a business safety rule, not an ML prediction.
    """

    query = query.lower().strip()

    fraud_patterns = [
        r"\bunauthori[sz]ed\b",
        r"\bdidn'?t make\b",
        r"\bdid not make\b",
        r"\bused my otp\b",
        r"\botp.*transfer\b",
        r"\btransfer.*otp\b",
        r"\botp.*without.*consent\b",
        r"\botp.*not.*requested\b",
        r"\bnot me\b",
        r"\bwithout my (?:permission|consent|knowledge)\b",
        r"\bwithout my approval\b",
        r"\bunknown (?:transaction|charge|merchant)\b",
        r"\b(?:fraud|fraudulent|scam|stolen)\b",
        r"\b(?:credit|debit) card\b.*\bwithout\b",
        r"\bupi\b.*\bwithout my consent\b",
        r"\btransaction\b.*\b(?:don't|do not|didn't)\b.*\bmake\b"
    ]

    return any(
        re.search(
            pattern,
            query
        )
        for pattern in fraud_patterns
    )

def predict_intent_safe(query: str):
    """
    Predict intent using the ML model plus an explicit
    fraud-safety override.

    Returns both the original ML prediction and the
    final routing intent.
    """

    if not isinstance(query, str):
        raise TypeError(
            "Query must be a string."
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    # ML prediction
    model_intent = predict_intent(
        query
    )

    # Safety rule
    fraud_override = fraud_safety_override(
        query
    )

    if fraud_override:
        final_intent = "Fraud/Unauthorized"
        intent_source = "fraud_safety_rule"
    else:
        final_intent = model_intent
        intent_source = "ml_model"

    return {
        "intent": final_intent,
        "model_intent": model_intent,
        "fraud_override": fraud_override,
        "intent_source": intent_source
    }