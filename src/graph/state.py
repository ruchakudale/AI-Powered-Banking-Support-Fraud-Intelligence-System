from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class BankingState(TypedDict, total=False):

    # ----------------------------------------------------
    # Conversation memory
    # ----------------------------------------------------
    messages: Annotated[
        list[AnyMessage],
        add_messages
    ]

    # ----------------------------------------------------
    # Input
    # ----------------------------------------------------
    query: str
    sentiment: str
    urgency: str
    transaction: Optional[dict]

    # ----------------------------------------------------
    # Intent
    # ----------------------------------------------------
    intent: str
    model_intent: str
    fraud_override: bool
    intent_source: str

    # ----------------------------------------------------
    # Fraud
    # ----------------------------------------------------
    fraud_result: Optional[dict]
    fraud_probability: Optional[float]
    fraud_prediction: Optional[int]
    risk_level: Optional[str]
    suggested_action: list[str]
    escalate: bool

    # ----------------------------------------------------
    # Priority
    # ----------------------------------------------------
    handling_priority: str
    routing_reason: str

    # ----------------------------------------------------
    # RAG
    # ----------------------------------------------------
    policy_results: list[dict]
    historical_results: list[dict]
    rag_context: str

    # ----------------------------------------------------
    # LLM
    # ----------------------------------------------------
    prompt: str
    response: str

    # ----------------------------------------------------
    # Explainability
    # ----------------------------------------------------
    graph_path: list[str]