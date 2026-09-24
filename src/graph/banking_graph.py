from typing import Any

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from src.graph.state import BankingState

from src.ml.intent import (
    predict_intent_safe,
)

from src.ml.fraud import (
    predict_fraud_risk,
)

from src.rag.retriever import (
    retrieve_context,
    build_rag_context,
)

from src.llm.gemini import (
    classify_sentiment_and_urgency,
    build_conversation_history,
    build_llm_prompt,
    generate_response,
    build_policy_fallback,
)


# ============================================================
# PREPROCESS NODE
# ============================================================

def preprocess_node(
    state: BankingState
):
    """
    Clean the current query and add the current customer
    message to conversation memory.
    """

    query = state["query"].strip()

    if not query:
        raise ValueError(
            "Customer query cannot be empty."
        )

    graph_path = list(
        state.get("graph_path", [])
    )

    graph_path.append(
        "preprocess"
    )

    # --------------------------------------------------------
    # Avoid duplicate HumanMessage
    # --------------------------------------------------------

    messages = state.get(
        "messages",
        []
    )

    add_current_message = True

    if messages:

        last_message = messages[-1]

        if (
            isinstance(
                last_message,
                HumanMessage
            )
            and last_message.content.strip()
            == query
        ):
            add_current_message = False

    result = {
        "query": query,
        "graph_path": graph_path,
    }

    if add_current_message:

        result["messages"] = [
            HumanMessage(
                content=query
            )
        ]

    return result


# ============================================================
# INTENT NODE
# ============================================================

def intent_node(
    state: BankingState
):
    """
    Predict intent using the ML model plus the
    fraud-safety override.

    Follow-up questions use the previous customer
    message as context.
    """

    query = state["query"].strip()

    messages = state.get(
        "messages",
        []
    )

    # --------------------------------------------------------
    # Get user messages
    # --------------------------------------------------------

    user_messages = [
        message
        for message in messages
        if isinstance(
            message,
            HumanMessage
        )
    ]

    # --------------------------------------------------------
    # Follow-up detection
    # --------------------------------------------------------

    follow_up_phrases = [
        "what should i do",
        "what do i do",
        "what next",
        "what should i do next",
        "how do i proceed",
        "what happens next",
        "can you help me with this",
        "what about this",
        "and now",
        "next step",
    ]

    is_follow_up = any(
        phrase in query.lower()
        for phrase in follow_up_phrases
    )

    # --------------------------------------------------------
    # Build classification query
    # --------------------------------------------------------

    classification_query = query

    if (
        is_follow_up
        and len(user_messages) >= 2
    ):
        previous_query = (
            user_messages[-2].content
        )

        classification_query = (
            f"Previous customer query: "
            f"{previous_query}\n"
            f"Current customer query: "
            f"{query}"
        )

    # --------------------------------------------------------
    # Safe intent prediction
    # --------------------------------------------------------

    intent_result = predict_intent_safe(
        classification_query
    )

    model_intent = (
        intent_result["model_intent"]
    )

    final_intent = (
        intent_result["intent"]
    )

    fraud_override = (
        intent_result["fraud_override"]
    )

    intent_source = (
        intent_result["intent_source"]
    )

    graph_path = list(
        state.get("graph_path", [])
    )

    graph_path.append(
        "intent"
    )

    return {
        "intent": final_intent,
        "model_intent": model_intent,
        "fraud_override": fraud_override,
        "intent_source": intent_source,

        # Reset fraud values for this turn.
        # fraud_node will populate them if needed.
        "fraud_result": None,
        "fraud_probability": None,
        "fraud_prediction": None,
        "risk_level": None,
        "suggested_action": [],
        "escalate": False,

        "graph_path": graph_path,
    }


# ============================================================
# SENTIMENT + URGENCY NODE
# ============================================================

def sentiment_node(
    state: BankingState
):
    """
    Classify runtime sentiment and urgency using Gemini.
    """

    query = state["query"]

    conversation_history = (
        build_conversation_history(
            state.get(
                "messages",
                []
            ),
            current_query=query,
        )
    )

    result = classify_sentiment_and_urgency(
        query=query,
        conversation_history=conversation_history,
    )

    graph_path = list(
        state.get("graph_path", [])
    )

    graph_path.append(
        "sentiment"
    )

    return {
        "sentiment": result["sentiment"],
        "urgency": result["urgency"],
        "graph_path": graph_path,
    }


# ============================================================
# PRIORITY NODE
# ============================================================

def priority_node(state: BankingState):

    urgency = state.get("urgency", "Medium")
    intent = state.get("intent")
    transaction = state.get("transaction") or {}

    required_features = {
        "amount_inr",
        "hour_of_day",
        "day_of_week",
    }

    has_transaction_data = required_features.issubset(
        transaction.keys()
    )

    priority_map = {
        "High": "High",
        "Medium": "Medium",
        "Low": "Low",
    }

    handling_priority = priority_map.get(
        urgency,
        "Medium"
    )

    if (
        intent == "Fraud/Unauthorized"
        and not has_transaction_data
    ):
        routing_reason = (
            "Fraud-related request detected. "
            "Transaction risk scoring was not performed because "
            "the required transaction features "
            "(amount, transaction hour, and day of week) "
            "were not fully available."
        )

    elif urgency == "High":
        routing_reason = (
            "Customer query marked high urgency."
        )

    elif urgency == "Medium":
        routing_reason = (
            "Customer query requires standard priority handling."
        )

    else:
        routing_reason = (
            "Customer query marked low urgency."
        )

    return {
        "handling_priority": handling_priority,
        "routing_reason": routing_reason,
    }


# ============================================================
# ROUTER
# ============================================================

def route_after_intent(state: BankingState):
    """
    Run the fraud model only when the request is fraud-related
    AND the required transaction features are available.
    """

    intent = state.get("intent")

    transaction = state.get("transaction") or {}

    required_features = {
        "amount_inr",
        "hour_of_day",
        "day_of_week",
    }

    has_transaction_data = required_features.issubset(
        transaction.keys()
    )

    if intent == "Fraud/Unauthorized" and has_transaction_data:
        return "fraud"

    return "rag"


# ============================================================
# FRAUD NODE
# ============================================================

def fraud_node(
    state: BankingState
):
    """
    Run fraud analysis when transaction data is available.

    If this is a follow-up and the transaction is already
    stored in the checkpoint, it remains available in state.
    """

    transaction = state.get(
        "transaction"
    )

    graph_path = list(
        state.get("graph_path", [])
    )

    graph_path.append(
        "fraud"
    )

    # --------------------------------------------------------
    # No transaction data
    # --------------------------------------------------------

    if transaction is None:

        return {
            "fraud_result": None,
            "fraud_probability": None,
            "fraud_prediction": None,
            "risk_level": None,
            "suggested_action": [],
            "escalate": False,
            "graph_path": graph_path,
        }

    # --------------------------------------------------------
    # Run fraud model
    # --------------------------------------------------------

    fraud_result = predict_fraud_risk(
        transaction
    )

    return {
        "fraud_result": fraud_result,
        "fraud_probability": (
            fraud_result["fraud_probability"]
        ),
        "fraud_prediction": (
            fraud_result["fraud_prediction"]
        ),
        "risk_level": (
            fraud_result["risk_level"]
        ),
        "suggested_action": (
            fraud_result["suggested_action"]
        ),
        "escalate": (
            fraud_result["escalate"]
        ),
        "graph_path": graph_path,
    }


# ============================================================
# RAG NODE
# ============================================================

def rag_node(
    state: BankingState
):
    """
    Retrieve authoritative policy information and
    historical support cases.
    """

    query = state["query"]
    intent = state["intent"]

    context = retrieve_context(
        query=query,
        intent=intent,
        policy_top_k=3,
        history_top_k=2,
    )

    rag_context = build_rag_context(
        query=query,
        intent=intent,
        context=context,
    )

    graph_path = list(
        state.get("graph_path", [])
    )

    graph_path.append(
        "rag"
    )

    return {
        "policy_results": (
            context["policy_results"]
        ),
        "historical_results": (
            context["historical_results"]
        ),
        "rag_context": rag_context,
        "graph_path": graph_path,
    }


# ============================================================
# ACTION NODE
# ============================================================

def action_node(state: BankingState):

    intent = state.get("intent")
    urgency = state.get("urgency", "Medium")

    # --------------------------------------------------------
    # Fraud / Unauthorized
    # --------------------------------------------------------

    if intent == "Fraud/Unauthorized":

        fraud_probability = state.get(
            "fraud_probability"
        )

        existing_actions = state.get(
            "suggested_action"
        ) or []

        # Fraud model was actually executed
        if fraud_probability is not None:

            return {
                "suggested_action": existing_actions,
                "escalate": state.get(
                    "escalate",
                    False
                ),
            }

        # Fraud-related support request but no transaction data
        actions = [
            "Do not share your OTP, PIN, or CVV with anyone.",
            "Contact the bank through its official support channel if you suspect unauthorized activity.",
            "Report any unauthorized transaction promptly through the bank's official fraud-reporting process.",
        ]

        return {
            "suggested_action": actions,
            "escalate": False,
        }

    # --------------------------------------------------------
    # KYC
    # --------------------------------------------------------

    if intent == "KYC":

        actions = [
            "Provide the required KYC documents.",
            "Use the bank's official KYC channel.",
        ]

        return {
            "suggested_action": actions,
            "escalate": urgency == "High",
        }

    # --------------------------------------------------------
    # Loan
    # --------------------------------------------------------

    if intent == "Loan":

        actions = [
            "Review the required loan documents.",
            "Submit the application through the bank's official channel.",
        ]

        return {
            "suggested_action": actions,
            "escalate": urgency == "High",
        }

    # --------------------------------------------------------
    # Account Access
    # --------------------------------------------------------

    if intent == "Account Access":

        actions = [
            "Use the bank's official account-recovery process.",
            "Contact customer support if access cannot be restored.",
        ]

        return {
            "suggested_action": actions,
            "escalate": urgency == "High",
        }

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return {
        "suggested_action": [
            "Route the request to customer support."
        ],
        "escalate": urgency == "High",
    }





# ============================================================
# LLM NODE
# ============================================================

def llm_node(
    state: BankingState
):
    """
    Generate the final customer-facing response.

    If no authoritative policy is available, use the
    deterministic safe fallback instead of Gemini.
    """

    query = state["query"]
    intent = state["intent"]

    sentiment = state.get(
        "sentiment",
        "Unknown"
    )

    urgency = state.get(
        "urgency",
        "Low"
    )

    handling_priority = state.get(
        "handling_priority",
        "Low"
    )

    routing_reason = state.get(
        "routing_reason",
        ""
    )

    risk_level = state.get(
        "risk_level"
    )

    fraud_probability = state.get(
        "fraud_probability"
    )

    suggested_action = state.get(
        "suggested_action",
        []
    )

    escalate = state.get(
        "escalate",
        False
    )

    rag_context = state.get(
        "rag_context",
        ""
    )

    policy_results = state.get(
        "policy_results",
        []
    )

    graph_path = list(
        state.get("graph_path", [])
    )

    graph_path.append(
        "llm"
    )

    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------

    conversation_history = (
        build_conversation_history(
            state.get(
                "messages",
                []
            ),
            current_query=query,
        )
    )

    # --------------------------------------------------------
    # No authoritative policy
    # --------------------------------------------------------

    if not policy_results:

        response = build_policy_fallback(
            intent
        )

        prompt = ""

    # --------------------------------------------------------
    # Authoritative policy available
    # --------------------------------------------------------

    else:

        prompt = build_llm_prompt(
            query=query,
            intent=intent,
            sentiment=sentiment,
            urgency=urgency,
            risk_level=risk_level,
            fraud_probability=fraud_probability,
            suggested_action=suggested_action,
            escalate=escalate,
            rag_context=rag_context,
            conversation_history=conversation_history,
            handling_priority=handling_priority,
            routing_reason=routing_reason,
        )

        response = generate_response(
            prompt
        )

    # --------------------------------------------------------
    # Store AI response in MessagesState
    # --------------------------------------------------------

    ai_message = AIMessage(
        content=response
    )

    return {
        "prompt": prompt,
        "response": response,
        "messages": [
            ai_message
        ],
        "graph_path": graph_path,
    }


# ============================================================
# GRAPH FACTORY
# ============================================================

def create_banking_graph(
    checkpointer: Any = None
):
    """
    Create and compile the banking support LangGraph.

    Parameters
    ----------
    checkpointer : optional
        LangGraph checkpointer.

        Example:
            SqliteSaver(connection)

    Returns
    -------
    CompiledStateGraph
        Ready-to-use banking graph.
    """

    graph_builder = StateGraph(
        BankingState
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    graph_builder.add_node(
        "preprocess",
        preprocess_node
    )

    graph_builder.add_node(
        "intent",
        intent_node
    )

    graph_builder.add_node(
        "sentiment",
        sentiment_node
    )

    graph_builder.add_node(
        "priority",
        priority_node
    )

    graph_builder.add_node(
        "fraud",
        fraud_node
    )

    graph_builder.add_node(
        "rag",
        rag_node
    )

    graph_builder.add_node(
        "action",
        action_node
    )

    graph_builder.add_node(
        "llm",
        llm_node
    )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    graph_builder.add_edge(
        START,
        "preprocess"
    )

    graph_builder.add_edge(
        "preprocess",
        "intent"
    )

    graph_builder.add_edge(
        "intent",
        "sentiment"
    )

    graph_builder.add_edge(
        "sentiment",
        "priority"
    )

    # Conditional routing
    graph_builder.add_conditional_edges(
        "priority",
        route_after_intent,
        {
            "fraud": "fraud",
            "rag": "rag",
        }
    )

    # Fraud → RAG
    graph_builder.add_edge(
        "fraud",
        "rag"
    )

    # RAG → Action
    graph_builder.add_edge(
        "rag",
        "action"
    )

    # Action → LLM
    graph_builder.add_edge(
        "action",
        "llm"
    )

    # LLM → END
    graph_builder.add_edge(
        "llm",
        END
    )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    if checkpointer is not None:

        return graph_builder.compile(
            checkpointer=checkpointer
        )
   

    return graph_builder.compile()