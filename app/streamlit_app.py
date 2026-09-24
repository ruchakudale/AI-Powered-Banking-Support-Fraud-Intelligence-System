# ============================================================
# AI-POWERED BANKING SUPPORT & FRAUD INTELLIGENCE SYSTEM
# Streamlit Application
# ============================================================

import sqlite3
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.banking_graph import create_banking_graph


RETRIEVAL_EVAL_PATH = (
    PROJECT_ROOT / "data" / "retrieval_evaluation_results.csv"
)

RESPONSE_EVAL_PATH = (
    PROJECT_ROOT / "data" / "response_evaluation_results.csv"
)

QA_PATH = PROJECT_ROOT / "data" / "qa_pairs.json"

CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "banking_checkpoints.sqlite"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Banking AI Support",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# UI STYLING
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        .app-subtitle {
            color: #6b7280;
            font-size: 0.92rem;
            line-height: 1.5;
            margin-top: -8px;
            margin-bottom: 22px;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.82rem;
            font-weight: 500;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.05rem;
            font-weight: 600;
            line-height: 1.25;
        }

        div[data-testid="stMetric"] {
            padding: 0.25rem 0;
        }

        .stMarkdown {
            line-height: 1.55;
        }

        section[data-testid="stSidebar"] {
            padding-top: 1.5rem;
        }

        div[data-testid="stExpander"] {
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data
def load_evaluation_data():
    """Load saved evaluation results and restore QA metadata."""

    retrieval_df = pd.DataFrame()
    response_df = pd.DataFrame()

    if RETRIEVAL_EVAL_PATH.exists():
        retrieval_df = pd.read_csv(
            RETRIEVAL_EVAL_PATH
        )

    if RESPONSE_EVAL_PATH.exists():
        response_df = pd.read_csv(
            RESPONSE_EVAL_PATH
        )

    if (
        not response_df.empty
        and QA_PATH.exists()
    ):
        qa_df = pd.read_json(QA_PATH)

        qa_df["id"] = qa_df["id"].astype(str)

        if "id" in response_df.columns:
            response_df["id"] = (
                response_df["id"].astype(str)
            )

            if "category" not in response_df.columns:
                response_df = response_df.merge(
                    qa_df[["id", "category"]],
                    on="id",
                    how="left",
                )

        elif "question" in response_df.columns:
            response_df = response_df.merge(
                qa_df[
                    [
                        "id",
                        "category",
                        "question",
                    ]
                ],
                on="question",
                how="left",
            )

    return retrieval_df, response_df


@st.cache_resource(scope="session")
def load_graph():
    """Create the LangGraph with a persistent SQLite checkpointer."""

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        str(CHECKPOINT_PATH),
        check_same_thread=False,
    )

    checkpointer = SqliteSaver(
        connection
    )

    graph = create_banking_graph(
        checkpointer=checkpointer
    )

    return graph, connection


graph, db_connection = load_graph()

retrieval_eval_df, response_eval_df = (
    load_evaluation_data()
)


# ============================================================
# SESSION STATE
# ============================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = (
        f"streamlit_{uuid.uuid4().hex[:12]}"
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def start_new_conversation():
    """Reset the UI conversation and create a fresh graph thread."""

    st.session_state.thread_id = (
        f"streamlit_{uuid.uuid4().hex[:12]}"
    )

    st.session_state.chat_history = []
    st.session_state.last_result = None


def format_probability(value):
    """Format a probability for display."""

    if value is None:
        return "Not assessed"

    return f"{float(value):.2%}"


def display_actions(actions):
    """Display suggested actions consistently."""

    if not actions:
        st.write("No predefined action.")
        return

    for action in actions:
        st.markdown(
            f"<div style='margin-bottom:6px; font-size:0.92rem;'>"
            f"• {action}"
            f"</div>",
            unsafe_allow_html=True,
        )


def display_metric_row(metrics):
    """Display up to four metrics in a compact two-column layout."""

    left, right = st.columns(2)

    for column, (label, value) in zip(
        [left, right],
        metrics[:2],
    ):
        with column:
            st.metric(label, value)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🏦 Banking AI")

    st.caption(
        "AI-powered banking support and fraud intelligence"
    )

    st.divider()

    st.markdown("### Conversation")

    st.caption(
        f"Thread: `{st.session_state.thread_id}`"
    )

    if st.button(
        "🆕 New Conversation",
        use_container_width=True,
    ):
        start_new_conversation()
        st.rerun()

    st.divider()

    st.markdown("### Transaction Risk Assessment")

    st.caption(
        "Optional transaction features for the behavioral "
        "fraud model."
    )

    use_transaction = st.checkbox(
        "Run transaction fraud model",
        value=False,
    )

    transaction = None

    if use_transaction:

        amount_inr = st.number_input(
            "Transaction amount (₹)",
            min_value=0.0,
            value=5000.0,
            step=100.0,
        )

        hour_of_day = st.slider(
            "Transaction hour",
            min_value=0,
            max_value=23,
            value=23,
        )

        day_of_week = st.selectbox(
            "Day of week",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
        )

        transaction = {
            "amount_inr": amount_inr,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
        }

        st.success(
            "Fraud model will be executed."
        )

    else:
        st.info(
            "Fraud model is skipped unless complete "
            "transaction features are provided."
        )

    st.divider()

    st.markdown("### System Components")

    st.caption("LangGraph orchestration")
    st.caption("Linear SVM intent classification")
    st.caption("Gemini sentiment + urgency")
    st.caption("Behavioral Random Forest fraud model")
    st.caption("FAISS + BGE-small RAG")
    st.caption("SQLite conversation memory")

    st.divider()

    if st.button(
        "🔄 Refresh Evaluation",
        use_container_width=True,
    ):
        load_evaluation_data.clear()
        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🏦 AI Banking Support Assistant")

st.markdown(
    """
    <div class="app-subtitle">
    Intelligent banking support using ML, RAG, LangGraph,
    fraud intelligence and conversational memory.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# USER QUERY
# ============================================================

query = st.chat_input(
    "Describe your banking issue..."
)


if query:

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    graph_input = {
        "query": query,
    }

    if transaction is not None:
        graph_input["transaction"] = transaction

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
        }
    }

    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing your request..."
        ):

            try:

                result = graph.invoke(
                    graph_input,
                    config=config,
                )

                st.session_state.last_result = result

                response = result.get(
                    "response",
                    "Unable to generate a response.",
                )

                st.markdown(response)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )

            except Exception as exc:

                st.error(
                    "The support workflow could not complete "
                    "this request."
                )

                with st.expander(
                    "Technical details",
                    expanded=False,
                ):
                    st.exception(exc)


# ============================================================
# EXPLAINABILITY
# ============================================================

result = st.session_state.last_result

if result:

    st.divider()

    st.subheader(
        "🔎 Explainability"
    )

    analysis_tab, knowledge_tab, technical_tab = (
        st.tabs(
            [
                "📊 Decision Analysis",
                "📚 Retrieved Knowledge",
                "⚙️ Technical Details",
            ]
        )
    )

    # --------------------------------------------------------
    # DECISION ANALYSIS
    # --------------------------------------------------------

    with analysis_tab:

        st.markdown(
            "#### Request Classification"
        )

        display_metric_row(
            [
                (
                    "Final Intent",
                    result.get(
                        "intent",
                        "N/A",
                    ),
                ),
                (
                    "ML Intent",
                    result.get(
                        "model_intent",
                        "N/A",
                    ),
                ),
            ]
        )

        display_metric_row(
            [
                (
                    "Sentiment",
                    result.get(
                        "sentiment",
                        "N/A",
                    ),
                ),
                (
                    "Urgency",
                    result.get(
                        "urgency",
                        "N/A",
                    ),
                ),
            ]
        )

        st.markdown(
            "#### Fraud Assessment"
        )

        fraud_probability = result.get(
            "fraud_probability"
        )

        fraud_prediction = result.get(
            "fraud_prediction"
        )

        risk_level = result.get(
            "risk_level"
        )

        escalation = (
            "Required"
            if result.get("escalate")
            else "Not required"
        )

        display_metric_row(
            [
                (
                    "Fraud Probability",
                    format_probability(
                        fraud_probability
                    ),
                ),
                (
                    "Fraud Prediction",
                    (
                        "Not assessed"
                        if fraud_prediction is None
                        else str(fraud_prediction)
                    ),
                ),
            ]
        )

        display_metric_row(
            [
                (
                    "Risk Level",
                    (
                        "Not assessed"
                        if risk_level is None
                        else risk_level
                    ),
                ),
                (
                    "Escalation",
                    escalation,
                ),
            ]
        )

        if fraud_probability is None:

            st.caption(
                "Transaction-level fraud scoring was not performed "
                "because the required transaction features were "
                "not fully available."
            )

        st.markdown(
            "#### Suggested Actions"
        )

        display_actions(
            result.get(
                "suggested_action"
            )
        )

        st.markdown(
            "#### Routing"
        )

        st.info(
            result.get(
                "routing_reason",
                "No routing information available.",
            )
        )

        st.markdown(
            "#### Intent Decision"
        )

        if result.get(
            "fraud_override",
            False,
        ):
            st.warning(
                "Fraud safety rule triggered."
            )
        else:
            st.success(
                "No fraud safety override required."
            )

        st.caption(
            "Intent source: "
            + str(
                result.get(
                    "intent_source",
                    "N/A",
                )
            )
        )

    # --------------------------------------------------------
    # RETRIEVED KNOWLEDGE
    # --------------------------------------------------------

    with knowledge_tab:

        policy_results = result.get(
            "policy_results",
            [],
        )

        historical_results = result.get(
            "historical_results",
            [],
        )

        st.markdown(
            f"#### 📘 Authoritative Policies ({len(policy_results)})"
        )

        if not policy_results:

            st.info(
                "No authoritative policy was retrieved."
            )

        else:

            for index, item in enumerate(
                policy_results,
                start=1,
            ):

                source = item.get(
                    "source",
                    "Unknown",
                )

                similarity = item.get(
                    "similarity_score",
                    "N/A",
                )

                with st.expander(
                    f"{index}. {source}",
                    expanded=(index == 1),
                ):

                    st.caption(
                        f"Similarity: {similarity}"
                    )

                    st.write(
                        item.get(
                            "text",
                            "",
                        )
                    )

        st.markdown(
            f"#### 📂 Historical Support Cases ({len(historical_results)})"
        )

        if not historical_results:

            st.info(
                "No similar historical cases were retrieved."
            )

        else:

            st.caption(
                "Historical tickets are examples only; "
                "they are not authoritative policy."
            )

            for index, item in enumerate(
                historical_results,
                start=1,
            ):

                category = item.get(
                    "category",
                    "N/A",
                )

                similarity = item.get(
                    "similarity_score",
                    "N/A",
                )

                with st.expander(
                    f"Historical Case {index}",
                    expanded=False,
                ):

                    st.caption(
                        f"Category: {category} | "
                        f"Similarity: {similarity}"
                    )

                    st.write(
                        item.get(
                            "text",
                            "",
                        )
                    )

    # --------------------------------------------------------
    # TECHNICAL DETAILS
    # --------------------------------------------------------

    with technical_tab:

        st.markdown(
            "#### LangGraph Execution Path"
        )

        graph_path = result.get(
            "graph_path",
            [],
        )

        if graph_path:
            st.code(
                " → ".join(graph_path),
                language="text",
            )
        else:
            st.info(
                "Graph path unavailable."
            )

        st.markdown(
            "#### RAG Context"
        )

        with st.expander(
            "View context supplied to the LLM",
            expanded=False,
        ):
            st.text(
                result.get(
                    "rag_context",
                    "",
                )
            )

        st.markdown(
            "#### Runtime State"
        )

        runtime_state = {
            key: result.get(key)
            for key in [
                "intent",
                "model_intent",
                "fraud_override",
                "intent_source",
                "sentiment",
                "urgency",
                "handling_priority",
                "fraud_probability",
                "fraud_prediction",
                "risk_level",
                "escalate",
                "suggested_action",
            ]
        }

        st.json(runtime_state)


# ============================================================
# EVALUATION
# ============================================================

st.divider()

st.subheader(
    "📈 System Evaluation"
)

st.caption(
    "Validation uses the QA benchmark. Response semantic "
    "similarity measures alignment with reference answers; "
    "it is not a factual accuracy score."
)

evaluation_available = (
    not retrieval_eval_df.empty
    or not response_eval_df.empty
)

if not evaluation_available:

    st.warning(
        "Evaluation results are not available. "
        "Run the evaluation notebook and save the results "
        "to the data folder."
    )

else:

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    st.markdown(
        "#### Retrieval Performance"
    )

    if retrieval_eval_df.empty:

        st.info(
            "Retrieval evaluation results are unavailable."
        )

    else:

        hit_at_1 = retrieval_eval_df[
            "hit_at_1"
        ].mean()

        hit_at_3 = retrieval_eval_df[
            "hit_at_3"
        ].mean()

        mrr = retrieval_eval_df[
            "reciprocal_rank"
        ].mean()

        mean_top1 = retrieval_eval_df[
            "top1_similarity"
        ].mean()

        mean_relevant = retrieval_eval_df[
            "relevant_similarity"
        ].mean()

        display_metric_row(
            [
                (
                    "Hit@1",
                    f"{hit_at_1:.2%}",
                ),
                (
                    "Hit@3",
                    f"{hit_at_3:.2%}",
                ),
            ]
        )

        display_metric_row(
            [
                (
                    "MRR",
                    f"{mrr:.4f}",
                ),
                (
                    "Policy Cases",
                    str(
                        len(retrieval_eval_df)
                    ),
                ),
            ]
        )

        display_metric_row(
            [
                (
                    "Mean Top-1 Similarity",
                    f"{mean_top1:.4f}",
                ),
                (
                    "Mean Relevant Similarity",
                    f"{mean_relevant:.4f}",
                ),
            ]
        )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    st.markdown(
        "#### Response Performance"
    )

    if response_eval_df.empty:

        st.info(
            "Response evaluation results are unavailable."
        )

    else:

        similarity = response_eval_df[
            "semantic_similarity"
        ]

        mean_similarity = similarity.mean()
        median_similarity = similarity.median()
        min_similarity = similarity.min()
        max_similarity = similarity.max()

        match_rate = (
            similarity >= 0.75
        ).mean()

        display_metric_row(
            [
                (
                    "Mean Semantic Similarity",
                    f"{mean_similarity:.2%}",
                ),
                (
                    "Median Semantic Similarity",
                    f"{median_similarity:.2%}",
                ),
            ]
        )

        display_metric_row(
            [
                (
                    "Similarity ≥ 0.75",
                    f"{match_rate:.2%}",
                ),
                (
                    "Response Cases",
                    str(
                        len(response_eval_df)
                    ),
                ),
            ]
        )

        display_metric_row(
            [
                (
                    "Minimum Similarity",
                    f"{min_similarity:.4f}",
                ),
                (
                    "Maximum Similarity",
                    f"{max_similarity:.4f}",
                ),
            ]
        )

        # ----------------------------------------------------
        # CATEGORY PERFORMANCE
        # ----------------------------------------------------

        if "category" in response_eval_df.columns:

            st.markdown(
                "#### Response Performance by Category"
            )

            category_eval = (
                response_eval_df
                .groupby("category")[
                    "semantic_similarity"
                ]
                .agg(
                    [
                        "count",
                        "mean",
                        "median",
                        "min",
                        "max",
                    ]
                )
                .reset_index()
            )

            category_eval.columns = [
                "Category",
                "Cases",
                "Mean Similarity",
                "Median Similarity",
                "Minimum",
                "Maximum",
            ]

            st.dataframe(
                category_eval.round(4),
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # LOWEST SCORING CASES
        # ----------------------------------------------------

        st.markdown(
            "#### Lowest Semantic Similarity Cases"
        )

        weakest_columns = [
            column
            for column in [
                "id",
                "category",
                "question",
                "semantic_similarity",
            ]
            if column in response_eval_df.columns
        ]

        if weakest_columns:

            weakest = (
                response_eval_df
                .sort_values(
                    "semantic_similarity"
                )
                [weakest_columns]
                .head(5)
                .copy()
            )

            weakest[
                "semantic_similarity"
            ] = weakest[
                "semantic_similarity"
            ].round(4)

            st.dataframe(
                weakest,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Powered Banking Support & Fraud Intelligence System"
    " • LangGraph • FAISS • SQLite • Gemini"
)
