from pathlib import Path
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root:
# AI-Powered-Banking-Support-Fraud-Intelligence-System/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store"

POLICY_INDEX_PATH = VECTOR_STORE_PATH / "policy.index"
POLICY_METADATA_PATH = VECTOR_STORE_PATH / "policy_metadata.pkl"

TICKET_INDEX_PATH = VECTOR_STORE_PATH / "support_tickets.index"
TICKET_METADATA_PATH = VECTOR_STORE_PATH / "support_ticket_metadata.pkl"


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# ============================================================
# LOAD POLICY VECTOR STORE
# ============================================================

if not POLICY_INDEX_PATH.exists():
    raise FileNotFoundError(
        f"Policy FAISS index not found:\n{POLICY_INDEX_PATH}"
    )

if not POLICY_METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Policy metadata not found:\n{POLICY_METADATA_PATH}"
    )

loaded_index = faiss.read_index(
    str(POLICY_INDEX_PATH)
)

with open(
    POLICY_METADATA_PATH,
    "rb"
) as f:
    loaded_metadata = pickle.load(f)


# ============================================================
# LOAD HISTORICAL SUPPORT-TICKET VECTOR STORE
# ============================================================

if not TICKET_INDEX_PATH.exists():
    raise FileNotFoundError(
        f"Support-ticket FAISS index not found:\n{TICKET_INDEX_PATH}"
    )

if not TICKET_METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Support-ticket metadata not found:\n{TICKET_METADATA_PATH}"
    )

loaded_ticket_index = faiss.read_index(
    str(TICKET_INDEX_PATH)
)

with open(
    TICKET_METADATA_PATH,
    "rb"
) as f:
    loaded_ticket_metadata = pickle.load(f)


# ============================================================
# INTENT → POLICY SOURCE MAPPING
# ============================================================

INTENT_POLICY_MAP = {
    "Fraud/Unauthorized": [
        "fraud_handling_policy.txt",
        "refund_dispute_policy.txt"
    ],
    "KYC": [
        "kyc_policy.txt"
    ],
    "Loan": [
        "loan_processing_policy.txt"
    ],
    "Account Access": [
        "account_access_policy.txt"
    ]
}

# ============================================================
# INTENT → HISTORICAL TICKET CATEGORY
# ============================================================

INTENT_TICKET_CATEGORY_MAP = {

    "Fraud/Unauthorized": "Fraud/Unauthorized",

    "KYC": "KYC",

    "Loan": "Loan",

    "Account Access": "Account Access"
}


# ============================================================
# EMBED QUERY
# ============================================================

def embed_query(query: str) -> np.ndarray:
    """
    Convert a user query into a normalized BGE embedding.

    Returns
    -------
    np.ndarray
        Shape: (1, 384)
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

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    return np.asarray(
        query_embedding,
        dtype="float32"
    )


# ============================================================
# POLICY RETRIEVER
# ============================================================

def retrieve_policy(
    query: str,
    top_k: int = 3,
    allowed_sources=None
):
    """
    Retrieve relevant policy chunks.

    Parameters
    ----------
    query : str
        Customer question.

    top_k : int
        Number of policy chunks to return.

    allowed_sources : list[str] | None
        Optional list of policy filenames.
        Used for intent-aware retrieval.

    Returns
    -------
    list[dict]
    """

    if top_k <= 0:
        return []

    query_embedding = embed_query(
        query
    )

    # Search all vectors first.
    # This allows us to filter by source afterward.
    search_k = loaded_index.ntotal

    scores, indices = loaded_index.search(
        query_embedding,
        search_k
    )

    results = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index == -1:
            continue

        chunk = loaded_metadata[index]

        # Intent/source filter
        if (
            allowed_sources is not None
            and chunk["source"] not in allowed_sources
        ):
            continue

        results.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "similarity_score": round(
                float(score),
                4
            ),
            "text": chunk["text"]
        })

        if len(results) >= top_k:
            break

    return results


# ============================================================
# HISTORICAL CASE RETRIEVER
# ============================================================

def retrieve_historical_cases(
    query: str,
    intent: str | None = None,
    top_k: int = 3
):
    """
    Retrieve similar historical support cases.

    Parameters
    ----------
    query : str
        Customer question.

    intent : str | None
        Optional detected banking intent.
        When provided, retrieval is restricted to
        that historical ticket category.

    top_k : int
        Number of historical cases to return.

    Returns
    -------
    list[dict]
    """

    if top_k <= 0:
        return []

    query_embedding = embed_query(
        query
    )

    search_k = loaded_ticket_index.ntotal

    scores, indices = loaded_ticket_index.search(
        query_embedding,
        search_k
    )

    results = []

    target_category = None

    if intent is not None:
        target_category = (
            INTENT_TICKET_CATEGORY_MAP.get(
                intent
            )
        )

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index == -1:
            continue

        case = loaded_ticket_metadata[index]

        # Intent/category filtering
        if (
            target_category is not None
            and case["category"] != target_category
        ):
            continue

        results.append({
            "chunk_id": case["chunk_id"],
            "source": case["source"],
            "category": case["category"],
            "similarity_score": round(
                float(score),
                4
            ),
            "text": case["text"]
        })

        if len(results) >= top_k:
            break

    return results


# ============================================================
# UNIFIED RETRIEVAL
# ============================================================

def retrieve_context(
    query: str,
    intent: str,
    policy_top_k: int = 3,
    history_top_k: int = 2
):
    """
    Retrieve both authoritative policy information
    and similar historical support cases.
    """

    policy_sources = INTENT_POLICY_MAP.get(
        intent,
        []
    )

    # --------------------------------------------------------
    # Policy retrieval
    # --------------------------------------------------------

    policy_results = []

    if policy_sources:

        policy_results = retrieve_policy(
            query=query,
            top_k=policy_top_k,
            allowed_sources=policy_sources
        )

    # --------------------------------------------------------
    # Historical retrieval
    # --------------------------------------------------------

    historical_results = retrieve_historical_cases(
        query=query,
        intent=intent,
        top_k=history_top_k
    )

    return {
        "policy_results": policy_results,
        "historical_results": historical_results
    }


# ============================================================
# BUILD LLM CONTEXT
# ============================================================

def build_rag_context(
    query: str,
    intent: str,
    context: dict
):
    """
    Convert retrieved policy and historical results
    into an LLM-ready context block.

    Policy information is explicitly marked as authoritative.
    Historical cases are explicitly marked as examples.
    """

    context_parts = []

    # --------------------------------------------------------
    # AUTHORITATIVE POLICY
    # --------------------------------------------------------

    policy_results = context.get(
        "policy_results",
        []
    )

    if policy_results:

        policy_text = "\n\n".join(
            [
                (
                    f"[Policy Source: {result['source']}]\n"
                    f"{result['text']}"
                )
                for result in policy_results
            ]
        )

        context_parts.append(
            "AUTHORITATIVE POLICY INFORMATION:\n"
            + policy_text
        )

    # --------------------------------------------------------
    # HISTORICAL CASES
    # --------------------------------------------------------

    historical_results = context.get(
        "historical_results",
        []
    )

    if historical_results:

        history_text = "\n\n".join(
            [
                (
                    f"[Historical Case: {result['chunk_id']}]\n"
                    f"{result['text']}"
                )
                for result in historical_results
            ]
        )

        context_parts.append(
            "SIMILAR HISTORICAL SUPPORT CASES "
            "(EXAMPLES ONLY — NOT POLICY):\n"
            + history_text
        )

    # --------------------------------------------------------
    # Final context
    # --------------------------------------------------------

    if not context_parts:
        return (
            "NO AUTHORITATIVE OR HISTORICAL INFORMATION "
            "WAS RETRIEVED."
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_policy_count():
    """Return number of policy vectors."""

    return loaded_index.ntotal


def get_historical_case_count():
    """Return number of historical ticket vectors."""

    return loaded_ticket_index.ntotal