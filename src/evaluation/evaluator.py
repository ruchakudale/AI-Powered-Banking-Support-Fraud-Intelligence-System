from pathlib import Path
import json
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

QA_PATH = PROJECT_ROOT / "data" / "qa_pairs.json"


# ============================================================
# EMBEDDING MODEL
# ============================================================

_embedding_model = None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

    return _embedding_model

def normalize_policy_reference(policy_ref):

    if not isinstance(policy_ref, str):
        return None

    ref = policy_ref.lower()

    if "fraud handling policy" in ref:
        return "fraud_handling_policy.txt"

    if "loan processing policy" in ref:
        return "loan_processing_policy.txt"

    if "kyc policy" in ref:
        return "kyc_policy.txt"

    if "refund" in ref and "dispute" in ref:
        return "refund_dispute_policy.txt"

    return None

# ============================================================
# LOAD QA DATA
# ============================================================

def load_qa_pairs():

    with open(
        QA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return pd.DataFrame(data)


# ============================================================
# RETRIEVAL EVALUATION
# ============================================================

def evaluate_retrieval(
    retriever_function,
    qa_df,
    top_k=3
):
    """
    Evaluate policy retrieval against the expected policy
    referenced by the QA benchmark.
    """

    rows = []

    for _, row in qa_df.iterrows():

        raw_policy_ref = row["policy_ref"]

        expected_policy = normalize_policy_reference(
            raw_policy_ref
        )

        # Skip questions for which the current policy corpus
        # does not contain an authoritative policy.
        if expected_policy is None:
            continue

        results = retriever_function(
            query=row["question"],
            intent=row["category"],
            top_k=top_k
        )

        sources = [
            item.get("source", "")
            for item in results
        ]

        similarities = [
            item.get(
                "similarity_score",
                np.nan
            )
            for item in results
        ]

        # ----------------------------------------------------
        # Find first relevant result
        # ----------------------------------------------------

        rank = None

        for i, source in enumerate(
            sources,
            start=1
        ):

            if source == expected_policy:

                rank = i
                break

        # ----------------------------------------------------
        # Retrieval metrics
        # ----------------------------------------------------

        hit_at_1 = (
            1
            if rank == 1
            else 0
        )

        hit_at_3 = (
            1
            if rank is not None and rank <= 3
            else 0
        )

        reciprocal_rank = (
            1 / rank
            if rank is not None
            else 0
        )

        top1_similarity = (
            similarities[0]
            if similarities
            else np.nan
        )

        relevant_similarity = (
            similarities[rank - 1]
            if rank is not None
            else np.nan
        )

        rows.append(
            {
                "id": row["id"],
                "category": row["category"],
                "question": row["question"],
                "expected_policy": expected_policy,
                "policy_reference": raw_policy_ref,
                "retrieved_sources": sources,
                "rank": rank,
                "hit_at_1": hit_at_1,
                "hit_at_3": hit_at_3,
                "reciprocal_rank": reciprocal_rank,
                "top1_similarity": top1_similarity,
                "relevant_similarity": relevant_similarity,
            }
        )

    result_df = pd.DataFrame(rows)

    if result_df.empty:

        return result_df, {}

    metrics = {
        "Hit@1": result_df["hit_at_1"].mean(),

        "Hit@3": result_df["hit_at_3"].mean(),

        "MRR": result_df["reciprocal_rank"].mean(),

        "Mean Top-1 Similarity": (
            result_df["top1_similarity"].mean()
        ),

        "Mean Relevant Similarity": (
            result_df["relevant_similarity"].mean()
        ),
    }

    return result_df, metrics

# ============================================================
# RESPONSE SEMANTIC EVALUATION
# ============================================================

def evaluate_response_similarity(
    questions,
    reference_answers,
    generated_answers
):
    """
    Compare generated answers with reference answers
    using BGE-small embeddings and cosine similarity.
    """

    model = get_embedding_model()

    reference_embeddings = model.encode(
        reference_answers,
        normalize_embeddings=True
    )

    generated_embeddings = model.encode(
        generated_answers,
        normalize_embeddings=True
    )

    similarities = np.sum(
        reference_embeddings * generated_embeddings,
        axis=1
    )

    result_df = pd.DataFrame(
        {
            "question": questions,
            "reference_answer": reference_answers,
            "generated_answer": generated_answers,
            "semantic_similarity": similarities,
        }
    )

    metrics = {
        "Mean Similarity": float(
            similarities.mean()
        ),
        "Median Similarity": float(
            np.median(similarities)
        ),
        "Minimum Similarity": float(
            similarities.min()
        ),
        "Maximum Similarity": float(
            similarities.max()
        ),
        "Similarity >= 0.75": float(
            np.mean(similarities >= 0.75)
        ),
    }

    return result_df, metrics