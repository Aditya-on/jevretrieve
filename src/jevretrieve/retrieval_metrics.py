import math

from .retrievers import RetrievedDocument


def recall_at_k(
    documents: list[RetrievedDocument],
    gold_ids: set[str],
    k: int,
) -> float:
    """Fraction of gold documents retrieved in top-k."""

    if not gold_ids:
        return 0.0

    retrieved_ids = {
        document.document_id
        for document in documents[:k]
    }

    return len(
        retrieved_ids & gold_ids
    ) / len(gold_ids)


def mrr(
    documents: list[RetrievedDocument],
    gold_ids: set[str],
) -> float:
    """Mean reciprocal rank for the first gold document."""

    for rank, document in enumerate(
        documents,
        start=1,
    ):
        if document.document_id in gold_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    documents: list[RetrievedDocument],
    gold_ids: set[str],
    k: int,
) -> float:
    """Binary-relevance nDCG@k."""

    ranked = documents[:k]

    dcg = 0.0

    for rank, document in enumerate(
        ranked,
        start=1,
    ):
        if document.document_id in gold_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(
        len(gold_ids),
        k,
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1,
        )
    )

    return dcg / idcg


def evidence_recall(
    documents: list[RetrievedDocument],
    gold_ids: set[str],
) -> float:
    """Fraction of gold supporting documents retrieved."""

    if not gold_ids:
        return 0.0

    retrieved_ids = {
        document.document_id
        for document in documents
    }

    return len(
        retrieved_ids & gold_ids
    ) / len(gold_ids)