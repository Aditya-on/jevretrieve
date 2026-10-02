from sentence_transformers import SentenceTransformer

from .retrievers import RetrievedDocument


class MMRAssessor:
    """
    Maximal Marginal Relevance document diversifier.

    Balances relevance to the query against redundancy
    with already selected documents.
    """

    def __init__(
        self,
        encoder: SentenceTransformer | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        lambda_weight: float = 0.7,
    ):
        if not 0.0 <= lambda_weight <= 1.0:
            raise ValueError(
                "lambda_weight must be between 0 and 1."
            )

        self.lambda_weight = lambda_weight

        self.encoder = (
            encoder
            if encoder is not None
            else SentenceTransformer(model_name)
        )

    def diversify(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int,
    ) -> list[RetrievedDocument]:
        """Select a diverse subset using MMR."""

        if not documents:
            return []

        if top_k >= len(documents):
            return documents.copy()

        texts = [
            document.text
            for document in documents
        ]

        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        document_embeddings = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        relevance_scores = (
            document_embeddings @ query_embedding
        )

        selected_indices: list[int] = []
        remaining_indices = set(
            range(len(documents))
        )

        while (
            remaining_indices
            and len(selected_indices) < top_k
        ):
            if not selected_indices:
                best_index = max(
                    remaining_indices,
                    key=lambda i: relevance_scores[i],
                )
            else:

                def mmr_score(index: int) -> float:
                    redundancy = max(
                        document_embeddings[index]
                        @ document_embeddings[selected]
                        for selected in selected_indices
                    )

                    return (
                        self.lambda_weight
                        * relevance_scores[index]
                        - (
                            1.0 - self.lambda_weight
                        )
                        * redundancy
                    )

                best_index = max(
                    remaining_indices,
                    key=mmr_score,
                )

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

        return [
            documents[index]
            for index in selected_indices
        ]