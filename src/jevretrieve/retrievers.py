from abc import ABC, abstractmethod
from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedDocument:
    """A document returned by a retriever."""

    text: str
    score: float
    document_id: str


class BaseRetriever(ABC):
    """Common interface for all JevRetrieve retrievers."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        """Retrieve the top-k documents for a query."""
        raise NotImplementedError


class BM25Retriever(BaseRetriever):
    """BM25 lexical retriever."""

    def __init__(
        self,
        documents: list[RetrievedDocument],
    ):
        self.documents = documents

        self.tokenized_documents = [
            self._tokenize(document.text)
            for document in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace tokenizer."""
        return text.lower().split()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedDocument]:

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            RetrievedDocument(
                text=self.documents[i].text,
                score=float(scores[i]),
                document_id=self.documents[i].document_id,
            )
            for i in ranked_indices
        ]


class DenseRetriever(BaseRetriever):
    """Dense vector retriever using Sentence Transformers and FAISS."""

    def __init__(
        self,
        documents: list[RetrievedDocument],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.documents = documents

        self.encoder = SentenceTransformer(model_name)

        texts = [document.text for document in documents]

        embeddings = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedDocument]:

        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, len(self.documents)),
        )

        return [
            RetrievedDocument(
                text=self.documents[index].text,
                score=float(score),
                document_id=self.documents[index].document_id,
            )
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]


class HybridRetriever(BaseRetriever):
    """Combine BM25 and dense retrieval using weighted score fusion."""

    def __init__(
        self,
        documents: list[RetrievedDocument],
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dense_weight: float = 0.5,
    ):
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError("dense_weight must be between 0 and 1.")

        self.documents = documents
        self.dense_weight = dense_weight
        self.bm25_weight = 1.0 - dense_weight

        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever(
            documents,
            model_name=dense_model,
        )

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores to [0, 1]."""

        minimum = scores.min()
        maximum = scores.max()

        if maximum == minimum:
            return np.zeros_like(scores)

        return (scores - minimum) / (maximum - minimum)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedDocument]:

        query_tokens = self.bm25._tokenize(query)

        bm25_scores = self.bm25.bm25.get_scores(query_tokens)

        query_embedding = self.dense.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        dense_scores, _ = self.dense.index.search(
            query_embedding,
            len(self.documents),
        )

        dense_scores = dense_scores[0]

        bm25_normalized = self._normalize(
            np.asarray(bm25_scores, dtype="float32")
        )

        dense_normalized = self._normalize(
            dense_scores
        )

        hybrid_scores = (
            self.bm25_weight * bm25_normalized
            + self.dense_weight * dense_normalized
        )

        ranked_indices = np.argsort(
            hybrid_scores
        )[::-1][:top_k]

        return [
            RetrievedDocument(
                text=self.documents[i].text,
                score=float(hybrid_scores[i]),
                document_id=self.documents[i].document_id,
            )
            for i in ranked_indices
        ]