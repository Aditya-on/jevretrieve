from dataclasses import dataclass, field

from .controller import (
    ControllerDecision,
    CostAwareController,
    RetrievalAction,
)
from .cost_tracker import CostTracker
from .diversifier import MMRAssessor
from .evaluator import EvidenceEvaluator
from .models import EvidenceAssessment
from .query_expansion import QueryExpander
from .retrievers import BaseRetriever, RetrievedDocument


@dataclass
class JevRetrieveResult:
    """Final result produced by JevRetrieve."""

    query: str
    documents: list[RetrievedDocument]

    final_assessment: EvidenceAssessment

    iterations: int

    cost_tracker: CostTracker

    decisions: list[ControllerDecision] = field(
        default_factory=list
    )

    @property
    def retrieval_calls(self) -> int:
        """Number of retrieval operations."""

        return self.cost_tracker.retrieval_calls

    @property
    def jev_calls(self) -> int:
        """Number of Jev API calls."""

        return self.cost_tracker.jev_calls

    @property
    def input_tokens(self) -> int:
        """Total Jev input tokens."""

        return self.cost_tracker.input_tokens

    @property
    def output_tokens(self) -> int:
        """Total Jev output tokens."""

        return self.cost_tracker.output_tokens

    @property
    def total_jev_tokens(self) -> int:
        """Total tokens consumed by Jev."""

        return self.cost_tracker.total_tokens

    @property
    def total_cost(self) -> float:
        """Total estimated retrieval + Jev cost."""

        return self.cost_tracker.total_cost


class JevRetrieve:
    """
    Cost-aware iterative retrieval system.

    V1 supports:
    - Initial retrieval
    - Jev evidence assessment
    - Cost-aware decisions
    - Query expansion
    - Iterative retrieval
    - MMR diversification
    - Cost and token tracking
    - Decision trace recording
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        evaluator: EvidenceEvaluator | None = None,
        controller: CostAwareController | None = None,
        query_expander: QueryExpander | None = None,
        diversifier: MMRAssessor | None = None,
        max_iterations: int | None = None,
        top_k: int = 5,
    ):
        if max_iterations is not None and max_iterations < 1:
            raise ValueError(
                "max_iterations must be at least 1."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        self.retriever = retriever

        self.evaluator = (
            evaluator or EvidenceEvaluator()
        )

        self.controller = (
            controller or CostAwareController()
        )

        self.query_expander = (
            query_expander or QueryExpander()
        )

        if diversifier is not None:
            self.diversifier = diversifier
        else:
            encoder = self._get_retriever_encoder()

            if encoder is not None:
                self.diversifier = MMRAssessor(
                    encoder=encoder
                )
            else:
                self.diversifier = MMRAssessor()

        self.max_iterations = max_iterations
        self.top_k = top_k

    def run(
        self,
        query: str,
    ) -> JevRetrieveResult:
        """Run iterative retrieval until Jev decides to stop."""

        documents: list[RetrievedDocument] = []

        decisions: list[ControllerDecision] = []

        cost_tracker = CostTracker()

        final_assessment = None

        current_query = query

        # Reset controller state for every query.
        self.controller.reset()

        iteration = 0

        while True:
            iteration += 1

            retrieval_k = self.top_k * iteration

            retrieved = self.retriever.retrieve(
                current_query,
                top_k=retrieval_k,
            )

            cost_tracker.record_retrieval()

            previous_document_ids = {
                document.document_id
                for document in documents
            }

            documents = self._merge_documents(
                documents,
                retrieved,
            )

            new_document_ids = {
                document.document_id
                for document in documents
            } - previous_document_ids

            # If retrieval produced no new evidence,
            # stop before spending another Jev call.
            if (
                iteration > 1
                and not new_document_ids
            ):
                break

            evidence = [
                document.text
                for document in documents
            ]

            # 🔴✓ API CALL — TypeSafe Jev
            assessment = self.evaluator.evaluate(
                query=query,
                evidence=evidence,
            )

            jev_cost = (
                assessment.input_tokens
                / 1_000_000
                * self.controller.jev_input_cost_per_million
            )

            cost_tracker.record_jev(
                input_tokens=assessment.input_tokens,
                output_tokens=assessment.output_tokens,
                cost=jev_cost,
            )

            final_assessment = assessment

            decision = self.controller.decide(
                assessment
            )

            decisions.append(decision)

            if decision.action == RetrievalAction.STOP:
                break

            if (
                self.max_iterations is not None
                and iteration >= self.max_iterations
            ):
                break

            if decision.action == RetrievalAction.QUERY_EXPAND:
                current_query = (
                    self.query_expander.expand(
                        original_query=query,
                        assessment=assessment,
                    )
                )
                continue

            if decision.action == RetrievalAction.RETRIEVE_MORE:
                continue

            if decision.action == RetrievalAction.DIVERSIFY:
                diversified = (
                    self.diversifier.diversify(
                        query=current_query,
                        documents=documents,
                        top_k=min(
                            self.top_k,
                            len(documents),
                        ),
                    )
                )

                diversified_ids = {
                    document.document_id
                    for document in diversified
                }

                # Keep all previously collected evidence.
                # MMR only changes the priority/order of
                # documents rather than deleting them.
                remaining = [
                    document
                    for document in documents
                    if document.document_id
                    not in diversified_ids
                ]

                documents = (
                    diversified
                    + remaining
                )

                continue

            break

        if final_assessment is None:
            raise RuntimeError(
                "JevRetrieve produced no evidence assessment."
            )

        return JevRetrieveResult(
            query=query,
            documents=documents,
            final_assessment=final_assessment,
            iterations=iteration,
            cost_tracker=cost_tracker,
            decisions=decisions,
        )

    def _get_retriever_encoder(self):
        """Return the retriever's shared encoder when available."""

        if hasattr(self.retriever, "encoder"):
            return self.retriever.encoder

        if hasattr(self.retriever, "dense"):
            dense_retriever = self.retriever.dense

            if hasattr(dense_retriever, "encoder"):
                return dense_retriever.encoder

        return None

    @staticmethod
    def _merge_documents(
        existing: list[RetrievedDocument],
        new_documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Merge documents while removing duplicates."""

        documents_by_id = {
            document.document_id: document
            for document in existing
        }

        for document in new_documents:
            documents_by_id[document.document_id] = document

        return list(documents_by_id.values())