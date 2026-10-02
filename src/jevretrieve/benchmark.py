from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .diversifier import MMRAssessor
from .evaluator import EvidenceEvaluator
from .pipeline import JevRetrieve
from .retrievers import BaseRetriever


@dataclass
class BenchmarkResult:
    """Result for one system on one query."""

    system: str
    query: str

    document_count: int

    jev_calls: int
    retrieval_calls: int

    input_tokens: int
    output_tokens: int
    total_jev_tokens: int

    estimated_cost: float

    sufficiency: float
    relevance: float
    coverage: float
    redundancy: float

    actions: list[str]


class BenchmarkRunner:
    """
    Run retrieval systems under a common evaluation protocol.

    Static baselines are evaluated at a fixed document budget.
    Fixed-Jev uses a fixed number of retrieval/evaluation rounds.
    JevRetrieve remains adaptive.
    """

    def __init__(
        self,
        evaluator: EvidenceEvaluator | None = None,
        jev_cost_per_million: float = 0.042,
    ):
        self.evaluator = evaluator or EvidenceEvaluator()
        self.jev_cost_per_million = jev_cost_per_million

    def run_baseline(
        self,
        name: str,
        retriever: BaseRetriever,
        queries: list[str],
        top_k: int = 5,
        use_mmr: bool = False,
    ) -> list[BenchmarkResult]:
        """Run a fixed-budget retrieval baseline."""

        results: list[BenchmarkResult] = []

        diversifier = None

        if use_mmr:
            encoder = self._get_encoder(retriever)

            if encoder is None:
                diversifier = MMRAssessor()
            else:
                diversifier = MMRAssessor(
                    encoder=encoder
                )

        for query in queries:
            documents = retriever.retrieve(
                query,
                top_k=top_k,
            )

            if diversifier is not None:
                documents = diversifier.diversify(
                    query=query,
                    documents=documents,
                    top_k=top_k,
                )

            assessment = self.evaluator.evaluate(
                query=query,
                evidence=[
                    document.text
                    for document in documents
                ],
            )

            results.append(
                self._make_result(
                    system=name,
                    query=query,
                    document_count=len(documents),
                    retrieval_calls=1,
                    assessment=assessment,
                    actions=["EVALUATE"],
                )
            )

        return results

    def run_fixed_jev(
        self,
        retriever: BaseRetriever,
        queries: list[str],
        top_k: int = 2,
        iterations: int = 3,
    ) -> list[BenchmarkResult]:
        """
        Run a fixed iterative Jev baseline.

        Every query receives exactly `iterations` retrieval
        and Jev evaluation rounds unless the retriever has
        exhausted its unique document set.
        """

        results: list[BenchmarkResult] = []

        for query in queries:
            documents = []
            input_tokens = 0
            output_tokens = 0
            jev_calls = 0
            retrieval_calls = 0

            for iteration in range(1, iterations + 1):
                retrieval_k = top_k * iteration

                retrieved = retriever.retrieve(
                    query,
                    top_k=retrieval_k,
                )

                retrieval_calls += 1

                documents = self._merge_documents(
                    documents,
                    retrieved,
                )

                assessment = self.evaluator.evaluate(
                    query=query,
                    evidence=[
                        document.text
                        for document in documents
                    ],
                )

                input_tokens += assessment.input_tokens
                output_tokens += assessment.output_tokens
                jev_calls += 1

            results.append(
                BenchmarkResult(
                    system="Fixed-Jev",
                    query=query,
                    document_count=len(documents),
                    jev_calls=jev_calls,
                    retrieval_calls=retrieval_calls,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_jev_tokens=(
                        input_tokens + output_tokens
                    ),
                    estimated_cost=(
                        input_tokens
                        / 1_000_000
                        * self.jev_cost_per_million
                    ),
                    sufficiency=assessment.sufficiency,
                    relevance=assessment.relevance,
                    coverage=assessment.coverage,
                    redundancy=assessment.redundancy,
                    actions=[
                        "RETRIEVE_MORE"
                        for _ in range(iterations - 1)
                    ] + ["STOP"],
                )
            )

        return results

    def run_jev_retrieve(
        self,
        pipeline: JevRetrieve,
        queries: list[str],
    ) -> list[BenchmarkResult]:
        """Run the adaptive JevRetrieve system."""

        results: list[BenchmarkResult] = []

        for query in queries:
            result = pipeline.run(query)

            results.append(
                BenchmarkResult(
                    system="JevRetrieve",
                    query=query,
                    document_count=len(result.documents),
                    jev_calls=result.jev_calls,
                    retrieval_calls=result.retrieval_calls,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    total_jev_tokens=result.total_jev_tokens,
                    estimated_cost=result.total_cost,
                    sufficiency=(
                        result.final_assessment.sufficiency
                    ),
                    relevance=(
                        result.final_assessment.relevance
                    ),
                    coverage=(
                        result.final_assessment.coverage
                    ),
                    redundancy=(
                        result.final_assessment.redundancy
                    ),
                    actions=[
                        decision.action.value
                        for decision in result.decisions
                    ],
                )
            )

        return results

    def _make_result(
        self,
        system: str,
        query: str,
        document_count: int,
        retrieval_calls: int,
        assessment,
        actions: list[str],
    ) -> BenchmarkResult:
        """Create a benchmark result from an assessment."""

        return BenchmarkResult(
            system=system,
            query=query,
            document_count=document_count,
            jev_calls=1,
            retrieval_calls=retrieval_calls,
            input_tokens=assessment.input_tokens,
            output_tokens=assessment.output_tokens,
            total_jev_tokens=assessment.total_tokens,
            estimated_cost=(
                assessment.input_tokens
                / 1_000_000
                * self.jev_cost_per_million
            ),
            sufficiency=assessment.sufficiency,
            relevance=assessment.relevance,
            coverage=assessment.coverage,
            redundancy=assessment.redundancy,
            actions=actions,
        )

    @staticmethod
    def _get_encoder(retriever: BaseRetriever):
        """Find a shared encoder on supported retrievers."""

        if hasattr(retriever, "encoder"):
            return retriever.encoder

        if hasattr(retriever, "dense"):
            dense = retriever.dense

            if hasattr(dense, "encoder"):
                return dense.encoder

        return None

    @staticmethod
    def _merge_documents(
        existing,
        new_documents,
    ):
        """Merge documents by document ID."""

        documents_by_id = {
            document.document_id: document
            for document in existing
        }

        for document in new_documents:
            documents_by_id[document.document_id] = document

        return list(documents_by_id.values())

    @staticmethod
    def save_json(
        results: list[BenchmarkResult],
        path: str | Path,
    ) -> None:
        """Save benchmark results as JSON."""

        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination.write_text(
            json.dumps(
                [asdict(result) for result in results],
                indent=2,
            ),
            encoding="utf-8",
        )