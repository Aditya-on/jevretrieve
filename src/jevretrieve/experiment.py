from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .pipeline import JevRetrieve


@dataclass
class QueryEvaluation:
    """Recorded result for one evaluation query."""

    query: str

    iterations: int
    retrieval_calls: int
    jev_calls: int

    input_tokens: int
    output_tokens: int
    total_jev_tokens: int

    retrieval_cost: float
    jev_cost: float
    total_cost: float

    final_sufficiency: float
    final_relevance: float
    final_coverage: float
    final_redundancy: float
    missing_information: str

    actions: list[str]
    utilities: list[float]
    estimated_costs: list[float]
    expected_gains: list[float]

    document_count: int


class ExperimentRunner:
    """Run JevRetrieve over a collection of evaluation queries."""

    def __init__(self, pipeline: JevRetrieve):
        self.pipeline = pipeline

    def run(
        self,
        queries: list[str],
    ) -> list[QueryEvaluation]:
        """Evaluate all supplied queries."""

        results: list[QueryEvaluation] = []

        for query in queries:
            result = self.pipeline.run(query)

            results.append(
                QueryEvaluation(
                    query=query,
                    iterations=result.iterations,
                    retrieval_calls=result.retrieval_calls,
                    jev_calls=result.jev_calls,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    total_jev_tokens=result.total_jev_tokens,
                    retrieval_cost=(
                        result.cost_tracker.retrieval_cost
                    ),
                    jev_cost=result.cost_tracker.jev_cost,
                    total_cost=result.total_cost,
                    final_sufficiency=(
                        result.final_assessment.sufficiency
                    ),
                    final_relevance=(
                        result.final_assessment.relevance
                    ),
                    final_coverage=(
                        result.final_assessment.coverage
                    ),
                    final_redundancy=(
                        result.final_assessment.redundancy
                    ),
                    missing_information=(
                        result.final_assessment.missing_information
                    ),
                    actions=[
                        decision.action.value
                        for decision in result.decisions
                    ],
                    utilities=[
                        decision.utility
                        for decision in result.decisions
                    ],
                    estimated_costs=[
                        decision.estimated_cost
                        for decision in result.decisions
                    ],
                    expected_gains=[
                        decision.expected_gain
                        for decision in result.decisions
                    ],
                    document_count=len(result.documents),
                )
            )

        return results

    @staticmethod
    def save_json(
        results: list[QueryEvaluation],
        path: str | Path,
    ) -> None:
        """Save experiment results as JSON."""

        output = [
            asdict(result)
            for result in results
        ]

        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination.write_text(
            json.dumps(
                output,
                indent=2,
            ),
            encoding="utf-8",
        )