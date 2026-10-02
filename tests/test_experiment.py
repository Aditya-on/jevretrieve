from dataclasses import dataclass

from jevretrieve.controller import (
    ControllerDecision,
    RetrievalAction,
)
from jevretrieve.experiment import ExperimentRunner


@dataclass
class FakeAssessment:
    sufficiency: float = 0.90
    relevance: float = 0.80
    coverage: float = 0.85
    redundancy: float = 0.10
    missing_information: str = "NONE"


@dataclass
class FakeCostTracker:
    retrieval_cost: float = 0.0
    jev_cost: float = 0.0


@dataclass
class FakeResult:
    iterations: int
    retrieval_calls: int
    jev_calls: int
    input_tokens: int
    output_tokens: int
    total_jev_tokens: int
    total_cost: float
    final_assessment: FakeAssessment
    cost_tracker: FakeCostTracker
    decisions: list
    documents: list


class FakePipeline:
    def run(self, query: str) -> FakeResult:
        decision = ControllerDecision(
            action=RetrievalAction.STOP,
            utility=0.0,
            estimated_cost=0.0,
            expected_gain=0.0,
            reason="Evidence is sufficient.",
        )

        return FakeResult(
            iterations=1,
            retrieval_calls=1,
            jev_calls=1,
            input_tokens=100,
            output_tokens=20,
            total_jev_tokens=120,
            total_cost=0.0,
            final_assessment=FakeAssessment(),
            cost_tracker=FakeCostTracker(),
            decisions=[decision],
            documents=["document"],
        )


def test_experiment_runner():
    runner = ExperimentRunner(
        pipeline=FakePipeline()
    )

    results = runner.run(
        ["What is the answer?"]
    )

    assert len(results) == 1

    result = results[0]

    assert result.query == "What is the answer?"
    assert result.iterations == 1
    assert result.retrieval_calls == 1
    assert result.jev_calls == 1

    assert result.input_tokens == 100
    assert result.output_tokens == 20
    assert result.total_jev_tokens == 120

    assert result.final_sufficiency == 0.90
    assert result.final_coverage == 0.85

    assert result.actions == ["STOP"]
    assert result.utilities == [0.0]
    assert result.estimated_costs == [0.0]
    assert result.expected_gains == [0.0]