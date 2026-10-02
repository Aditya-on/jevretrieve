from jevretrieve.controller import (
    CostAwareController,
    RetrievalAction,
)
from jevretrieve.models import EvidenceAssessment


def make_assessment(
    sufficiency: float,
    relevance: float,
    coverage: float,
    redundancy: float,
    missing_information: str,
) -> EvidenceAssessment:
    return EvidenceAssessment(
        sufficiency=sufficiency,
        relevance=relevance,
        coverage=coverage,
        redundancy=redundancy,
        missing_information=missing_information,
        input_tokens=100,
        output_tokens=20,
    )


def test_controller_stops_when_evidence_is_sufficient():
    controller = CostAwareController()

    assessment = make_assessment(
        sufficiency=0.90,
        relevance=0.95,
        coverage=0.90,
        redundancy=0.10,
        missing_information="NONE",
    )

    decision = controller.decide(assessment)

    assert decision.action == RetrievalAction.STOP
    assert decision.expected_gain == 0.0


def test_controller_retrieves_more_when_coverage_is_low():
    controller = CostAwareController()

    assessment = make_assessment(
        sufficiency=0.30,
        relevance=0.80,
        coverage=0.30,
        redundancy=0.10,
        missing_information="Important evidence is missing.",
    )

    decision = controller.decide(assessment)

    assert decision.action in {
        RetrievalAction.RETRIEVE_MORE,
        RetrievalAction.QUERY_EXPAND,
    }


def test_controller_diversifies_when_redundancy_is_high():
    controller = CostAwareController()

    assessment = make_assessment(
        sufficiency=0.40,
        relevance=0.90,
        coverage=0.70,
        redundancy=0.90,
        missing_information="NONE",
    )

    decision = controller.decide(assessment)

    assert decision.action == RetrievalAction.DIVERSIFY


def test_controller_continues_below_quality_floor():
    controller = CostAwareController(
        minimum_sufficiency=0.85,
        min_gain=0.02,
    )

    assessment = make_assessment(
        sufficiency=0.70,
        relevance=0.90,
        coverage=0.60,
        redundancy=0.20,
        missing_information="NONE",
    )

    decision = controller.decide(assessment)

    assert decision.action != RetrievalAction.STOP
    assert decision.expected_gain >= 0.02


def test_controller_stops_at_quality_floor():
    controller = CostAwareController(
        minimum_sufficiency=0.85,
    )

    assessment = make_assessment(
        sufficiency=0.90,
        relevance=0.90,
        coverage=0.90,
        redundancy=0.20,
        missing_information="NONE",
    )

    decision = controller.decide(assessment)

    assert decision.action == RetrievalAction.STOP
    assert decision.expected_gain == 0.0