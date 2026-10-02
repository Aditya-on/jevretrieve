from jevretrieve.models import EvidenceAssessment
from jevretrieve.query_expansion import QueryExpander


def test_query_expansion():

    assessment = EvidenceAssessment(
        sufficiency=0.40,
        relevance=0.80,
        coverage=0.40,
        redundancy=0.10,
        missing_information="ENTITY",
        input_tokens=100,
        output_tokens=20,
    )

    expander = QueryExpander()

    expanded = expander.expand(
        "Who founded the company?",
        assessment,
    )

    assert expanded != "Who founded the company?"
    assert "entity" in expanded.lower()


def test_no_query_expansion_when_nothing_is_missing():

    assessment = EvidenceAssessment(
        sufficiency=0.95,
        relevance=0.95,
        coverage=0.95,
        redundancy=0.10,
        missing_information="NONE",
        input_tokens=100,
        output_tokens=20,
    )

    expander = QueryExpander()

    expanded = expander.expand(
        "What is the capital of France?",
        assessment,
    )

    assert expanded == "What is the capital of France?"
