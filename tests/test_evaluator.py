from jevretrieve.evaluator import EvidenceEvaluator


def test_evidence_evaluator():

    evaluator = EvidenceEvaluator()

    assessment = evaluator.evaluate(
        query="What is the capital of France?",
        evidence=[
            "Paris is the capital city of France.",
            "Paris is located in Europe.",
        ],
    )

    assert 0.0 <= assessment.sufficiency <= 1.0
    assert 0.0 <= assessment.relevance <= 1.0
    assert 0.0 <= assessment.coverage <= 1.0
    assert 0.0 <= assessment.redundancy <= 1.0

    assert assessment.input_tokens > 0
    assert assessment.output_tokens > 0

    print("\nEvidence assessment:")
    print("Sufficiency:", assessment.sufficiency)
    print("Relevance:", assessment.relevance)
    print("Coverage:", assessment.coverage)
    print("Redundancy:", assessment.redundancy)
    print("Missing:", assessment.missing_information)