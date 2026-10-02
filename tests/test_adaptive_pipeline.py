from jevretrieve.evaluator import EvidenceEvaluator
from jevretrieve.pipeline import JevRetrieve
from jevretrieve.retrievers import BM25Retriever, RetrievedDocument


def test_adaptive_pipeline():

    documents = [
        RetrievedDocument(
            text=(
                "Ada Lovelace is widely regarded as the first "
                "computer programmer."
            ),
            score=0.0,
            document_id="doc_1",
        ),
        RetrievedDocument(
            text=(
                "Ada Lovelace worked on Charles Babbage's "
                "Analytical Engine."
            ),
            score=0.0,
            document_id="doc_2",
        ),
        RetrievedDocument(
            text=(
                "Lovelace was born in London in 1815."
            ),
            score=0.0,
            document_id="doc_3",
        ),
        RetrievedDocument(
            text=(
                "The Analytical Engine was a mechanical "
                "general-purpose computer designed by Babbage."
            ),
            score=0.0,
            document_id="doc_4",
        ),
    ]

    retriever = BM25Retriever(documents)
    evaluator = EvidenceEvaluator()

    pipeline = JevRetrieve(
        retriever=retriever,
        evaluator=evaluator,
        top_k=2,
        max_iterations=2,
    )

    result = pipeline.run(
        "What was Ada Lovelace's connection to the Analytical Engine?"
    )

    assert result.query == (
        "What was Ada Lovelace's connection to the Analytical Engine?"
    )

    assert result.iterations >= 1
    assert result.retrieval_calls >= 1
    assert result.jev_calls >= 1

    assert len(result.documents) >= 1

    assert 0.0 <= result.final_assessment.sufficiency <= 1.0
    assert 0.0 <= result.final_assessment.coverage <= 1.0

    print("\nAdaptive pipeline result:")
    print("Iterations:", result.iterations)
    print("Retrieval calls:", result.retrieval_calls)
    print("Jev calls:", result.jev_calls)
    print("Documents:", len(result.documents))
    print(
        "Missing information:",
        result.final_assessment.missing_information,
    )