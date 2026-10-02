from jevretrieve.controller import (
    CostAwareController,
)
from jevretrieve.evaluator import EvidenceEvaluator
from jevretrieve.pipeline import JevRetrieve
from jevretrieve.retrievers import (
    BM25Retriever,
    RetrievedDocument,
)


def test_jev_retrieve_pipeline():

    documents = [
        RetrievedDocument(
            text="Paris is the capital city of France.",
            score=0.0,
            document_id="doc_1",
        ),
        RetrievedDocument(
            text="France is a country in Western Europe.",
            score=0.0,
            document_id="doc_2",
        ),
        RetrievedDocument(
            text="Berlin is the capital city of Germany.",
            score=0.0,
            document_id="doc_3",
        ),
        RetrievedDocument(
            text="The Eiffel Tower is located in Paris.",
            score=0.0,
            document_id="doc_4",
        ),
    ]

    retriever = BM25Retriever(documents)

    system = JevRetrieve(
        retriever=retriever,
        evaluator=EvidenceEvaluator(),
        controller=CostAwareController(),
        max_iterations=2,
        top_k=2,
    )

    result = system.run(
        "What is the capital of France?"
    )

    assert result.query == (
        "What is the capital of France?"
    )

    assert len(result.documents) > 0

    assert result.iterations >= 1

    assert result.retrieval_calls >= 1

    assert result.jev_calls >= 1

    assert result.input_tokens > 0

    assert result.output_tokens > 0

    assert result.final_assessment is not None

    print("\nJevRetrieve result:")
    print("Iterations:", result.iterations)
    print("Retrieval calls:", result.retrieval_calls)
    print("Jev calls:", result.jev_calls)
    print("Jev tokens:", result.total_jev_tokens)
    print("Final sufficiency:",
          result.final_assessment.sufficiency)