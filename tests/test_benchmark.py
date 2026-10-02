from jevretrieve.benchmark import BenchmarkRunner
from jevretrieve.retrievers import (
    BM25Retriever,
    RetrievedDocument,
)


class FakeEvaluator:
    def evaluate(self, query, evidence):
        from jevretrieve.models import EvidenceAssessment

        return EvidenceAssessment(
            sufficiency=0.80,
            relevance=0.80,
            coverage=0.80,
            redundancy=0.20,
            missing_information="NONE",
            input_tokens=100,
            output_tokens=20,
        )


def test_benchmark_runs_baseline():
    documents = [
        RetrievedDocument(
            document_id="1",
            text="Canada has a larger population than Australia.",
            score=0.0,
        ),
        RetrievedDocument(
            document_id="2",
            text="Australia is a country and continent.",
            score=0.0,
        ),
    ]

    retriever = BM25Retriever(documents)

    runner = BenchmarkRunner(
        evaluator=FakeEvaluator()
    )

    results = runner.run_baseline(
        name="BM25",
        retriever=retriever,
        queries=[
            "Which country has a larger population?"
        ],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].system == "BM25"
    assert results[0].retrieval_calls == 1
    assert results[0].jev_calls == 1
    assert results[0].document_count == 1