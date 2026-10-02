from jevretrieve.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RetrievedDocument,
)


def test_bm25_retrieval():

    documents = [
        RetrievedDocument(
            text="Paris is the capital of France.",
            score=0.0,
            document_id="doc_1",
        ),
        RetrievedDocument(
            text="Berlin is the capital of Germany.",
            score=0.0,
            document_id="doc_2",
        ),
        RetrievedDocument(
            text="Paris has many famous museums.",
            score=0.0,
            document_id="doc_3",
        ),
    ]

    retriever = BM25Retriever(documents)

    results = retriever.retrieve(
        "capital of France",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].document_id == "doc_1"


def test_dense_retrieval():

    documents = [
        RetrievedDocument(
            text="Paris is the capital city of France.",
            score=0.0,
            document_id="doc_1",
        ),
        RetrievedDocument(
            text="Berlin is the capital city of Germany.",
            score=0.0,
            document_id="doc_2",
        ),
        RetrievedDocument(
            text="The Eiffel Tower is located in Paris.",
            score=0.0,
            document_id="doc_3",
        ),
    ]

    retriever = DenseRetriever(documents)

    results = retriever.retrieve(
        "What city is the capital of France?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].document_id == "doc_1"


def test_hybrid_retrieval():

    documents = [
        RetrievedDocument(
            text="Paris is the capital city of France.",
            score=0.0,
            document_id="doc_1",
        ),
        RetrievedDocument(
            text="Berlin is the capital city of Germany.",
            score=0.0,
            document_id="doc_2",
        ),
        RetrievedDocument(
            text="The Eiffel Tower is located in Paris.",
            score=0.0,
            document_id="doc_3",
        ),
    ]

    retriever = HybridRetriever(
        documents,
        dense_weight=0.5,
    )

    results = retriever.retrieve(
        "What city is the capital of France?",
        top_k=2,
    )

    assert len(results) == 2

    document_ids = {
        result.document_id
        for result in results
    }

    assert len(document_ids) == 2

    assert document_ids.issubset(
        {"doc_1", "doc_2", "doc_3"}
    )

    for result in results:
        assert 0.0 <= result.score <= 1.0