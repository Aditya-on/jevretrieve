from jevretrieve.diversifier import MMRAssessor
from jevretrieve.retrievers import RetrievedDocument


def test_mmr_diversification():

    documents = [
        RetrievedDocument(
            text="Ada Lovelace worked on the Analytical Engine.",
            score=0.95,
            document_id="doc_1",
        ),
        RetrievedDocument(
            text="Ada Lovelace worked on the Analytical Engine.",
            score=0.94,
            document_id="doc_2",
        ),
        RetrievedDocument(
            text="Charles Babbage designed the Analytical Engine.",
            score=0.90,
            document_id="doc_3",
        ),
        RetrievedDocument(
            text="Ada Lovelace was born in London in 1815.",
            score=0.80,
            document_id="doc_4",
        ),
    ]

    diversifier = MMRAssessor(
        lambda_weight=0.7,
    )

    selected = diversifier.diversify(
        query="Ada Lovelace and the Analytical Engine",
        documents=documents,
        top_k=3,
    )

    assert len(selected) == 3

    selected_ids = [
        document.document_id
        for document in selected
    ]

    assert len(selected_ids) == len(set(selected_ids))