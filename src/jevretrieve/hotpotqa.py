from dataclasses import dataclass

from datasets import load_dataset

from .retrievers import RetrievedDocument


@dataclass
class HotpotExample:
    """One HotpotQA question with its retrieval corpus."""

    question_id: str
    question: str
    answer: str
    level: str
    question_type: str

    documents: list[RetrievedDocument]
    gold_document_ids: set[str]


def load_hotpotqa(
    split: str = "validation[:10]",
) -> list[HotpotExample]:
    """Load HotpotQA distractor examples."""

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split=split,
    )

    examples = []

    for row in dataset:
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]

        supporting_titles = set(
            row["supporting_facts"]["title"]
        )

        documents = []
        gold_document_ids = set()

        for title, paragraph_sentences in zip(
            titles,
            sentences,
        ):
            document_id = (
                f"{row['id']}::{title}"
            )

            text = " ".join(
                paragraph_sentences
            ).strip()

            documents.append(
                RetrievedDocument(
                    document_id=document_id,
                    text=text,
                    score=0.0,
                )
            )

            if title in supporting_titles:
                gold_document_ids.add(
                    document_id
                )

        examples.append(
            HotpotExample(
                question_id=row["id"],
                question=row["question"],
                answer=row["answer"],
                level=row["level"],
                question_type=row["type"],
                documents=documents,
                gold_document_ids=gold_document_ids,
            )
        )

    return examples