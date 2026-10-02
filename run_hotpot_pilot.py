
import json
from pathlib import Path

from jevretrieve import JevRetrieve
from jevretrieve.controller import CostAwareController
from jevretrieve.evaluator import EvidenceEvaluator
from jevretrieve.hotpotqa import load_hotpotqa
from jevretrieve.retrieval_metrics import (
    evidence_recall,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from jevretrieve.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
)


def evaluate_retrieval(
    documents,
    gold_ids,
):
    return {
        "recall_at_2": recall_at_k(
            documents,
            gold_ids,
            2,
        ),
        "recall_at_5": recall_at_k(
            documents,
            gold_ids,
            5,
        ),
        "mrr": mrr(
            documents,
            gold_ids,
        ),
        "ndcg_at_5": ndcg_at_k(
            documents,
            gold_ids,
            5,
        ),
        "evidence_recall": evidence_recall(
            documents,
            gold_ids,
        ),
    }


def run_static_baseline(
    name,
    retriever_cls,
    example,
    top_k=5,
):
    retriever = retriever_cls(
        example.documents
    )

    documents = retriever.retrieve(
        example.question,
        top_k=top_k,
    )

    metrics = evaluate_retrieval(
        documents,
        example.gold_document_ids,
    )

    return {
        "system": name,
        "question_id": example.question_id,
        "question": example.question,
        "level": example.level,
        "type": example.question_type,
        "document_count": len(documents),
        "jev_calls": 0,
        "retrieval_calls": 1,
        **metrics,
    }


def run_fixed_jev(
    example,
    evaluator,
    iterations=3,
    top_k=2,
):
    retriever = HybridRetriever(
        example.documents
    )

    documents = []
    total_input_tokens = 0
    total_output_tokens = 0

    for iteration in range(
        1,
        iterations + 1,
    ):
        retrieved = retriever.retrieve(
            example.question,
            top_k=top_k * iteration,
        )

        existing_ids = {
            document.document_id
            for document in documents
        }

        for document in retrieved:
            if document.document_id not in existing_ids:
                documents.append(document)

        # 🔴✓ API CALL — TypeSafe Jev
        assessment = evaluator.evaluate(
            query=example.question,
            evidence=[
                document.text
                for document in documents
            ],
        )

        total_input_tokens += (
            assessment.input_tokens
        )
        total_output_tokens += (
            assessment.output_tokens
        )

    metrics = evaluate_retrieval(
        documents,
        example.gold_document_ids,
    )

    return {
        "system": "Fixed-Jev",
        "question_id": example.question_id,
        "question": example.question,
        "level": example.level,
        "type": example.question_type,
        "document_count": len(documents),
        "jev_calls": iterations,
        "retrieval_calls": iterations,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        **metrics,
    }


def run_adaptive(
    example,
    evaluator,
):
    retriever = HybridRetriever(
        example.documents
    )

    controller = CostAwareController()

    pipeline = JevRetrieve(
        retriever=retriever,
        evaluator=evaluator,
        controller=controller,
        top_k=2,
        max_iterations=None,
    )

    # 🔴✓ API CALL — TypeSafe Jev
    result = pipeline.run(
        example.question
    )

    metrics = evaluate_retrieval(
        result.documents,
        example.gold_document_ids,
    )

    return {
        "system": "JevRetrieve",
        "question_id": example.question_id,
        "question": example.question,
        "level": example.level,
        "type": example.question_type,
        "document_count": len(
            result.documents
        ),
        "jev_calls": result.jev_calls,
        "retrieval_calls": result.retrieval_calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "sufficiency": (
            result.final_assessment.sufficiency
        ),
        "coverage": (
            result.final_assessment.coverage
        ),
        "relevance": (
            result.final_assessment.relevance
        ),
        "redundancy": (
            result.final_assessment.redundancy
        ),
        "actions": [
            decision.action.value
            for decision in result.decisions
        ],
        **metrics,
    }


def main():
    examples = load_hotpotqa(
        split="validation[:10]"
    )

    print(
        f"Loaded {len(examples)} HotpotQA questions."
    )

    evaluator = EvidenceEvaluator()

    results = []

    # --------------------------------------------------
    # Static retrieval baselines
    # --------------------------------------------------

    for example in examples:
        print()
        print(
            f"[STATIC] {example.question_id}"
        )

        results.append(
            run_static_baseline(
                "BM25",
                BM25Retriever,
                example,
            )
        )

        results.append(
            run_static_baseline(
                "Dense",
                DenseRetriever,
                example,
            )
        )

        results.append(
            run_static_baseline(
                "Hybrid",
                HybridRetriever,
                example,
            )
        )

    # --------------------------------------------------
    # Fixed-Jev
    # --------------------------------------------------

    for example in examples:
        print()
        print(
            f"[FIXED-JEV / API] "
            f"{example.question_id}"
        )

        results.append(
            run_fixed_jev(
                example,
                evaluator,
                iterations=3,
                top_k=2,
            )
        )

    # --------------------------------------------------
    # Adaptive JevRetrieve
    # --------------------------------------------------

    for example in examples:
        print()
        print(
            f"[JEVRETRIEVE / API] "
            f"{example.question_id}"
        )

        results.append(
            run_adaptive(
                example,
                evaluator,
            )
        )

    # --------------------------------------------------
    # V2 output file
    # --------------------------------------------------

    output = Path(
        "results/hotpotqa_pilot_10_v2.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved {len(results)} records "
        f"to {output}"
    )


if __name__ == "__main__":
    main()

