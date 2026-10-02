from pathlib import Path

from jevretrieve import (
    BenchmarkRunner,
    JevRetrieve,
)
from jevretrieve.controller import CostAwareController
from jevretrieve.evaluator import EvidenceEvaluator
from jevretrieve.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RetrievedDocument,
)


def load_queries(path: str) -> list[str]:
    return [
        line.strip()
        for line in Path(path)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def build_documents() -> list[RetrievedDocument]:
    return [
        RetrievedDocument(
            document_id="victoria_1",
            text=(
                "Queen Victoria was the daughter of "
                "Prince Edward, Duke of Kent, and "
                "Princess Victoria, Duchess of Kent."
            ),
            score=0.0,
        ),
        RetrievedDocument(
            document_id="victoria_2",
            text=(
                "Princess Victoria, Duchess of Kent, "
                "was the mother of Queen Victoria."
            ),
            score=0.0,
        ),
        RetrievedDocument(
            document_id="victoria_3",
            text=(
                "Princess Victoria, Duchess of Kent, "
                "was born Princess Victoire of "
                "Saxe-Coburg-Saalfeld."
            ),
            score=0.0,
        ),
        RetrievedDocument(
            document_id="victoria_4",
            text=(
                "Princess Victoire of "
                "Saxe-Coburg-Saalfeld was the daughter "
                "of Franz Frederick Anton, Duke of "
                "Saxe-Coburg-Saalfeld, and Countess "
                "Augusta Caroline Reuss of Ebersdorf."
            ),
            score=0.0,
        ),
        RetrievedDocument(
            document_id="penicillin_1",
            text=(
                "Alexander Fleming is credited with "
                "discovering penicillin in 1928."
            ),
            score=0.0,
        ),
        RetrievedDocument(
            document_id="penicillin_2",
            text=(
                "Fleming's discovery of penicillin "
                "occurred while he was researching "
                "antibacterial substances."
            ),
            score=0.0,
        ),
        RetrievedDocument(
            document_id="penicillin_3",
            text=(
                "Alexander Fleming was born in Darvel, "
                "East Ayrshire, Scotland, in 1881."
            ),
            score=0.0,
        ),
        RetrievedDocument(
            document_id="penicillin_4",
            text=(
                "Darvel is a town in East Ayrshire, "
                "Scotland."
            ),
            score=0.0,
        ),
        RetrievedDocument(
            document_id="population_1",
            text=(
                "Canada is the second-largest country "
                "in the world by total area."
            ),
            score=0.0,
        ),
        RetrievedDocument(
            document_id="population_2",
            text=(
                "Australia is both a country and "
                "a continent."
            ),
            score=0.0,
        ),
        RetrievedDocument(
            document_id="population_3",
            text=(
                "Canada has a larger population "
                "than Australia."
            ),
            score=0.0,
        ),
        RetrievedDocument(
            document_id="population_4",
            text=(
                "Australia has a population of more "
                "than twenty-five million people."
            ),
            score=0.0,
        ),
    ]


def print_summary(results):
    for result in results:
        print()
        print(
            f"{result.system}: "
            f"{result.query}"
        )
        print(
            f"  Documents: {result.document_count}"
        )
        print(
            f"  Retrieval calls: "
            f"{result.retrieval_calls}"
        )
        print(
            f"  Jev calls: "
            f"{result.jev_calls}"
        )
        print(
            f"  Sufficiency: "
            f"{result.sufficiency:.3f}"
        )
        print(
            f"  Coverage: "
            f"{result.coverage:.3f}"
        )
        print(
            f"  Estimated cost: "
            f"${result.estimated_cost:.8f}"
        )
        print(
            f"  Actions: "
            f"{result.actions}"
        )


def main():
    queries = load_queries(
        "datasets/dev_queries.txt"
    )

    documents = build_documents()

    evaluator = EvidenceEvaluator()

    runner = BenchmarkRunner(
        evaluator=evaluator
    )

    results = []

    # --------------------------------------------------
    # Fixed-budget retrieval baselines
    # --------------------------------------------------

    bm25 = BM25Retriever(documents)

    results.extend(
        runner.run_baseline(
            name="BM25",
            retriever=bm25,
            queries=queries,
            top_k=2,
        )
    )

    dense = DenseRetriever(documents)

    results.extend(
        runner.run_baseline(
            name="Dense",
            retriever=dense,
            queries=queries,
            top_k=2,
        )
    )

    hybrid = HybridRetriever(documents)

    results.extend(
        runner.run_baseline(
            name="Hybrid",
            retriever=hybrid,
            queries=queries,
            top_k=2,
        )
    )

    results.extend(
        runner.run_baseline(
            name="Hybrid+MMR",
            retriever=hybrid,
            queries=queries,
            top_k=2,
            use_mmr=True,
        )
    )

    # --------------------------------------------------
    # Fixed Jev baseline
    # --------------------------------------------------

    fixed_jev_retriever = HybridRetriever(
        documents
    )

    results.extend(
        runner.run_fixed_jev(
            retriever=fixed_jev_retriever,
            queries=queries,
            top_k=2,
            iterations=3,
        )
    )

    # --------------------------------------------------
    # Adaptive JevRetrieve
    # --------------------------------------------------

    adaptive_retriever = HybridRetriever(
        documents
    )

    pipeline = JevRetrieve(
        retriever=adaptive_retriever,
        evaluator=evaluator,
        controller=CostAwareController(),
        top_k=2,
        max_iterations=None,
    )

    results.extend(
        runner.run_jev_retrieve(
            pipeline=pipeline,
            queries=queries,
        )
    )

    output_path = (
        "results/benchmark_results.json"
    )

    runner.save_json(
        results,
        output_path,
    )

    print(
        f"Saved {len(results)} benchmark results "
        f"to {output_path}"
    )

    print_summary(results)


if __name__ == "__main__":
    main()