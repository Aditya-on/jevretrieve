from pathlib import Path

from jevretrieve import ExperimentRunner, JevRetrieve
from jevretrieve.retrievers import (
    HybridRetriever,
    RetrievedDocument,
)


def load_queries(path: str) -> list[str]:
    """Load non-empty queries from a text file."""

    return [
        line.strip()
        for line in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def build_documents() -> list[RetrievedDocument]:
    """Build a deliberately difficult development corpus."""

    return [
        # --- Queen Victoria ---
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
                "was born Princess Victoire of Saxe-Coburg-Saalfeld."
            ),
            score=0.0,
        ),
        RetrievedDocument(
            document_id="victoria_4",
            text=(
                "Princess Victoire of Saxe-Coburg-Saalfeld "
                "was the daughter of Franz Frederick Anton, "
                "Duke of Saxe-Coburg-Saalfeld, and "
                "Countess Augusta Caroline Reuss of Ebersdorf."
            ),
            score=0.0,
        ),

        # --- Penicillin ---
        RetrievedDocument(
            document_id="penicillin_1",
            text=(
                "Alexander Fleming is credited with discovering "
                "penicillin in 1928."
            ),
            score=0.0,
        ),
        RetrievedDocument(
            document_id="penicillin_2",
            text=(
                "Fleming's discovery of penicillin occurred "
                "while he was researching antibacterial substances."
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
                "Darvel is a town in East Ayrshire, Scotland."
            ),
            score=0.0,
        ),

        # --- Canada / Australia ---
        RetrievedDocument(
            document_id="population_1",
            text=(
                "Canada is the second-largest country in the "
                "world by total area."
            ),
            score=0.0,
        ),
        RetrievedDocument(
            document_id="population_2",
            text=(
                "Australia is both a country and a continent."
            ),
            score=0.0,
        ),
        RetrievedDocument(
            document_id="population_3",
            text=(
                "Canada has a larger population than Australia."
            ),
            score=0.0,
        ),
        RetrievedDocument(
            document_id="population_4",
            text=(
                "Australia has a population of more than "
                "twenty-five million people."
            ),
            score=0.0,
        ),
    ]


def main() -> None:
    queries = load_queries(
        "datasets/dev_queries.txt"
    )

    documents = build_documents()

    retriever = HybridRetriever(
        documents=documents,
    )

    pipeline = JevRetrieve(
        retriever=retriever,
        top_k=2,
        max_iterations=None,
    )

    runner = ExperimentRunner(
        pipeline=pipeline,
    )

    results = runner.run(queries)

    output_path = (
        "results/dev_experiment.json"
    )

    runner.save_json(
        results,
        output_path,
    )

    print(
        f"Saved {len(results)} results to "
        f"{output_path}"
    )

    for result in results:
        print()
        print(f"Query: {result.query}")
        print(f"Iterations: {result.iterations}")
        print(f"Retrieval calls: {result.retrieval_calls}")
        print(f"Jev calls: {result.jev_calls}")
        print(f"Documents: {result.document_count}")
        print(
            f"Sufficiency: "
            f"{result.final_sufficiency:.3f}"
        )
        print(
            f"Coverage: "
            f"{result.final_coverage:.3f}"
        )
        print(
            f"Actions: {result.actions}"
        )
        print(
            f"Total cost: "
            f"${result.total_cost:.8f}"
        )


if __name__ == "__main__":
    main()