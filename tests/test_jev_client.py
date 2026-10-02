from jevretrieve.jev_client import JevClient
from jevretrieve.models import JevJudgment


def test_jev_client():

    client = JevClient()

    judgment = client.judge(
        state=(
            "Retrieval-augmented generation (RAG) combines "
            "information retrieval with a language model."
        ),
        question=(
            "Does the supplied evidence correctly explain "
            "what retrieval-augmented generation is?"
        ),
    )

    assert isinstance(judgment, JevJudgment)
    assert 0.0 <= judgment.score <= 1.0
    assert judgment.input_tokens > 0
    assert judgment.output_tokens > 0

    print("\nJev score:", judgment.score)
    print("Input tokens:", judgment.input_tokens)
    print("Output tokens:", judgment.output_tokens)