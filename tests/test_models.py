from jevretrieve.models import EvidenceState, JevJudgment


def test_jev_judgment():

    judgment = JevJudgment(
        score=0.89,
        judgment_type="noul",
        input_tokens=299,
        output_tokens=20,
    )

    assert judgment.score == 0.89
    assert judgment.total_tokens == 319


def test_evidence_state():

    state = EvidenceState(
        query="What is retrieval-augmented generation?"
    )

    judgment = JevJudgment(
        score=0.89,
        judgment_type="noul",
        input_tokens=299,
        output_tokens=20,
    )

    state.add_judgment(judgment)

    assert state.jev_calls == 1
    assert state.total_jev_tokens == 319
    assert state.best_score == 0.89
    