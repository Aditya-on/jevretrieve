import pytest

from jevretrieve.cost_tracker import CostTracker


def test_cost_tracker():

    tracker = CostTracker()

    tracker.record_retrieval(
        cost=0.01,
    )

    tracker.record_jev(
        input_tokens=100,
        output_tokens=20,
        cost=0.05,
    )

    tracker.record_jev(
        input_tokens=200,
        output_tokens=30,
        cost=0.10,
    )

    assert tracker.retrieval_calls == 1
    assert tracker.jev_calls == 2

    assert tracker.input_tokens == 300
    assert tracker.output_tokens == 50

    assert tracker.total_tokens == 350

    assert tracker.retrieval_cost == pytest.approx(0.01)
    assert tracker.jev_cost == pytest.approx(0.15)

    assert tracker.total_cost == pytest.approx(0.16)

    assert tracker.cost_per_jev_call == pytest.approx(0.075)


def test_cost_tracker_reset():

    tracker = CostTracker()

    tracker.record_jev(
        input_tokens=100,
        output_tokens=20,
        cost=0.05,
    )

    tracker.reset()

    assert tracker.jev_calls == 0
    assert tracker.input_tokens == 0
    assert tracker.output_tokens == 0
    assert tracker.total_cost == 0.0