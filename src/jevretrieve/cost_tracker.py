from dataclasses import dataclass


@dataclass
class CostTracker:
    """Track retrieval and Jev usage during a run."""

    retrieval_calls: int = 0
    jev_calls: int = 0

    input_tokens: int = 0
    output_tokens: int = 0

    retrieval_cost: float = 0.0
    jev_cost: float = 0.0

    def record_retrieval(
        self,
        cost: float = 0.0,
    ) -> None:
        """Record one retrieval operation."""

        self.retrieval_calls += 1
        self.retrieval_cost += cost

    def record_jev(
        self,
        input_tokens: int,
        output_tokens: int,
        cost: float = 0.0,
    ) -> None:
        """Record one Jev API call."""

        self.jev_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.jev_cost += cost

    @property
    def total_tokens(self) -> int:
        """Total Jev tokens consumed."""

        return self.input_tokens + self.output_tokens

    @property
    def total_cost(self) -> float:
        """Total estimated retrieval + Jev cost."""

        return self.retrieval_cost + self.jev_cost

    @property
    def cost_per_jev_call(self) -> float:
        """Average estimated cost per Jev call."""

        if self.jev_calls == 0:
            return 0.0

        return self.jev_cost / self.jev_calls

    def reset(self) -> None:
        """Reset all tracked usage."""

        self.retrieval_calls = 0
        self.jev_calls = 0

        self.input_tokens = 0
        self.output_tokens = 0

        self.retrieval_cost = 0.0
        self.jev_cost = 0.0