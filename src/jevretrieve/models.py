from dataclasses import dataclass, field
from typing import Any


@dataclass
class JevJudgment:
    """Structured result from a Jev evaluation."""

    score: float
    judgment_type: str
    input_tokens: int
    output_tokens: int
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed by this judgment."""
        return self.input_tokens + self.output_tokens


@dataclass
class EvidenceAssessment:
    """Assessment of a retrieved evidence set."""

    sufficiency: float
    relevance: float
    coverage: float
    redundancy: float
    missing_information: str

    input_tokens: int
    output_tokens: int

    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed by this assessment."""
        return self.input_tokens + self.output_tokens


@dataclass
class EvidenceState:
    """Current state of evidence collected for a query."""

    query: str

    documents: list[str] = field(default_factory=list)

    judgments: list[JevJudgment] = field(default_factory=list)

    assessments: list[EvidenceAssessment] = field(
        default_factory=list
    )

    iteration: int = 0

    retrieval_calls: int = 0

    jev_calls: int = 0

    input_tokens: int = 0

    output_tokens: int = 0

    def add_judgment(self, judgment: JevJudgment) -> None:
        """Add a Jev judgment and update cost statistics."""

        self.judgments.append(judgment)

        self.jev_calls += 1
        self.input_tokens += judgment.input_tokens
        self.output_tokens += judgment.output_tokens

    def add_assessment(
        self,
        assessment: EvidenceAssessment,
    ) -> None:
        """Add an evidence assessment and update cost statistics."""

        self.assessments.append(assessment)

        self.jev_calls += 1
        self.input_tokens += assessment.input_tokens
        self.output_tokens += assessment.output_tokens

    @property
    def total_jev_tokens(self) -> int:
        """Total tokens used by Jev so far."""

        return self.input_tokens + self.output_tokens

    @property
    def best_score(self) -> float:
        """Highest Jev evidence score observed so far."""

        if not self.judgments:
            return 0.0

        return max(
            judgment.score
            for judgment in self.judgments
        )