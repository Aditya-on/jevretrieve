from .jev_client import JevClient
from .models import EvidenceAssessment


class EvidenceEvaluator:
    """Evaluate retrieved evidence using Jev."""

    def __init__(
        self,
        jev_client: JevClient | None = None,
    ):
        self.jev = jev_client or JevClient()

    def evaluate(
        self,
        query: str,
        evidence: list[str],
    ) -> EvidenceAssessment:
        """Evaluate the current evidence set."""

        if not evidence:
            raise ValueError(
                "Evidence cannot be empty."
            )

        return self.jev.assess_evidence(
            query=query,
            evidence=evidence,
        )