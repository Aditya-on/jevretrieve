from dataclasses import dataclass
from enum import Enum

from .models import EvidenceAssessment


class RetrievalAction(str, Enum):
    STOP = "STOP"
    RETRIEVE_MORE = "RETRIEVE_MORE"
    QUERY_EXPAND = "QUERY_EXPAND"
    DIVERSIFY = "DIVERSIFY"


@dataclass
class ControllerDecision:
    """Decision made by the retrieval controller."""

    action: RetrievalAction
    utility: float
    estimated_cost: float
    expected_gain: float
    reason: str


class CostAwareController:
    """
    Cost-aware adaptive retrieval controller.

    Decision order:

    1. Stop when evidence satisfies the quality floor.
    2. Continue when evidence is critically insufficient.
    3. Stop when expected gain is too small.
    4. Stop when retrieval is converging.
    5. Apply the economic utility gate.
    """

    def __init__(
        self,
        utility_threshold: float = 1.0,
        expected_gain: float = 0.20,
        retrieval_cost: float = 0.0,
        jev_input_cost_per_million: float = 0.042,
        cost_scale: float = 1000.0,
        min_gain: float = 0.02,
        convergence_ratio: float = 0.25,
        minimum_sufficiency: float = 0.85,
        critical_sufficiency: float = 0.50,
        critical_coverage: float = 0.50,
    ):
        if utility_threshold < 0:
            raise ValueError(
                "utility_threshold must be non-negative."
            )

        if expected_gain < 0:
            raise ValueError(
                "expected_gain must be non-negative."
            )

        if retrieval_cost < 0:
            raise ValueError(
                "retrieval_cost must be non-negative."
            )

        if jev_input_cost_per_million < 0:
            raise ValueError(
                "jev_input_cost_per_million must be non-negative."
            )

        if cost_scale <= 0:
            raise ValueError(
                "cost_scale must be greater than zero."
            )

        if min_gain < 0:
            raise ValueError(
                "min_gain must be non-negative."
            )

        if not 0.0 <= convergence_ratio <= 1.0:
            raise ValueError(
                "convergence_ratio must be between 0 and 1."
            )

        if not 0.0 <= minimum_sufficiency <= 1.0:
            raise ValueError(
                "minimum_sufficiency must be between 0 and 1."
            )

        if not 0.0 <= critical_sufficiency <= 1.0:
            raise ValueError(
                "critical_sufficiency must be between 0 and 1."
            )

        if not 0.0 <= critical_coverage <= 1.0:
            raise ValueError(
                "critical_coverage must be between 0 and 1."
            )

        if critical_sufficiency > minimum_sufficiency:
            raise ValueError(
                "critical_sufficiency cannot exceed "
                "minimum_sufficiency."
            )

        if critical_coverage > minimum_sufficiency:
            raise ValueError(
                "critical_coverage cannot exceed "
                "minimum_sufficiency."
            )

        self.utility_threshold = utility_threshold
        self.expected_gain = expected_gain
        self.retrieval_cost = retrieval_cost

        self.jev_input_cost_per_million = (
            jev_input_cost_per_million
        )

        self.cost_scale = cost_scale
        self.min_gain = min_gain
        self.convergence_ratio = convergence_ratio
        self.minimum_sufficiency = minimum_sufficiency

        self.critical_sufficiency = (
            critical_sufficiency
        )
        self.critical_coverage = (
            critical_coverage
        )

        self._previous_gain: float | None = None

    def reset(self) -> None:
        """Reset controller history for a new query."""

        self._previous_gain = None

    def decide(
        self,
        assessment: EvidenceAssessment,
    ) -> ControllerDecision:
        """Choose whether and how to continue retrieval."""

        # --------------------------------------------------
        # 1. Quality gate
        # --------------------------------------------------

        if (
            assessment.sufficiency
            >= self.minimum_sufficiency
        ):
            self._previous_gain = 0.0

            return ControllerDecision(
                action=RetrievalAction.STOP,
                utility=0.0,
                estimated_cost=0.0,
                expected_gain=0.0,
                reason=(
                    "Evidence satisfies the minimum "
                    "sufficiency threshold."
                ),
            )

        # --------------------------------------------------
        # 2. Select the retrieval action
        # --------------------------------------------------

        action = self._select_action(
            assessment
        )

        expected_gain = self._estimate_gain(
            assessment,
            action,
        )

        estimated_cost = self._estimate_action_cost(
            assessment,
            action,
        )

        normalized_cost = (
            estimated_cost
            * self.cost_scale
        )

        utility = (
            expected_gain / normalized_cost
            if normalized_cost > 0
            else float("inf")
        )

        # --------------------------------------------------
        # 3. Critical insufficiency override
        # --------------------------------------------------
        #
        # If evidence is seriously incomplete, do not let
        # the economic gate stop retrieval prematurely.
        #
        # This is the main refinement introduced in V2.
        #

        critically_insufficient = (
            assessment.sufficiency
            < self.critical_sufficiency
            or assessment.coverage
            < self.critical_coverage
        )

        if critically_insufficient:
            self._previous_gain = expected_gain

            return ControllerDecision(
                action=action,
                utility=utility,
                estimated_cost=estimated_cost,
                expected_gain=expected_gain,
                reason=(
                    "Evidence is critically insufficient; "
                    "continue retrieval before applying "
                    "the economic stopping rule."
                ),
            )

        # --------------------------------------------------
        # 4. Minimum useful gain
        # --------------------------------------------------

        if expected_gain < self.min_gain:
            self._previous_gain = expected_gain

            return ControllerDecision(
                action=RetrievalAction.STOP,
                utility=utility,
                estimated_cost=estimated_cost,
                expected_gain=expected_gain,
                reason=(
                    "Expected evidence gain is below "
                    "the minimum useful-gain threshold."
                ),
            )

        # --------------------------------------------------
        # 5. Convergence check
        # --------------------------------------------------

        if self._previous_gain is not None:
            if (
                self._previous_gain > 0
                and expected_gain
                < self._previous_gain
                * self.convergence_ratio
            ):
                previous_gain = (
                    self._previous_gain
                )

                self._previous_gain = expected_gain

                return ControllerDecision(
                    action=RetrievalAction.STOP,
                    utility=utility,
                    estimated_cost=estimated_cost,
                    expected_gain=expected_gain,
                    reason=(
                        "Expected gain has sharply decreased "
                        f"from {previous_gain:.4f} to "
                        f"{expected_gain:.4f}; retrieval is "
                        "converging."
                    ),
                )

        self._previous_gain = expected_gain

        # --------------------------------------------------
        # 6. Economic gate
        # --------------------------------------------------

        if utility < self.utility_threshold:
            return ControllerDecision(
                action=RetrievalAction.STOP,
                utility=utility,
                estimated_cost=estimated_cost,
                expected_gain=expected_gain,
                reason=(
                    "Expected evidence gain does not "
                    "justify the normalized incremental cost."
                ),
            )

        # --------------------------------------------------
        # 7. Continue
        # --------------------------------------------------

        return ControllerDecision(
            action=action,
            utility=utility,
            estimated_cost=estimated_cost,
            expected_gain=expected_gain,
            reason=(
                "Evidence is below the sufficiency threshold "
                "and expected gain justifies another retrieval "
                "iteration."
            ),
        )

    def _estimate_action_cost(
        self,
        assessment: EvidenceAssessment,
        action: RetrievalAction,
    ) -> float:
        """Estimate incremental cost of an action."""

        if action == RetrievalAction.STOP:
            return 0.0

        jev_cost = (
            assessment.input_tokens
            / 1_000_000
            * self.jev_input_cost_per_million
        )

        return (
            self.retrieval_cost
            + jev_cost
        )

    def _estimate_gain(
        self,
        assessment: EvidenceAssessment,
        action: RetrievalAction,
    ) -> float:
        """Estimate expected evidence-quality improvement."""

        missing = 1.0 - assessment.coverage
        low_relevance = 1.0 - assessment.relevance
        high_redundancy = assessment.redundancy

        base_gain = self.expected_gain * (
            0.5 * missing
            + 0.3 * low_relevance
            + 0.2 * high_redundancy
        )

        action_multiplier = {
            RetrievalAction.RETRIEVE_MORE: 1.00,
            RetrievalAction.QUERY_EXPAND: 1.10,
            RetrievalAction.DIVERSIFY: 0.90,
            RetrievalAction.STOP: 0.0,
        }[action]

        gain = (
            base_gain
            * action_multiplier
        )

        return max(
            0.0,
            min(1.0, gain),
        )

    @staticmethod
    def _select_action(
        assessment: EvidenceAssessment,
    ) -> RetrievalAction:
        """Select the action suited to the evidence state."""

        if assessment.coverage < 0.50:
            return RetrievalAction.RETRIEVE_MORE

        if (
            assessment.missing_information
            .strip()
            .upper()
            != "NONE"
        ):
            return RetrievalAction.QUERY_EXPAND

        if assessment.redundancy > 0.50:
            return RetrievalAction.DIVERSIFY

        return RetrievalAction.RETRIEVE_MORE