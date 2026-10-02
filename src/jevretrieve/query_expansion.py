from .models import EvidenceAssessment


class QueryExpander:
    """Create a focused follow-up query from Jev's missing-information category."""

    def expand(
        self,
        original_query: str,
        assessment: EvidenceAssessment,
    ) -> str:
        """Build a targeted follow-up query."""

        category = assessment.missing_information.strip().upper()

        if not category or category == "NONE":
            return original_query

        prompts = {
            "ENTITY": (
                "Find the missing person, organization, "
                "or entity needed to answer the question."
            ),
            "RELATIONSHIP": (
                "Find the missing relationship or connection "
                "between the entities in the question."
            ),
            "DATE": (
                "Find the missing date, year, or time information "
                "needed to answer the question."
            ),
            "LOCATION": (
                "Find the missing location, place, or geographic "
                "information needed to answer the question."
            ),
            "CAUSE": (
                "Find the missing cause, reason, or explanation "
                "needed to answer the question."
            ),
            "COMPARISON": (
                "Find the missing information needed to compare "
                "the relevant entities or events."
            ),
            "ADDITIONAL_FACT": (
                "Find the additional factual information needed "
                "to answer the question."
            ),
        }

        instruction = prompts.get(
            category,
            "Find additional information needed to answer the question.",
        )

        return f"{original_query} {instruction}"
