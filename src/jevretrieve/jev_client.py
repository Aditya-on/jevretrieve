import os

import requests
from dotenv import load_dotenv

from .models import EvidenceAssessment, JevJudgment


load_dotenv()


class JevClient:
    """Client for the official TypeSafe Jev API."""

    def __init__(
        self,
        model: str = "jev-latest",
        timeout: int = 60,
    ):
        self.api_key = os.getenv("TYPESAFE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "TYPESAFE_API_KEY is not set. "
                "Add it to your .env file."
            )

        self.model = model
        self.timeout = timeout
        self.url = "https://api.typesafe.ai/v1/systemone"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        state: str,
        questions: dict,
    ) -> dict:
        """Send a request to the Jev API."""

        payload = {
            "model": self.model,
            "state": state,
            "questions": questions,
        }

        response = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )

        if not response.ok:
            raise RuntimeError(
                f"Jev API error {response.status_code}: "
                f"{response.text}"
            )

        return response.json()

    def judge(
        self,
        state: str,
        question: str,
    ) -> JevJudgment:
        """Ask Jev for a numeric judgment."""

        data = self._request(
            state=state,
            questions={
                "evaluation": {
                    "type": "noul",
                    "instructions": question,
                }
            },
        )

        answer = data["answers"]["evaluation"]

        return JevJudgment(
            score=float(answer["noul"]),
            judgment_type=answer["type"],
            input_tokens=data["usage"]["input_tokens"],
            output_tokens=data["usage"]["output_tokens"],
            raw_response=data,
        )

    def assess_evidence(
        self,
        query: str,
        evidence: list[str],
    ) -> EvidenceAssessment:
        """Assess the quality and completeness of retrieved evidence."""

        evidence_text = "\n\n".join(
            f"[Document {i + 1}]\n{text}"
            for i, text in enumerate(evidence)
        )

        state = f"""
Question:
{query}

Retrieved evidence:
{evidence_text}
""".strip()

        questions = {
            "sufficiency": {
                "type": "noul",
                "instructions": (
                    "Is the supplied evidence sufficient to "
                    "answer the question? Return a probability "
                    "from 0 to 1, where 0 means completely "
                    "insufficient and 1 means fully sufficient."
                ),
            },
            "relevance": {
                "type": "noul",
                "instructions": (
                    "How relevant is the retrieved evidence "
                    "to the question? Return a probability "
                    "from 0 to 1, where 0 means irrelevant "
                    "and 1 means highly relevant."
                ),
            },
            "coverage": {
                "type": "noul",
                "instructions": (
                    "How completely does the evidence cover "
                    "the information needed to answer the "
                    "question? Return a probability from 0 "
                    "to 1."
                ),
            },
            "redundancy": {
                "type": "noul",
                "instructions": (
                    "How redundant is the retrieved evidence? "
                    "Return a probability from 0 to 1, where "
                    "0 means little redundancy and 1 means "
                    "highly redundant."
                ),
            },
            "missing_information": {
                "type": "choice",
                "criteria": {
                    "NONE": (
                        "The evidence contains the information "
                        "needed to answer the question."
                    ),
                    "ENTITY": (
                        "A person, organization, object, or "
                        "other entity is missing."
                    ),
                    "RELATIONSHIP": (
                        "A relationship or connection between "
                        "entities is missing."
                    ),
                    "DATE": (
                        "A date, year, or time is missing."
                    ),
                    "LOCATION": (
                        "A location or geographic detail "
                        "is missing."
                    ),
                    "CAUSE": (
                        "A cause, reason, or explanation "
                        "is missing."
                    ),
                    "COMPARISON": (
                        "Information needed to compare "
                        "entities or events is missing."
                    ),
                    "ADDITIONAL_FACT": (
                        "Another important factual detail "
                        "is missing."
                    ),
                },
                "instructions": (
                    "Identify the most important category of "
                    "information that is still missing from "
                    "the retrieved evidence."
                ),
            },
        }

        data = self._request(
            state=state,
            questions=questions,
        )

        answers = data["answers"]

        return EvidenceAssessment(
            sufficiency=float(
                answers["sufficiency"]["noul"]
            ),
            relevance=float(
                answers["relevance"]["noul"]
            ),
            coverage=float(
                answers["coverage"]["noul"]
            ),
            redundancy=float(
                answers["redundancy"]["noul"]
            ),
            missing_information=str(
                answers["missing_information"]["choice"]
            ),
            input_tokens=data["usage"]["input_tokens"],
            output_tokens=data["usage"]["output_tokens"],
            raw_response=data,
        )