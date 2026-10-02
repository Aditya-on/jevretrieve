import os

import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("TYPESAFE_API_KEY")


def test_jev_api_connection():
    assert API_KEY, "TYPESAFE_API_KEY is not set"

    response = requests.get(
        "https://api.typesafe.ai/v1/models",
        headers={
            "Authorization": f"Bearer {API_KEY}",
        },
        timeout=30,
    )

    assert response.status_code == 200, (
        f"Jev API returned {response.status_code}: {response.text}"
    )

    print("\nJev API connection successful.")
    print(response.json())