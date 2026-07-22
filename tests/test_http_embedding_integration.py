import math
import os
from pathlib import Path

import httpx
import pytest


pytestmark = pytest.mark.integration

BASE_URL = os.getenv(
    "HTTP_EMBEDDING_BASE_URL",
    "http://127.0.0.1:8090",
).rstrip("/")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = 384
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase9_multitopic.txt"


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm)


def test_docker_embedding_service_semantic_similarity() -> None:
    try:
        health = httpx.get(f"{BASE_URL}/health", timeout=3)
        health.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        pytest.skip(f"Docker embedding service is unavailable: {exc}")

    assert health.json() == {
        "status": "ok",
        "model": MODEL_NAME,
        "dimension": DIMENSIONS,
    }

    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        document_response = client.post(
            "/embed",
            json={
                "texts": [
                    "The FIFA World Cup 2026 will be hosted by the "
                    "United States, Canada, and Mexico."
                ]
            },
        )
        related_response = client.post(
            "/embed-query",
            json={
                "query": (
                    "Which countries are hosting the 2026 football "
                    "tournament?"
                )
            },
        )
        unrelated_response = client.post(
            "/embed-query",
            json={"query": "How do I configure PostgreSQL backups?"},
        )

    document_response.raise_for_status()
    related_response.raise_for_status()
    unrelated_response.raise_for_status()

    document_payload = document_response.json()
    related_payload = related_response.json()
    unrelated_payload = unrelated_response.json()
    document_vector = document_payload["embeddings"][0]
    related_vector = related_payload["embedding"]
    unrelated_vector = unrelated_payload["embedding"]

    assert document_payload["model"] == MODEL_NAME
    assert document_payload["dimension"] == DIMENSIONS
    assert len(document_vector) == DIMENSIONS
    assert len(related_vector) == DIMENSIONS
    assert len(unrelated_vector) == DIMENSIONS
    assert cosine_similarity(document_vector, related_vector) > (
        cosine_similarity(document_vector, unrelated_vector)
    )


def test_multitopic_fixture_has_topic_sensitive_real_embeddings() -> None:
    try:
        health = httpx.get(f"{BASE_URL}/health", timeout=3)
        health.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        pytest.skip(f"Docker embedding service is unavailable: {exc}")

    parts = [
        part.replace("\n", " ").strip()
        for part in FIXTURE_PATH.read_text(encoding="utf-8").split("\n\n")
        if part.strip()
    ]
    headings = parts[0::2]
    passages = parts[1::2]
    assert len(headings) == len(passages) == 5

    queries = (
        "Which countries are hosting the 2026 football tournament?",
        "Which programming language is widely used for machine learning?",
        "What can be used for caching and as a Celery message broker?",
        "Which PostgreSQL extension stores vectors for similarity search?",
        "What provides a consistent Linux runtime in isolated containers?",
    )

    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        passage_response = client.post("/embed", json={"texts": passages})
        passage_response.raise_for_status()
        passage_vectors = passage_response.json()["embeddings"]

        query_vectors = []
        for query in queries:
            response = client.post("/embed-query", json={"query": query})
            response.raise_for_status()
            query_vectors.append(response.json()["embedding"])

        unrelated_response = client.post(
            "/embed-query",
            json={"query": "How should chicken biryani be seasoned and cooked?"},
        )
        unrelated_response.raise_for_status()
        unrelated_vector = unrelated_response.json()["embedding"]

    for expected_index, query_vector in enumerate(query_vectors):
        scores = [
            cosine_similarity(query_vector, passage_vector)
            for passage_vector in passage_vectors
        ]
        assert scores.index(max(scores)) == expected_index

    unrelated_scores = [
        cosine_similarity(unrelated_vector, passage_vector)
        for passage_vector in passage_vectors
    ]
    assert max(unrelated_scores) < 0.25
