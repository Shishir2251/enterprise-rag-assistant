import pytest

from app.core.exceptions import ConfigurationError
from app.infrastructure.embeddings.local_embedding_provider import (
    LocalEmbeddingProvider,
)


pytestmark = pytest.mark.integration


def _similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


@pytest.fixture(scope="module")
def provider() -> LocalEmbeddingProvider:
    return LocalEmbeddingProvider(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        batch_size=32,
        device="cpu",
    )


@pytest.mark.parametrize(
    ("relevant", "unrelated", "query"),
    [
        (
            "The FIFA World Cup 2026 begins in June.",
            "The quarterly finance report describes operating expenses.",
            "When does the 2026 World Cup start?",
        ),
        (
            "The patient was diagnosed with sinusitis.",
            "The warehouse received a shipment of office chairs.",
            "What diagnosis was given to the patient?",
        ),
    ],
)
def test_semantically_related_text_scores_above_unrelated_text(
    provider: LocalEmbeddingProvider,
    relevant: str,
    unrelated: str,
    query: str,
) -> None:
    try:
        relevant_vector, unrelated_vector = provider.embed_texts(
            [relevant, unrelated]
        )
        query_vector = provider.embed_query(query)
    except ConfigurationError as exc:
        pytest.skip(
            "SentenceTransformers runtime is unavailable on this host: "
            f"{exc.detail}"
        )

    relevant_score = _similarity(query_vector, relevant_vector)
    unrelated_score = _similarity(query_vector, unrelated_vector)

    assert relevant_score > unrelated_score
