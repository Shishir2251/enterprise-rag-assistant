from app.business.interfaces.embedding_provider_interface import (
    IEmbeddingProvider,
)
from app.core.config import Settings, settings
from app.core.exceptions import ConfigurationError
from app.infrastructure.embeddings.fake_embedding_provider import (
    FakeEmbeddingProvider,
)
from app.infrastructure.embeddings.openai_embedding_provider import (
    OpenAIEmbeddingProvider,
)


def create_embedding_provider(
    config: Settings = settings,
) -> IEmbeddingProvider:
    provider_name = config.EMBEDDING_PROVIDER.strip().lower()

    if provider_name == "fake":
        return FakeEmbeddingProvider(
            dimensions=config.EMBEDDING_DIMENSION,
            model_name=config.EMBEDDING_MODEL,
        )
    if provider_name == "openai":
        return OpenAIEmbeddingProvider(
            api_key=config.OPENAI_API_KEY.get_secret_value(),
            model_name=config.EMBEDDING_MODEL,
            dimensions=config.EMBEDDING_DIMENSION,
        )

    raise ConfigurationError(
        "Unsupported EMBEDDING_PROVIDER. Expected 'fake' or 'openai'"
    )
