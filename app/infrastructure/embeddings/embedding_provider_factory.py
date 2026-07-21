from app.business.interfaces.embedding_provider_interface import (
    IEmbeddingProvider,
)
from app.core.config import Settings, settings
from app.core.exceptions import ConfigurationError
from app.infrastructure.embeddings.fake_embedding_provider import (
    FakeEmbeddingProvider,
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
    if provider_name == "local":
        from app.infrastructure.embeddings.local_embedding_provider import (
            LocalEmbeddingProvider,
        )

        active_model = config.EMBEDDING_MODEL.strip()
        local_model = config.LOCAL_EMBEDDING_MODEL.strip()
        if active_model != local_model:
            raise ConfigurationError(
                "EMBEDDING_MODEL must match LOCAL_EMBEDDING_MODEL in local mode"
            )

        return LocalEmbeddingProvider(
            model_name=local_model,
            dimensions=config.EMBEDDING_DIMENSION,
            batch_size=config.LOCAL_EMBEDDING_BATCH_SIZE,
            device=config.LOCAL_EMBEDDING_DEVICE,
        )
    if provider_name == "http":
        from app.infrastructure.embeddings.http_embedding_provider import (
            HTTPEmbeddingProvider,
        )

        return HTTPEmbeddingProvider(
            base_url=config.HTTP_EMBEDDING_BASE_URL,
            model_name=config.EMBEDDING_MODEL,
            dimensions=config.EMBEDDING_DIMENSION,
            timeout_seconds=config.HTTP_EMBEDDING_TIMEOUT_SECONDS,
        )
    if provider_name == "openai":
        from app.infrastructure.embeddings.openai_embedding_provider import (
            OpenAIEmbeddingProvider,
        )

        configured_api_key = config.OPENAI_API_KEY
        return OpenAIEmbeddingProvider(
            api_key=(
                configured_api_key.get_secret_value()
                if configured_api_key is not None
                else ""
            ),
            model_name=config.EMBEDDING_MODEL,
            dimensions=config.EMBEDDING_DIMENSION,
        )

    raise ConfigurationError(
        "Unsupported EMBEDDING_PROVIDER. Expected 'fake', 'local', 'http', "
        "or 'openai'"
    )
