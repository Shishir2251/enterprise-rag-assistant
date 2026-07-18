from app.business.interfaces.llm_provider_interface import ILLMProvider
from app.core.config import Settings, settings
from app.core.exceptions import ConfigurationError
from app.infrastructure.llm.no_llm_provider import NoLLMProvider


def create_llm_provider(config: Settings = settings) -> ILLMProvider:
    provider_name = config.LLM_PROVIDER.strip().lower()

    if provider_name in {"none", "disabled"}:
        return NoLLMProvider()

    raise ConfigurationError(
        "Unsupported LLM_PROVIDER. Only 'none' is currently available"
    )
