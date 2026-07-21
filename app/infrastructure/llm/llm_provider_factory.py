from app.business.interfaces.llm_provider_interface import ILLMProvider
from app.core.config import Settings, settings
from app.core.exceptions import ConfigurationError
from app.infrastructure.llm.no_llm_provider import NoLLMProvider


def create_llm_provider(config: Settings = settings) -> ILLMProvider:
    provider_name = config.LLM_PROVIDER.strip().lower()

    if provider_name in {"none", "disabled"}:
        return NoLLMProvider()
    if provider_name == "openai":
        from app.infrastructure.llm.openai_llm_provider import (
            OpenAILLMProvider,
        )

        configured_api_key = config.OPENAI_API_KEY
        api_key = (
            configured_api_key.get_secret_value()
            if configured_api_key is not None
            else ""
        )
        return OpenAILLMProvider(
            api_key=api_key,
            model_name=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            timeout_seconds=config.LLM_TIMEOUT_SECONDS,
        )

    raise ConfigurationError(
        "Unsupported LLM_PROVIDER. Expected 'disabled' or 'openai'"
    )
