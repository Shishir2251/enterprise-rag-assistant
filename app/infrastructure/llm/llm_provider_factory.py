from app.business.interfaces.llm_provider_interface import ILLMProvider
from app.core.config import Settings, settings
from app.core.exceptions import ConfigurationError
from app.infrastructure.llm.no_llm_provider import NoLLMProvider


def create_llm_provider(config: Settings = settings) -> ILLMProvider:
    configured_provider = getattr(config, "LLM_PROVIDER", "")
    provider_name = (
        configured_provider.strip().lower()
        if isinstance(configured_provider, str)
        else ""
    )

    if provider_name in {"none", "disabled"}:
        return NoLLMProvider()
    if provider_name == "fake":
        from app.infrastructure.llm.fake_llm_provider import (
            DEFAULT_FAKE_LLM_MODEL,
            DEFAULT_NO_CONTEXT_MESSAGE,
            FakeLLMProvider,
        )

        return FakeLLMProvider(
            model_name=getattr(
                config,
                "FAKE_LLM_MODEL",
                DEFAULT_FAKE_LLM_MODEL,
            ),
            no_context_message=getattr(
                config,
                "CHAT_NO_CONTEXT_MESSAGE",
                DEFAULT_NO_CONTEXT_MESSAGE,
            ),
        )
    if provider_name == "openai":
        from app.infrastructure.llm.openai_llm_provider import (
            OpenAILLMProvider,
        )

        configured_api_key = getattr(config, "OPENAI_API_KEY", None)
        api_key = (
            configured_api_key.get_secret_value()
            if hasattr(configured_api_key, "get_secret_value")
            else configured_api_key
            if isinstance(configured_api_key, str)
            else ""
        )
        model_name = getattr(
            config,
            "OPENAI_CHAT_MODEL",
            getattr(config, "LLM_MODEL", ""),
        )
        return OpenAILLMProvider(
            api_key=api_key,
            model_name=model_name,
            temperature=config.LLM_TEMPERATURE,
            max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            timeout_seconds=config.LLM_TIMEOUT_SECONDS,
        )

    raise ConfigurationError(
        "Unsupported LLM_PROVIDER. Expected 'disabled', 'fake', or 'openai'"
    )
