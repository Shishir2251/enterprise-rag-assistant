from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    APP_NAME: str = "Enterprise RAG Assistant"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True

    DATABASE_URL: str

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: Literal["HS256", "HS384", "HS512"] = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    OPENAI_API_KEY: SecretStr | None = None
    EMBEDDING_PROVIDER: str = "fake"
    EMBEDDING_MODEL: str = "fake-embedding-v1"
    EMBEDDING_DIMENSION: int = Field(default=1536, gt=0)
    EMBEDDING_BATCH_SIZE: int = Field(default=50, gt=0)
    RETRIEVAL_TOP_K: int = Field(default=5, gt=0)
    RETRIEVAL_MIN_SCORE: float = Field(default=0.30, ge=0.0, le=1.0)
    LLM_PROVIDER: str = "disabled"
    LLM_MODEL: str = "gpt-4.1-mini"
    LLM_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)
    LLM_MAX_OUTPUT_TOKENS: int = Field(default=1200, gt=0)
    LLM_TIMEOUT_SECONDS: int = Field(default=30, gt=0)
    MAX_CONTEXT_CHARACTERS: int = Field(default=12000, gt=0)
    CHAT_HISTORY_MAX_MESSAGES: int = Field(default=10, ge=0)

    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    CELERY_TASK_ALWAYS_EAGER: bool = False
    CELERY_TASK_EAGER_PROPAGATES: bool = True
    DOCUMENT_PROCESSING_MAX_RETRIES: int = Field(default=3, ge=0)
    DOCUMENT_PROCESSING_RETRY_DELAY_SECONDS: int = Field(default=30, ge=0)
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        if value.startswith("postgres://"):
            value = value.replace("postgres://", "postgresql://", 1)
        if not value.startswith("postgresql://"):
            raise ValueError("DATABASE_URL must use the postgresql:// scheme")
        return value

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        if self.APP_ENV.lower() == "production":
            if self.APP_DEBUG:
                raise ValueError("APP_DEBUG must be false in production")
            if len(self.JWT_SECRET_KEY) < 32:
                raise ValueError(
                    "JWT_SECRET_KEY must contain at least 32 characters "
                    "in production"
                )
        return self


settings = Settings()
