from functools import cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_embeddings_model: str = "text-embedding-3-small"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@cache
def get_settings() -> Settings:
    """Return the single instance of the application settings."""

    # We initialize Settings without any arguments,
    # because the arguments must be set in the environment or the server should crash.
    return Settings()
