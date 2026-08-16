from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_path: str = "../ml/models/best_efficientnet_b0.pth"
    class_names_path: str = "app/class_names.json"
    device: str = "auto"
    confidence_threshold: float = 0.60
    max_upload_size_bytes: int = 10 * 1024 * 1024
    cors_origins: str = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
    hf_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    hf_token: str | None = None
    assistant_device: str = "auto"
    assistant_max_new_tokens: int = 320
    assistant_temperature: float = 0.2
    assistant_low_confidence_threshold: float = 0.60

    @property
    def resolved_model_path(self) -> Path:
        return Path(self.model_path).expanduser().resolve()

    @property
    def resolved_class_names_path(self) -> Path:
        return Path(self.class_names_path).expanduser().resolve()

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
