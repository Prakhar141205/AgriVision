from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# app/config.py -> app -> backend -> repository root
BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ============================================================
    # AgriVision CNN
    # ============================================================

    model_version: str = "v2"
    model_path_v1: str = "../ml/models/best_efficientnet_b0.pth"
    model_path_v2: str = "../ml/models/best_efficientnet_b0_v2.pth"
    class_names_path: str = "app/class_names.json"

    device: str = "auto"

    confidence_threshold: float = 0.60

    max_upload_size_bytes: int = 10 * 1024 * 1024

    # ============================================================
    # CORS
    # ============================================================

    cors_origins: str = (
        "http://localhost:3000,"
        "http://localhost:5173,"
        "http://127.0.0.1:3000,"
        "http://127.0.0.1:5173"
    )

    # ============================================================
    # Assistant
    # ============================================================

    hf_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"

    hf_token: str | None = None

    assistant_device: str = "auto"

    assistant_max_new_tokens: int = 320

    assistant_temperature: float = 0.2

    assistant_low_confidence_threshold: float = 0.60

    # ============================================================
    # Resolved paths
    # ============================================================

    @property
    def resolved_model_path(self) -> Path:
        version = self.model_version.lower().strip()
        if version == "v1":
            configured_path = self.model_path_v1
        elif version == "v2":
            configured_path = self.model_path_v2
        else:
            raise RuntimeError("MODEL_VERSION must be either v1 or v2.")

        return self._resolve_path(configured_path)

    @property
    def resolved_class_names_path(self) -> Path:
        return self._resolve_path(self.class_names_path)

    @staticmethod
    def _resolve_path(configured_path: str) -> Path:
        path = Path(configured_path).expanduser()
        if path.is_absolute():
            return path.resolve()

        # Resolve project settings consistently whether launched from the
        # repository root, backend/, or another working directory.
        candidates = (
            Path.cwd() / path,
            BACKEND_ROOT / path,
            PROJECT_ROOT / path,
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        return (BACKEND_ROOT / path).resolve()

    # ============================================================
    # CORS origins
    # ============================================================

    @property
    def allowed_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()