import os
from pathlib import Path


class Settings:
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    SECRET_KEY: str = os.getenv("SECRET_KEY", "some_random_secret_key")
    DEBUG: bool = bool(int(os.getenv("DEBUG", "1")))
    ALLOWED_ORIGINS: list[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5433")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "pgvector_db")
    POSTGRES_URL: str = (
        f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

    MEDIA_DIR: Path = BASE_DIR / "media"
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    PASSWORD_HASH_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    MATCH_THRESHOLD: float = 0.6
    VIDEO_FRAME_INTERVAL: int = 2

    FACE_RECOGNITION_MODEL: str = "ArcFace"
    FACE_EMBEDDING_MODEL: str = "ArcFace"


settings = Settings()
