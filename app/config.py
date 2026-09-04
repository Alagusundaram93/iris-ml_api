from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    MODEL_PATH: str = "ml/saved_model/model.joblib"
    METADATA_PATH: str = "ml/saved_model/metadata.json"
    LOG_LEVEL: str = "INFO"
    MAX_BATCH_SIZE: int = 100
    API_TITLE: str = "MY_IRIS_PREDICTOR"
    API_VERSION: str = "1.0.0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
settings = Settings()