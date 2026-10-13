from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATA_DIR: str
    TRAIN_FILE: str
    TEST_FILE: str
    VALID_FILE: str
    WINDOW_SIZE: int

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', env_nested_delimiter='__', extra='ignore')

cfg = Settings()
