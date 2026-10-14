from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATA_DIR: str
    CHECKPOINT_DIR: str
    TRAIN_FILE: str
    TEST_FILE: str
    VALID_FILE: str
    VOCAB_THRESHOLD: int
    WINDOW_SIZE: int
    EMBEDDING_DIM: int
    BATCH_SIZE: int
    EPOCHS: int

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', env_nested_delimiter='__', extra='ignore')

cfg = Settings()
