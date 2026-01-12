from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    # Настройки для MySQL
    MYSQL_DB: str = "myapp_db"
    MYSQL_USER: str = "admin"
    MYSQL_PASSWORD: str = "secret1234"
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306

    app_host: str = "localhost" #"0.0.0.0"
    app_port: str = "8080"
    #app_port: str = "8501" #для Streamlit

    SECRET_KEY: str = "SECRET_KEY"
    COOKIE_NAME: str = "auth_token"
    DEBUG: bool = True

    @property
    def DATABASE_URL_pymysql(self):
        return (
            f'mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}'
            f'@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}'
        )

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'  # Игнорировать лишние переменные
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()