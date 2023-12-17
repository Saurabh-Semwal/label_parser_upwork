from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    aws_access_key_id: str
    aws_secret_access_key: str
    s3_bucket_name: str
    openai_api_key: str
    aws_region_name: str
    
    class Config:
        env_file = ".env"

settings = Settings()
