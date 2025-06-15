from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Financial Analyst API"
    API_V1_STR: str = "/api/v1"
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Task settings
    TASK_TIMEOUT: int = 300  # 5 minutes
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
