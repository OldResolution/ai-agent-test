import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "qwen3:8b")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

settings = Settings()
