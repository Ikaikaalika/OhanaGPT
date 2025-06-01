import os
from pathlib import Path

class Config:
    # General settings
    APP_NAME = "ʻŌhanaGPT"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    UPLOAD_DIR = BASE_DIR / "uploads"
    MODEL_DIR = BASE_DIR / "models"
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/ohana_gpt")
    
    # Model settings
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 4
    DROPOUT = 0.1
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    
    # Training settings
    INCREMENTAL_TRAINING = True
    DEDUPLICATION_THRESHOLD = 0.95
    
    # API settings
    MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {'.ged', '.gedcom'}

config = Config()