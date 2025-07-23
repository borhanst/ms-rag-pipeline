# config/settings.py
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Model Configurations
EMBEDDING_MODEL = "models/gemini-embedding-001"
# CHAT_MODEL = "gemini-2.5-flash" # Note: Check the correct model identifier
CHAT_MODEL = "gpt-4.1-2025-04-14"
# Document Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 5

BASE_PATH = Path(__file__).resolve().parent.parent

# File Paths
PDF_PATH = BASE_PATH / "data/HSC26-Bangla1st-Paper.pdf" # Or pass as argument


# FAISS_INDEX_PATH = "data/faiss_index"
FAISS_INDEX_PATH = "data/multi/faiss_index"