# Centralized configuration & constants

APP_TITLE = "📄 RAG PDF Q&A (Chroma + HuggingFace + Streamlit)"
SIDEBAR_TITLE = "⚙️ Settings"

# I/O
DATA_DIR = "data"
CHROMA_PARENT_DIR = "chroma"

# Environment variables
ENV_HF_TOKEN = "HF_TOKEN"

# Defaults / heuristics
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt2"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_K = 1

# Chunking defaults
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
