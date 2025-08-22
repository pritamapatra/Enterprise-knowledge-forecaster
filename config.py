import os

# ChromaDB Configuration
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "uploaded_documents"

# Text Processing Configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
CHUNK_SEPARATORS = ["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Query Configuration
MAX_RESULTS = 5
SIMILARITY_THRESHOLD = 0.5

# File Upload Configuration
SUPPORTED_FILE_TYPES = ['pdf', 'txt', 'docx']
MAX_FILE_SIZE_MB = 10

# UI Configuration
APP_TITLE = "Enterprise Knowledge Evolution Forecaster"
APP_ICON = "📊"
