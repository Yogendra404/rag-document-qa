# RAG project configuration
DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION_NAME = "my_documents"

# Around 500 words is typically in the requested 300–800 token range.
CHUNK_WORDS = 500
OVERLAP_WORDS = 80
TOP_K = 5

# Local Ollama model. Change this if you install another model.
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
