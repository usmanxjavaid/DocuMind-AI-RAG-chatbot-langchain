import os   
from dotenv import load_dotenv

load_dotenv() # it loads your .env file

# LLM
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
LLM_MODEL = 'llama-3.1-8b-instant'
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# Embeddings
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Vectore Database
CHROMA_DIR = './data/vectorstore'
CHROMA_COLLECTION = 'documind'

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval 
TOP_K = 4

# File Uploads
UPLOAD_DIR = './data/uploads'
SUPPORTED_EXTENSIONS = ['.pdf', '.txt', 'docx']

