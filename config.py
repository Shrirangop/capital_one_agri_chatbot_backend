# /config.py

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Keys and Environment ---

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DOCUMENT_DIRECTORY = os.getenv("DOCUMENT_DIRECTORY", "documents")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Directory for CSV rows specific to capital-csv-index
CAPITAL_CSV_DIRECTORY = os.getenv("CAPITAL_CSV_DIRECTORY", r"E:\\Capital One\\chatbot_backend\\database\\imputed crops")

# Directory containing PDF and other unstructured documents for rag-chatbot-index
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", r"E:\\Capital One\\chatbot_backend\\database\\pdf")
FINANCE_DIRECTORY = os.getenv("FINANCE_DIRECTORY", r"E:\\Capital One\\chatbot_backend\\database\\Finances")
TOOLS_DIRECTORY = os.getenv("TOOLS_DIRECTORY", r"E:\\Capital One\\chatbot_backend\\database\\Tools")
# Directory for disease-related documents
DISEASE_DIRECTORY = os.getenv("DISEASE_DIRECTORY", r"E:\\Capital One\\chatbot_backend\\database\\Disease")

# Optional dedicated index names for finance & tools (else include in PINECONE_INDEX_NAMES)
FINANCE_INDEX_NAME = os.getenv("FINANCE_INDEX_NAME", "capital-finance-index")
TOOLS_INDEX_NAME = os.getenv("TOOLS_INDEX_NAME", "capital-tools-index")
DISEASE_INDEX_NAME = os.getenv("DISEASE_INDEX_NAME", "capital-dis-index")

# Web form base URL for first-time user data capture
USER_FORM_URL = os.getenv("USER_FORM_URL", "https://your-frontend.example.com/farmer-onboarding")




# --- Model and VectorDB Configuration ---
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")  # Backward compatibility (primary index)

# Support multiple Pinecone indexes (comma separated). Example in .env:
# PINECONE_INDEX_NAMES="rag-chatbot-index,capital-csv-index"
_multi_indexes_raw = os.getenv("PINECONE_INDEX_NAMES", "")
if _multi_indexes_raw.strip():
	PINECONE_INDEX_NAMES = [name.strip() for name in _multi_indexes_raw.split(",") if name.strip()]
else:
	# Fallback to just the single legacy index var if multi not provided
	PINECONE_INDEX_NAMES = [p for p in [PINECONE_INDEX_NAME] if p]
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # HuggingFace model for embeddings
LLM_MODEL = "llama3-8b-8192" # More standard model name
EMBEDDING_DIMENSION = 1024      # For 'text-embedding-004'

MONGO_DB_URL = os.getenv("MONGO_DB_URL", "mongodb://localhost:27017")
# --- Text Splitter Configuration ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# --- Retriever Configuration ---
RETRIEVER_SEARCH_TYPE = "mmr" 
RETRIEVER_SEARCH_KWARGS = {"k": 4,"fetch_k": 6, "lambda_mult": 0.8 }

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "youe_api_key_here")

logging.info("Configuration loaded successfully.")