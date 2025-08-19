# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import runllm_router
from routers.runllm_router import initialize_services_sync
from routers import whatsapp_router
from routers import save_user_data_router
import uvicorn
import logging
import os

# Import the client and database objects from your database.py file
from database.init_db import client, database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(
    title="Agri Chatbot Backend",
    description="RAG + multi-index crop/finance/tools assistant",
    version="1.0.0"
)

def _validate_config():
    required_env = [
        "PINECONE_API_KEY", # legacy single index (may be optional if multi provided)
        "MONGO_DB_URL"
    ]
    missing = [k for k in required_env if not os.getenv(k)]
    if missing:
        logging.warning(f"Missing environment variables: {missing}. Service may run with degraded functionality.")
    # Advisory for multi-index setup
    if not os.getenv("PINECONE_INDEX_NAMES"):
        logging.info("PINECONE_INDEX_NAMES not set; using single-index mode or defaults.")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(runllm_router.router, prefix="/api/v1", tags=["documents"])
app.include_router(whatsapp_router.router, prefix="/api/v1", tags=["whatsapp"])
app.include_router(save_user_data_router.router, prefix="/api/v1", tags=["user_data"])

@app.on_event("startup")
async def startup_event():
    """
    Initialize services and verify database connection on application startup.
    """
    logging.info("🚀 Starting Document Q&A API...")
    # Validate config first (non-fatal if missing values)
    _validate_config()

    # Attempt Mongo connection
    try:
        await client.admin.command('ping')
        logging.info("✅ MongoDB connection successful.")
    except Exception as e:
        logging.error(f"❌ MongoDB connection failed: {e}")
        # Do not raise here to allow app to start for endpoints that don't need DB.

    # Initialize other services (vector DBs, LLMs, etc.)
    try:
        initialize_services_sync()
        logging.info("✅ All services initialized successfully on startup")
    except Exception as e:
        logging.error(f"❌ Service initialization failed: {e}")
        # Continue running; specific endpoints may handle missing services gracefully.

@app.on_event("shutdown")
async def shutdown_event():
    """
    Close the database connection gracefully on application shutdown.
    """
    logging.info("🔌 Closing MongoDB connection...")
    client.close()
    logging.info("✅ MongoDB connection closed.")


@app.get("/")
async def root():
    return {
        "message": "Document Q&A API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "status_endpoint": "/api/v1/status"
    }

@app.get("/health")
async def health_check():
    # A more robust health check could also ping the database
    try:
        await client.admin.command('ping')
        db_status = "ok"
    except Exception:
        db_status = "error"
    return {"status": "healthy", "database_status": db_status}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
