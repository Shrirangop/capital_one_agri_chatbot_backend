# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import runllm_router
from routers.runllm_router import initialize_services_sync
import uvicorn
import logging
import os

# Import the client and database objects from your database.py file
from database import client, database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(
    title="Document Q&A API",
    description="API for processing PDF documents and answering questions using RAG with Google Gemini and Pinecone",
    version="1.0.0"
)

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

@app.on_event("startup")
async def startup_event():
    """
    Initialize services and verify database connection on application startup.
    """
    logging.info("🚀 Starting Document Q&A API...")
    try:
        # Verify MongoDB connection by pinging the admin database.
        # This will raise an exception if the connection fails.
        await client.admin.command('ping')
        logging.info("✅ MongoDB connection successful.")

        # Initialize your other services
        initialize_services_sync()
        logging.info("✅ All services initialized successfully on startup")

    except Exception as e:
        logging.error(f"❌ Failed to connect to MongoDB or initialize services on startup: {str(e)}")
        # Depending on your needs, you might want to exit the application if the DB connection fails
        # For example: raise SystemExit(f"Failed to connect to DB: {e}")

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
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")
