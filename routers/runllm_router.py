# routers/hackrx.py

import os
import tempfile
import requests
import secrets
import logging
from typing import List
from fastapi import APIRouter, HTTPException, status, Depends, File, Form, UploadFile
from pydantic import BaseModel, HttpUrl
from langchain_community.document_loaders import PyPDFLoader  # updated per deprecation warning
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # updated per deprecation warning
from fastapi.security import APIKeyHeader
from langchain_huggingface import HuggingFaceEmbeddings
import config

from datetime import datetime

from agents.brain_agent_invocation import route_query
from services.brain_agent_chain import initialize_llm_and_embeddings_brain

from database.init_db import users_collection, chat_histories_collection

# This tells FastAPI to look for a header named "Authorization"
api_key_header_scheme = APIKeyHeader(name="Authorization", auto_error=False)

async def get_api_key(auth_header: str = Depends(api_key_header_scheme)):
    """Dependency that extracts and validates the API key."""
    if auth_header is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header is missing.")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication scheme. Must be 'Bearer'.")
    
    api_key = auth_header.removeprefix("Bearer ")
    if not secrets.compare_digest(api_key, config.VALID_API_KEY):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key.")
    return api_key

# Initialize router
router = APIRouter()

# Pydantic models for request/response
class DocumentRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]
    
class QuestionAnswer(BaseModel):
    question: str
    answer: str
    
class DocumentResponse(BaseModel):
    status: str
    message: str
    results: List[QuestionAnswer]
    total_questions: int

# Core (PDF RAG) globals
llm = None
embeddings_model = None
pinecone_index = None
vector_store = None
rag_chain = None
brain_llm = None



def initialize_services_sync():
    """Placeholder core initialization (currently using crop multi-index only)."""
    if rag_chain is not None:
        return  # already initialized
    logging.info("Core RAG chain not configured; relying on crop agent endpoints only.")







# --- NON-STREAMING LOGIC (OPTIMIZED) ---
async def process_document_and_get_answers(request: DocumentRequest) -> dict:
    """
    Processes the document asynchronously, collects all answers in a single batch,
    and returns a single dictionary.
    """
    pdf_filename = None
    final_result = {
        "answers": []
    }

    try:
        # --- 1. Document Setup and Processing ---
        logging.info("Downloading document...")
        pdf_response = requests.get(str(request.documents), timeout=30)
        pdf_response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_response.content)
            pdf_filename = tmp_file.name

        logging.info("Processing document...")
        loader = PyPDFLoader(pdf_filename)
        docs = loader.load()
        if not docs:
            raise ValueError("Could not extract any content from the provided PDF.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise ValueError("Document content was empty or could not be split into chunks.")

        logging.info(f"Creating vector store for {len(chunks)} chunks...")
        vector_store = await FAISS.afrom_documents(chunks, embeddings_model)
        doc_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4,"fetch_k": 6, "lambda_mult": 0.8  })

        # --- 2. Batch Question Answering ---
        logging.info(f"Preparing to answer {len(request.questions)} questions in a batch...")

        # Prepare a list of inputs for the batch call
        batch_inputs = [
            {"question": q, "query_doc_retriever": doc_retriever}
            for q in request.questions
        ]

        # Invoke the chain with .batch() for parallel processing
        batch_results = rag_chain.batch(batch_inputs)

        # Structure the final result
        final_result["answers"] = [
             batch_results[i]
            for i in range(len(batch_results))
        ]
        logging.info("✅ All questions answered successfully.")


    except Exception as e:
        logging.error(f"A critical error occurred during document processing: {e}")
        return {"error": f"A critical error occurred: {str(e)}"}
    finally:
        # --- 3. Cleanup ---
        if pdf_filename and os.path.exists(pdf_filename):
            os.remove(pdf_filename)

    # --- 4. Return Final Result ---
    return final_result
# --- How to Use It in Your Endpoint ---

@router.post("/rag/document")
async def process_document_sync(request: DocumentRequest, api_key: str = Depends(get_api_key)):
    """
    Processes a PDF and returns a single JSON object with all answers.
    """
    if rag_chain is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Services are not initialized.")
    
    # Await the function and get the final dictionary
    result_dict = await process_document_and_get_answers(request)

    # Check if the result is an error object
    if "error" in result_dict:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result_dict["error"])

    # FastAPI will automatically convert the dictionary to a JSON response
    return result_dict

    # Removed legacy blocking endpoint in cleanup.

# @router.get("/status")
# async def get_status():
#     return {
#         "core_initialized": rag_chain is not None,
#         "crop_initialized": crop_chain is not None,
#         "indexes_loaded": bool(pinecone_index),
#         "multi_index": crop_ensemble_retriever is not None
#     }

class BrainAgentRequest(BaseModel):
    question: str
    user_id: int | None = None
    image_file: bytes | None = None  # For potential image uploads

@router.post("/ask")
async def ask_brain_agent(req: BrainAgentRequest):
    global brain_llm

    brain_llm = initialize_llm_and_embeddings_brain()

    # Collect the streamed response
    chunks = []
    async for chunk in route_query(
        req.question,
        brain_llm,
        req.user_id,
        req.image_file
    ):
        chunks.append(str(chunk))

    return {"user_id": req.user_id, "answer": "".join(chunks)}