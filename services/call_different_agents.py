
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

# Import your services

from services.initialise_vectordb import (
    get_vector_store,
    initialize_pinecone_indexes,
    build_multi_index_retriever,
    build_faiss_multi_index_retriever,
)
from services.initialise_crop_llm import (
    initialize_llm_and_embeddings_for_crop,
    create_rag_chain_for_crop,
    create_multi_index_rag_chain_for_crop,
)
from agents.crop_agent_invocation import invoke_crop_agent_chain
from agents.finance_agent_invocation import invoke_finance_agent_chain
from agents.tool_agent_invocation import invoke_tool_agent_chain
from agents.disease_agent_invocation import invoke_disease_agent_chain

from database.init_db import chat_histories_collection, users_collection


# Core (PDF RAG) globals
llm = None
embeddings_model = None
pinecone_index = None
vector_store = None
rag_chain = None


# Crop agent globals
crop_llm = None
crop_embeddings = None
crop_indexes = None
crop_ensemble_retriever = None
crop_chain = None
# Finance agent globals
finance_llm = None
finance_embeddings = None
finance_index = None
finance_retriever = None
finance_chain = None
tools_llm = None
tools_embeddings = None
tools_index = None
tools_retriever = None
tools_chain = None

disease_rag_chain, disease_retriever, disease_embeddings_model, disease_pinecone_index = None, None, None, None
disease_llm = None


async def _init_crop_services_if_needed():
    global crop_llm, crop_embeddings, crop_indexes, crop_ensemble_retriever, crop_chain
    if crop_chain is not None and crop_ensemble_retriever is not None:
        return
    logging.info("Initializing crop agent services (test endpoint)...")
    crop_llm, crop_embeddings = initialize_llm_and_embeddings_for_crop()
    # Multi-index init (reuses config.PINECONE_INDEX_NAMES)
    try:
        _, index_map = initialize_pinecone_indexes()
        crop_indexes = index_map
        # Use FAISS global re-ranking across indexes for higher quality context
        crop_ensemble_retriever = build_faiss_multi_index_retriever(
            index_map, crop_embeddings, per_index_k=6, final_k=4
        )
        crop_chain = create_multi_index_rag_chain_for_crop(crop_llm, crop_ensemble_retriever)
        logging.info("Crop multi-index FAISS RAG chain initialized.")
    except Exception as e:
        logging.warning(
            f"Multi-index init failed; falling back to single existing vector_store: {e}"
        )
        if vector_store is None:
            raise
        crop_ensemble_retriever = vector_store.as_retriever(
            search_type=config.RETRIEVER_SEARCH_TYPE,
            search_kwargs=config.RETRIEVER_SEARCH_KWARGS,
        )
        crop_chain = create_rag_chain_for_crop(crop_llm, vector_store)

async def _init_finance_services_if_needed():
    global finance_llm, finance_embeddings, finance_index, finance_retriever, finance_chain
    if finance_chain is not None and finance_retriever is not None:
        return
    from services.initialise_finances_llm import initialize_llm_and_embeddings_for_finances, create_rag_chain_for_finances
    from services.initialise_finance_vectordb import init_finance_index, build_finance_retriever
    logging.info("Initializing finance agent services...")
    finance_llm, finance_embeddings = initialize_llm_and_embeddings_for_finances()
    _, idx = init_finance_index()
    finance_index = idx
    finance_retriever = build_finance_retriever(finance_index, finance_embeddings)
    # Build chain (finance chain expects general_context provided in invocation wrapper)
    finance_chain = create_rag_chain_for_finances(finance_llm)
    logging.info("Finance agent initialized.")

async def _init_tools_services_if_needed():
    global tools_llm, tools_embeddings, tools_index, tools_retriever, tools_chain
    if tools_chain is not None and tools_retriever is not None:
        return
    from services.initialise_tool_llm import initialize_llm_and_embeddings_for_tool, create_rag_chain_for_tool
    from services.initialise_tools_vectordb import init_tools_index, build_tools_retriever
    logging.info("Initializing tools agent services...")
    tools_llm, tools_embeddings = initialize_llm_and_embeddings_for_tool()
    _, idx = init_tools_index()
    tools_index = idx
    tools_retriever = build_tools_retriever(tools_index, tools_embeddings)
    tools_chain = create_rag_chain_for_tool(tools_llm)
    logging.info("Tools agent initialized.")

async def _init_disease_services_if_needed():
    global disease_llm,disease_rag_chain, disease_retriever, disease_embeddings_model, disease_pinecone_index
    if disease_rag_chain is not None and disease_retriever is not None:
        return
    from services.initialise_disease_vectordb import init_disease_index, build_disease_retriever
    from services.initialise_disease_llm import initialize_llm_and_embeddings_for_disease, create_rag_chain_for_disease
    logging.info("Initializing disease agent services...")

    disease_llm, disease_embeddings_model = initialize_llm_and_embeddings_for_disease()
    _ ,disease_pinecone_index= init_disease_index()
    disease_retriever = build_disease_retriever(disease_pinecone_index, disease_embeddings_model)
    disease_rag_chain = create_rag_chain_for_disease(disease_llm)
    logging.info("Disease agent initialized.")

    return disease_rag_chain, disease_retriever, disease_embeddings_model


#-----Generate Embeddings and LLMs for Chat history-----

def _generate_embeddings_for_chat_history():
    """Generate embeddings model for chat history."""
    global embeddings_model
    if embeddings_model is None:
        logging.info("Initializing embeddings model for chat history...")
        embeddings_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": 'cpu'}
        )
    return embeddings_model



class CropTestRequest(BaseModel):
    question: str
    user_id: int | None = None
    # Optional overrides for dummy data
    location: str | None = None  # could be pincode
    crop_name: str | None = None
    irrigation_type: str | None = None



async def test_crop_agent(req: CropTestRequest):

   
    """Test the crop agent with dummy user/profile data and return full response.

    Steps:
    - Ensure crop agent services initialized
    - Upsert a dummy user + chat history (if absent)
    - Invoke crop RAG chain with provided question
    - Aggregate streamed chunks into a single answer string
    """
    await _init_crop_services_if_needed()

    print("Here")

    uid = req.user_id or 1234567890
    dummy_user = {
        "phone_number": uid,
    # For testing force location to 'Nagpur' (ignore provided pincode / location)
    "location": "Nagpur",
        "curr_crop_name": req.crop_name or "rice",
        "irrigation_type": req.irrigation_type or "drip",
    }
    # Upsert user
    await users_collection.update_one(
        {"phone_number": uid},
        {"$setOnInsert": dummy_user},
        upsert=True
    )

    # Ensure minimal chat history doc (empty embeddings etc.)
    await chat_histories_collection.update_one(
        {"user_id": uid},
        {"$setOnInsert": {
            "user_id": uid,
            "q_embeddings": [],
            "ans_embeddings": [],
            "last_two_qs": [],
            "last_two_ans": []
        }},
        upsert=True
    )

    

    # Run the crop agent invocation (collect streaming output)
    chunks = []
    async for part in invoke_crop_agent_chain(
        crop_chain,
        crop_ensemble_retriever,
        crop_embeddings,
        req.question,
        phone_number=uid
    ):
        chunks.append(str(part))

    answer = "".join(chunks)

    # Ensure embeddings model is initialized
    _generate_embeddings_for_chat_history()
    # Use the embeddings model to generate question embeddings
    question_embedding = embeddings_model.embed_query(req.question)
    answer_embedding = embeddings_model.embed_query(answer)


# Update chat history with new question and answer embeddings
    await chat_histories_collection.update_one(
        {"user_id": uid},
        {
            "$push": {
               "q_embeddings": {"embed":question_embedding,"question": req.question},
                "ans_embeddings": {"embed":answer_embedding,"answer": answer},
                "last_two_qs": {
                    "$each": [req.question], # Add the new question
                    "$slice": -2             # Keep only the last 2 elements
                },
                "last_two_ans": {
                    "$each": [answer],       # Add the new answer
                    "$slice": -2             # Keep only the last 2 elements
                }
            },
            "$set": {"last_updated": datetime.now()}
        }
    )

    return {"user_id": uid, "answer": answer}

class FinanceTestRequest(BaseModel):
    question: str
    user_id: int | None = None
    location: str | None = None
    crop_name: str | None = None


async def test_finance_agent(req: FinanceTestRequest):
    await _init_finance_services_if_needed()
    uid = req.user_id or 1122334455
    # Upsert minimal user record for contextual fields
    await users_collection.update_one(
        {"phone_number": uid},
        {"$set": {"phone_number": uid, "location": req.location or "Nagpur", "curr_crop_name": req.crop_name or "rice"}},
        upsert=True
    )
    chunks = []
    async for part in invoke_finance_agent_chain(
        finance_chain,
        finance_retriever,
        finance_embeddings,
        req.question,
        phone_number=uid
    ):
        chunks.append(str(part))

    

    answer = "".join(chunks)

    # Ensure embeddings model is initialized
    _generate_embeddings_for_chat_history()
    # Use the embeddings model to generate question embeddings
    question_embedding = embeddings_model.embed_query(req.question)
    answer_embedding = embeddings_model.embed_query(answer)

    
# Update chat history with new question and answer embeddings
    await chat_histories_collection.update_one(
        {"user_id": uid},
        {
            "$push": {
                "q_embeddings": {"embed":question_embedding,"question": req.question},
                "ans_embeddings": {"embed":answer_embedding,"answer": answer},
                "last_two_qs": {
                    "$each": [req.question], # Add the new question
                    "$slice": -2             # Keep only the last 2 elements
                },
                "last_two_ans": {
                    "$each": [answer],       # Add the new answer
                    "$slice": -2             # Keep only the last 2 elements
                }
            },
            "$set": {"last_updated": datetime.now()}
        }
    )
    return {"user_id": uid, "answer": "".join(chunks)}

class ToolsTestRequest(BaseModel):
    question: str
    user_id: int | None = None
    location: str | None = None
    crop_name: str | None = None


async def test_tools_agent(req: ToolsTestRequest):
    await _init_tools_services_if_needed()
    uid = req.user_id or 6677889900
    await users_collection.update_one(
        {"phone_number": uid},
        {"$set": {"phone_number": uid, "location": req.location or "Nagpur", "curr_crop_name": req.crop_name or "rice"}},
        upsert=True
    )
    chunks = []
    async for part in invoke_tool_agent_chain(
        tools_chain,
        tools_retriever,
        tools_embeddings,
        req.question,
        phone_number=uid
    ):
        chunks.append(str(part))


    answer = "".join(chunks)

    # Ensure embeddings model is initialized
    _generate_embeddings_for_chat_history()
    # Use the embeddings model to generate question embeddings
    question_embedding = embeddings_model.embed_query(req.question)
    answer_embedding = embeddings_model.embed_query(answer)

    
# Update chat history with new question and answer embeddings
    await chat_histories_collection.update_one(
        {"user_id": uid},
        {
            "$push": {
                "q_embeddings": {"embed":question_embedding,"question": req.question},
                "ans_embeddings": {"embed":answer_embedding,"answer": answer},
                "last_two_qs": {
                    "$each": [req.question], # Add the new question
                    "$slice": -2             # Keep only the last 2 elements
                },
                "last_two_ans": {
                    "$each": [answer],       # Add the new answer
                    "$slice": -2             # Keep only the last 2 elements
                }
            },
            "$set": {"last_updated": datetime.now()}
        }
    )

    
    return {"user_id": uid, "answer": "".join(chunks)}


async def diagnose_crop(
    crop_name: str = Form(..., description="The name of the crop, e.g., 'tomato'"),
    phone_number: int = Form(..., description="The user's phone number for context/logging."),
    image: UploadFile = File(..., description="The image file of the crop to be diagnosed.")
):
    """
    Accepts crop information and an image to diagnose a potential disease.

    This endpoint invokes the RAG agent, collects the full response, and
    returns it as a single JSON object.
    """
    logging.info(f"Received diagnosis request for crop: {crop_name}")
    image_bytes = await image.read()

    # Ensure disease services are initialized
    await _init_disease_services_if_needed()

    # 1. Create an empty list to hold the response chunks.
    response_chunks = []

    # 2. Asynchronously iterate through the generator and append each chunk.
    async for chunk in invoke_disease_agent_chain(
        rag_chain=disease_rag_chain,
        retriever=disease_retriever,
        embeddings_model=disease_embeddings_model,
        crop_name=crop_name,
        phone_number=phone_number,
        image_file=image_bytes
    ):
        response_chunks.append(chunk)

    # 3. Join the chunks into a single final string.
    final_diagnosis_text = "".join(response_chunks)

    # 4. Return the complete string in a JSON object.
    # FastAPI automatically handles the conversion from a Python dict to a JSON response.
    return {"diagnosis": final_diagnosis_text}
