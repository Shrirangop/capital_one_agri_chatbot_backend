import logging
from typing import AsyncGenerator,Union, IO
import math  # may be unused; retained if future numeric ops needed
import json as _json
from database.init_db import users_collection, chat_histories_collection


from fastapi import UploadFile
import requests
import json
import config
import numpy as np
import aiohttp


import httpx
from typing import Union, IO

# You'll need to install httpx:
# pip install httpx

# Assume DISEASE_API_URL is defined elsewhere, e.g.:
DISEASE_API_URL = "http://127.0.0.1:8000/predict/"

async def _fetch_disease(crop_name: str, image_file: Union[str, UploadFile], lang: str = "en") -> str:
    """
    Predicts crop disease by sending an image to the disease detection API asynchronously.
    """
    logging.info(f"Predicting disease for {crop_name}...")

    payload = {
        "crop_type": crop_name,
        "lang": lang
    }

    files = {
    "image": ("image.jpeg", image_file, "image/jpeg")
    }

    


    
    

    # Use an asynchronous client with a context manager
    async with httpx.AsyncClient() as client:
        try:
            # --- Execute the Request and Handle Response ---
            # Use 'await' for the asynchronous network call
            response = await client.post(DISEASE_API_URL, data=payload, files=files)
            
            # This will raise an exception for HTTP error codes (4xx or 5xx)
            response.raise_for_status()

            # Parse the JSON response from the API
            data = response.json()

            # --- Process Response ---
            if "error" in data:
                error_msg = f"API returned an error for {crop_name}: {data['error']}"
                logging.error(error_msg)
                return error_msg

            predicted_class = data.get("class")
            if not predicted_class:
                error_msg = f"API response for {crop_name} is missing the 'class' key."
                logging.error(error_msg)
                return error_msg

            logging.info(f"Successfully predicted disease for {crop_name}: {predicted_class}")
            return predicted_class

        # --- Error Handling ---
        # Updated to use httpx's specific exceptions
        except httpx.HTTPStatusError as http_err:
            error_msg = f"An HTTP error occurred for {crop_name}: {http_err}. Response: {http_err.response.text}"
            logging.error(error_msg)
            return error_msg
        except httpx.RequestError as req_err:
            error_msg = f"A network error occurred for {crop_name}: {req_err}"
            logging.error(error_msg)
            return error_msg
        except Exception as e:
            logging.exception(f"An unexpected error occurred for {crop_name}: {e}")
            error_msg = f"An unexpected error occurred for {crop_name}: {e}"
            logging.error(error_msg)
            return error_msg






async def invoke_disease_agent_chain(
    rag_chain,
    retriever,
    embeddings_model, 
    crop_name: str,
    phone_number: int,
    image_file: Union[str, UploadFile, None] = None # <-- Add image_file parameter
) -> AsyncGenerator[str, None]:
    
    logging.info(f"Disease Agent invoked for crop: {crop_name}")
    
    # 1. Fetch disease data by calling the API
    # You might want to get the language from the user's profile
    # disease = await _fetch_disease(crop_name, image_file, lang="en")

    disease = "GROUNDNUT LEAF SPOT (EARLY AND LATE)"

    

    # If the API call returned an error message, yield it and stop.
    if "error" in disease.lower() or "occurred" in disease.lower():
        yield disease
        return

    # 2. Retrieve relevant documents from the vector store based on the predicted disease

    crop_name = 'groundnut'

    query = f"{disease} for {crop_name}"


    logging.info(f"Retrieving documents for disease: {disease} and crop: {crop_name}")

    retrieved_docs = retriever.invoke(query)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    context_str = format_docs(retrieved_docs)

    # ... (rest of your function remains the same)
    
    # 5. Construct the input for the RAG chain

    logging.info("Constructing input for RAG chain...")
   
   

    chain_input = {
        "context": context_str,
        "crop_name": crop_name,
        "disease_name": disease
        # ... other parameters
    }

    # 6. Stream the response
    try:
        async for chunk in rag_chain.astream(chain_input):
            yield chunk
    except Exception as e:
        logging.error(f"Disease RAG chain streaming failed: {e}")
        yield f"Error: {e}"

    logging.info("Disease Agent streaming finished.")

# async def invoke_disease_agent_chain(
#     disease_rag_chain, disease_retriever, disease_embeddings,
#     crop_name: str, user_id: int, image_file: bytes
# ):
#     """
#     Calls the disease detection API with crop_name and image, yields the result.
#     """
#     DISEASE_API_URL = getattr(config, "DISEASE_API_URL", "http://127.0.0.1:8000/predict/")
#     lang = "en"  # or get from user profile

#     data = aiohttp.FormData()
#     data.add_field('crop_type', crop_name)
#     data.add_field('lang', lang)
#     data.add_field('image', image_file, filename="crop.jpg", content_type="image/jpeg")

#     async with aiohttp.ClientSession() as session:
#         async with session.post(DISEASE_API_URL, data=data) as resp:
#             if resp.status == 200:
#                 result = await resp.json()
#                 yield f"Disease prediction: {result.get('class', 'Unknown')} (confidence: {result.get('confidence', 0):.2f})"
#             else:
#                 yield f"Failed to get disease prediction. Status: {resp.status}"