
import logging
from typing import AsyncGenerator,Union, IO
import math  # may be unused; retained if future numeric ops needed
import json as _json
from database.init_db import users_collection, chat_histories_collection


import requests
import json
import config
import numpy as np

import requests


# Add this to your config.py or constants file
DISEASE_API_URL = "http://127.0.0.1:8000/predict/" # Or your actual deployed API URL

async def _fetch_disease(crop_name: str, image_file: Union[bytes, IO[bytes]], lang: str = "en") -> str:
    """
    Predicts crop disease by sending an image to the disease detection API.

    This function constructs a multipart/form-data request to the /predict/
    endpoint, sends it, and parses the response to return the predicted
    disease class.

    Args:
        crop_name: The type of crop (e.g., "tomato", "potato").
        image_file: The image file as bytes or a file-like object.
        lang: The language for the response (default: "en").

    Returns:
        A string containing the predicted disease name (e.g., "Tomato_Late_blight")
        or a descriptive error message if the request fails.
    """
    logging.info(f"Predicting disease for {crop_name}...")

    # --- API Request ---
    # The 'data' dictionary holds the form fields like 'crop_type' and 'lang'.
    payload = {
        "crop_type": crop_name,
        "lang": lang
    }


    files_payload = {
        "image": ("image.jpg", image_file, "image/jpeg")
    }

    try:
        # --- Execute the Request and Handle Response ---
        # Use requests.post for a POST request and pass form data and files separately.
        response = requests.post(DISEASE_API_URL, data=payload, files=files_payload)
        
        # This will raise an exception for HTTP error codes (4xx or 5xx)
        response.raise_for_status()

        # Parse the JSON response from the API
        data = response.json()

        # Check for application-level errors returned by the API in its JSON response
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
    except requests.exceptions.HTTPError as http_err:
        # Handle specific HTTP errors
        error_msg = f"An HTTP error occurred while predicting disease for {crop_name}: {http_err}. Response: {http_err.response.text}"
        logging.error(error_msg)
        return error_msg
    except requests.exceptions.RequestException as req_err:
        # Handle network-related errors (e.g., connection refused)
        error_msg = f"A network error occurred while predicting disease for {crop_name}: {req_err}"
        logging.error(error_msg)
        return error_msg
    except Exception as e:
        # Catch any other unexpected errors
        error_msg = f"An unexpected error occurred during disease prediction for {crop_name}: {e}"
        logging.error(error_msg)
        return error_msg





from typing import Union, IO # Make sure to import these types

async def invoke_disease_agent_chain(
    rag_chain,
    retriever,
    embeddings_model, 
    crop_name: str,
    phone_number: int,
    image_file: Union[bytes, IO[bytes]] # <-- Add image_file parameter
) -> AsyncGenerator[str, None]:
    
    logging.info(f"Disease Agent invoked for crop: {crop_name}")
    
    # 1. Fetch disease data by calling the API
    # You might want to get the language from the user's profile
    disease = await _fetch_disease(crop_name, image_file, lang="en")

    # If the API call returned an error message, yield it and stop.
    if "error" in disease.lower() or "occurred" in disease.lower():
        yield disease
        return

    # 2. Retrieve relevant documents from the vector store based on the predicted disease

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