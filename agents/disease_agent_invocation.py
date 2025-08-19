import logging
from typing import AsyncGenerator
from fastapi import UploadFile
import httpx

# The URL for your prediction service
PREDICT_API_URL = "http://127.0.0.1:8000/predict/"

async def predict_disease(crop_name: str, image_file_path: str, lang: str = "en") -> dict:
    """
    Correctly reads an image file from a given path and sends its content 
    to the disease prediction API.
    Returns a dictionary with the prediction result or an error.
    """
    if not image_file_path:
        return {"error": "No image file path was provided."}

    logging.info(f"Predicting disease for crop: {crop_name} from path: {image_file_path}")
    try:
        # --- THE KEY FIX ---
        # 1. Open the file from the path in binary read mode ('rb').
        #    The 'with' statement ensures the file is properly closed.
        with open(image_file_path, "rb") as f:
            # 2. Read the entire file content into bytes.
            image_bytes = f.read()
        
        # 3. Prepare the payload for the multipart/form-data request.
        payload = {"crop_type": crop_name, "lang": lang}
        # The filename in the tuple can be generic as it's part of the request body.
        files = {"image": ("image.jpeg", image_bytes, "image/jpeg")}

        # 4. Make the API call using an async client.
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(PREDICT_API_URL, data=payload, files=files)
            response.raise_for_status()  # Raises an exception for 4xx/5xx errors
            
            # Return the successful JSON response
            return response.json()

    except FileNotFoundError:
        error_msg = f"Error: The file was not found at the path '{image_file_path}'"
        logging.error(error_msg)
        return {"error": error_msg}
    except httpx.HTTPStatusError as http_err:
        error_msg = f"Prediction API returned an error: {http_err.response.status_code} {http_err.response.text}"
        logging.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"An unexpected error occurred while predicting disease: {e}"
        logging.exception(error_msg) # Use .exception to log stack trace
        return {"error": error_msg}


async def invoke_disease_agent_chain(
    rag_chain,
    retriever,
    crop_name: str,
    image_file: str, # Renamed for clarity
) -> AsyncGenerator[str, None]:
    """
    Full chain for the disease agent: predict, retrieve, and generate.
    """
    # 1. Get the disease prediction from the API using the file path.
    prediction_result = await predict_disease(crop_name, image_file)

    # 2. Check if the prediction failed.
    if "error" in prediction_result:
        yield f"Could not analyze the image. Reason: {prediction_result['error']}"
        return

    predicted_disease = prediction_result.get("class")
    if not predicted_disease:
        yield "Analysis failed: The API response did not contain a disease class."
        return

    yield f"**Diagnosis:** I've identified **{predicted_disease}** on your {crop_name} plant.\n\n"
    yield "Searching for treatment information...\n\n"

    # 3. Use the predicted disease to query the RAG chain for a solution.
    query = f"treatment and prevention for {predicted_disease} in {crop_name}"
    logging.info(f"Retrieving documents for RAG query: {query}")

    try:
        retrieved_docs = retriever.invoke(query)
        context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)

        chain_input = {
            "context": context_str,
            "crop_name": crop_name,
            "disease_name": predicted_disease
        }

        # 4. Stream the final response from the RAG chain.
        async for chunk in rag_chain.astream(chain_input):
            yield chunk

    except Exception as e:
        logging.error(f"Disease RAG chain failed: {e}")
        yield "Sorry, I couldn't retrieve treatment information due to an error."