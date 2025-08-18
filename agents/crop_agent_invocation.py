# agents/crop_agent_invocation.py

import logging
from typing import AsyncGenerator
import math  # may be unused; retained if future numeric ops needed
import json as _json
from database.init_db import users_collection, chat_histories_collection


import requests
import json
import config
import numpy as np
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:  # optional dependency
    _FAISS_AVAILABLE = False

async def _fetch_chat_history(phone_number: int) -> dict:
    """Fetch the user's chat history from the database."""
    logging.info(f"Fetching chat history for user {phone_number}...")
    chat_history = await chat_histories_collection.find_one({'user_id': phone_number})
    # print(f"Chat history fetched: {chat_history}")

    if not chat_history:
        logging.error(f"No chat history found for user {phone_number}.")

        return {}
    

    return {
        "q_embeddings": chat_history.get("q_embeddings", []),
        "ans_embeddings": chat_history.get("ans_embeddings", []),
        "last_two_qs": chat_history.get("last_two_qs", []),
        "last_two_ans": chat_history.get("last_two_ans", [])
    }

async def _fetch_user_data(phone_number: int) -> dict:
    """Fetch user data from the database."""
    logging.info(f"Fetching user data for user {phone_number}...")
    user = await users_collection.find_one({'phone_number': phone_number})

    print(f"User data fetched: {user}")


    if not user:
        logging.error(f"User {phone_number} not found in database.")
        return {}
    return {
        "phone_number": user.get("phone_number", ""),
        "location": user.get("location", ""),
        "curr_crop_name": user.get("curr_crop_name", ""),
    }
# --- END MODIFIED SECTION ---

def _fetch_location_data(pincode: str) -> str:
    """
    Fetches the location (District, State) for a given Indian pincode.

    Args:
        pincode: A string representing the 6-digit Indian pincode.

    Returns:
        A string in the format "District, State" if the pincode is valid,
        otherwise returns an error message.
    """
    logging.debug(f"Fetching location for pincode: {pincode}")
    url = f"https://api.postalpincode.in/pincode/{pincode}"
    try:
        response = requests.get(url, timeout=6)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            data = data[0]
        if data.get("Status") == "Success" and data.get("PostOffice"):
            po = data["PostOffice"][0]
            name = po.get("Name", "")
            district = po.get("District", "")
            state = po.get("State", "")
            resolved = ", ".join([v for v in [name, district, state] if v])
            logging.debug(f"Resolved pincode {pincode} -> {resolved}")
            return resolved
        logging.debug(f"Pincode {pincode} unresolved (status={data.get('Status')})")
        return ""
    except Exception as e:
        logging.debug(f"Pincode lookup failed for {pincode}: {e}")
        return ""

def _resolve_user_location(raw_location: str) -> str:
    """Return best human-readable location.

    If a 6-digit numeric pincode -> resolve via postal API. Else treat as given (e.g. 'Nagpur').
    """
    if not raw_location:
        return ""
    cleaned = str(raw_location).strip()
    if cleaned.isdigit() and len(cleaned) == 6:
        looked = _fetch_location_data(cleaned)
        return looked or cleaned
    return cleaned
    



async def _fetch_weather_data(location: str) -> str:
    """
    Fetches current weather data for a given location using the WeatherAPI.

    This function constructs a request to the WeatherAPI endpoint, sends it,
    and parses the response to return a human-readable weather summary.

    Args:
        location: The name of the city or district (e.g., "Bhubaneshwar", "Paris").

    Returns:
        A string containing the formatted weather information (e.g., 
        "Partly cloudy, 32.0°C, 85% humidity.") or a descriptive error message 
        if the request fails.
    """
    # --- API Key Management ---
    # Securely retrieve the API key from an environment variable.
    # Avoid hardcoding sensitive keys directly in your code.
    # You'll need to set this environment variable in your system.
    # For example, in your terminal: export WEATHER_API_KEY='your_key_here'
    api_key = config.WEATHER_API_KEY
    if not api_key:
        error_msg = "API key not found. Please set the 'WEATHER_API_KEY' environment variable."
        logging.error(error_msg)
        return error_msg

    # --- API Request ---
    base_url = "https://api.weatherapi.com/v1/current.json"
    
    # Parameters for the GET request
    params = {
        "key": api_key,
        "q": location,
        "aqi": "no"  # As specified in the example URL
    }

    logging.info(f"Fetching weather for {location}...")

    try:
        # --- Execute the Request and Handle Response ---
        response = requests.get(base_url, params=params)
        
        # This will raise an exception for HTTP error codes (4xx or 5xx)
        response.raise_for_status()

        # Parse the JSON response from the API
        data = response.json()

        weather = data.get("current", {})

        print(location)

        print(f"Weather data fetched: {weather}")

       
        logging.info(f"Successfully fetched weather for {location}.")
        return weather

    # --- Error Handling ---
    except requests.exceptions.HTTPError as http_err:
        # Handle specific HTTP errors, like location not found (400) or invalid key (401)
        if response.status_code == 400:
            error_details = response.json().get("error", {}).get("message", "No additional details.")
            error_msg = f"Could not find weather for '{location}'. API Error: {error_details}"
            logging.warning(error_msg)
            return error_msg
        else:
            error_msg = f"An HTTP error occurred: {http_err}"
            logging.error(error_msg)
            return error_msg
    except requests.exceptions.RequestException as req_err:
        # Handle network-related errors (e.g., no internet connection)
        error_msg = f"A network error occurred: {req_err}"
        logging.error(error_msg)
        return error_msg
    except Exception as e:
        # Catch any other unexpected errors
        error_msg = f"An unexpected error occurred: {e}"
        logging.error(error_msg)
        return error_msg

async def invoke_crop_agent_chain(
    rag_chain,
    ensemble_retriever,
    embeddings_model,
    query: str,
    phone_number: int# Example user ID
) -> AsyncGenerator[str, None]:
    """
    Gathers all necessary context, including chat history, and invokes the Crop RAG chain.

    Args:
        rag_chain: The initialized, runnable RAG chain for the Crop Agent.
        ensemble_retriever: The retriever for fetching context documents.
        query (str): The user's question.
        user_id (str): The ID of the user to fetch personalized context.

    Yields:
        str: Chunks of the response as they are generated by the LLM.
    """
    logging.info(f"Crop Agent invoked for query: '{query}'")

    # 1. Fetch user data and chat history
    user_data = await _fetch_user_data(phone_number)
    chat_history = await _fetch_chat_history(phone_number)

    # Resolve location (prefer explicit name; only lookup if pincode)
    # location_raw = user_data.get('location', '')
    # location = _resolve_user_location(location_raw)
    # logging.debug(f"Resolved user location raw='{location_raw}' -> '{location}'")

    location = 'Kanpur'
    crop_name = user_data['curr_crop_name']
    weather = await _fetch_weather_data(location)

    
    # 2. Retrieve relevant documents from the vector store
    retrieved_docs = ensemble_retriever.invoke(query)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    context_str = format_docs(retrieved_docs)

    # 3. Build short-term history (last two Q/A pairs) text
    last_qs = chat_history.get("last_two_qs", [])
    last_ans = chat_history.get("last_two_ans", [])
    short_pairs = []
    for i, q in enumerate(last_qs):
        ans = last_ans[i] if i < len(last_ans) else ""
        short_pairs.append(f"Q: {q}\nA: {ans}")
    short_term_history = "\n---\n".join(short_pairs) if short_pairs else ""

    # 4. Compute long-term summary using FAISS if available, else fallback cosine loop
    long_term_summary = []
    try:
        stored_q_embs = chat_history.get("q_embeddings", []) or []
        stored_a_embs = chat_history.get("ans_embeddings", []) or []
        top_k = 3  # limit to top 3
        if stored_q_embs and hasattr(embeddings_model, 'embed_query'):
            # Build query embedding
            query_emb = np.array(embeddings_model.embed_query(query), dtype=np.float32)
            q_matrix = np.array(stored_q_embs, dtype=np.float32)
            # Normalize (safety) for inner product similarity
            def _normalize(mat):
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return mat / norms
            query_emb = _normalize(query_emb.reshape(1, -1))[0]
            q_matrix = _normalize(q_matrix)
            if _FAISS_AVAILABLE and q_matrix.shape[0] >= 1:
                index = faiss.IndexFlatIP(q_matrix.shape[1])
                index.add(q_matrix)
                sims, idxs = index.search(query_emb.reshape(1, -1), min(top_k, q_matrix.shape[0]))
                for pos, (idx, sim) in enumerate(zip(idxs[0], sims[0])):
                    entry = {
                        "rank": pos + 1,
                        "q_index": int(idx),
                        "similarity": float(sim),
                        "q_embedding": stored_q_embs[idx],
                        "ans_embedding": stored_a_embs[idx] if idx < len(stored_a_embs) else None
                    }
                    long_term_summary.append(entry)
                # Combined (averaged) embedding of top 3
                if long_term_summary:
                    try:
                        top_q_matrix = np.array([e["q_embedding"] for e in long_term_summary], dtype=np.float32)
                        combined_q = top_q_matrix.mean(axis=0).tolist()
                        top_a_embs = [e["ans_embedding"] for e in long_term_summary if e.get("ans_embedding") is not None]
                        combined_a = None
                        if top_a_embs:
                            combined_a = np.array(top_a_embs, dtype=np.float32).mean(axis=0).tolist()
                        long_term_summary.append({
                            "combined": True,
                            "count": len(long_term_summary),
                            "q_indices": [e["q_index"] for e in long_term_summary if not e.get("combined")],
                            "q_embedding": combined_q,
                            "ans_embedding": combined_a
                        })
                    except Exception as _ce:
                        logging.warning(f"Failed to compute combined top-3 embedding: {_ce}")
            else:
                # Fallback manual scoring
                sims = []
                for idx, vec in enumerate(q_matrix):
                    sims.append((idx, float(np.dot(query_emb, vec))))
                selected = sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]
                for pos, (idx, sim) in enumerate(selected):
                    entry = {
                        "rank": pos + 1,
                        "q_index": int(idx),
                        "similarity": float(sim),
                        "q_embedding": stored_q_embs[idx],
                        "ans_embedding": stored_a_embs[idx] if idx < len(stored_a_embs) else None
                    }
                    long_term_summary.append(entry)
                if long_term_summary:
                    try:
                        top_q_matrix = np.array([e["q_embedding"] for e in long_term_summary], dtype=np.float32)
                        combined_q = top_q_matrix.mean(axis=0).tolist()
                        top_a_embs = [e["ans_embedding"] for e in long_term_summary if e.get("ans_embedding") is not None]
                        combined_a = None
                        if top_a_embs:
                            combined_a = np.array(top_a_embs, dtype=np.float32).mean(axis=0).tolist()
                        long_term_summary.append({
                            "combined": True,
                            "count": len(long_term_summary),
                            "q_indices": [e["q_index"] for e in long_term_summary if not e.get("combined")],
                            "q_embedding": combined_q,
                            "ans_embedding": combined_a
                        })
                    except Exception as _ce:
                        logging.warning(f"Failed to compute combined top-3 embedding (fallback): {_ce}")
    except Exception as e:
        logging.error(f"Failed to compute long_term_summary (FAISS stage): {e}")

    # Represent long_term_summary as JSON string (LLM-safe) if list not empty
    long_term_summary_str = _json.dumps(long_term_summary) if long_term_summary else ""


    

    # 5. Construct the input object for the RAG chain (align with prompt variables)
    chain_input = {
        "query": query,
        "location": location,
        "crop_name": crop_name,
        "weather": weather,
        "general_context": context_str,  # matches create_rag_chain_for_crop
        "short_term_history": short_term_history,
        "long_term_summary": long_term_summary_str
    }

    # 6. Stream the response from the RAG chain
    try:
        async for chunk in rag_chain.astream(chain_input):
            yield chunk
    except Exception as e:
        logging.error(f"RAG chain streaming failed: {e}")
        yield f"Error generating response: {e}"

    logging.info("Crop Agent streaming finished.")