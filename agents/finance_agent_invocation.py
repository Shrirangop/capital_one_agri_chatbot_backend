# agents/finance_agent_invocation.py

import logging
from typing import AsyncGenerator
import math  # may be unused; retained for consistency
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

# --- START: ADDED FUNCTIONS FROM CROP AGENT ---

async def _fetch_chat_history(phone_number: int) -> dict:
    """Fetch the user's chat history from the database."""
    logging.info(f"Fetching chat history for user {phone_number}...")
    chat_history = await chat_histories_collection.find_one({'user_id': phone_number})
    
    if not chat_history:
        logging.warning(f"No chat history found for user {phone_number}.")
        return {}
    
    return {
        "q_embeddings": chat_history.get("q_embeddings", []),
        "ans_embeddings": chat_history.get("ans_embeddings", []),
        "last_two_qs": chat_history.get("last_two_qs", []),
        "last_two_ans": chat_history.get("last_two_ans", [])
    }

async def _fetch_user_data(phone_number: int) -> dict:
    """Fetch user data from the database (enhanced version)."""
    logging.info(f"Fetching user data for user {phone_number}...")
    user = await users_collection.find_one({'phone_number': phone_number})

    if not user:
        logging.error(f"User {phone_number} not found in database.")
        return {}
    return {
        "phone_number": user.get("phone_number", ""),
        "location": user.get("location", ""),
        "curr_crop_name": user.get("curr_crop_name", ""),
    }

def _fetch_location_data(pincode: str) -> str:
    """
    Fetches the location (District, State) for a given Indian pincode.
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
    """Return best human-readable location."""
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
    """
    api_key = config.WEATHER_API_KEY
    if not api_key:
        error_msg = "API key not found. Please set the 'WEATHER_API_KEY' environment variable."
        logging.error(error_msg)
        return error_msg

    base_url = "https://api.weatherapi.com/v1/current.json"
    params = {
        "key": api_key,
        "q": location,
        "aqi": "no"
    }

    logging.info(f"Fetching weather for {location}...")
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        weather = data.get("current", {})
        logging.info(f"Successfully fetched weather for {location}.")
        return weather
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 400:
            error_details = response.json().get("error", {}).get("message", "No additional details.")
            error_msg = f"Could not find weather for '{location}'. API Error: {error_details}"
            logging.warning(error_msg)
            return error_msg
        else:
            error_msg = f"An HTTP error occurred: {http_err}"
            logging.error(error_msg)
            return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        logging.error(error_msg)
        return error_msg

# --- END: ADDED FUNCTIONS ---


async def invoke_finance_agent_chain(
    rag_chain,
    retriever,
    embeddings_model, # NOTE: Added to support long-term history summary
    query: str,
    phone_number: int
) -> AsyncGenerator[str, None]:
    """Invoke finance RAG chain with full context including history and location."""
    logging.info(f"Finance Agent invoked for query: '{query}'")

    # 1. Fetch user data and chat history
    user_data = await _fetch_user_data(phone_number)
    chat_history = await _fetch_chat_history(phone_number)

    # Resolve location from pincode if necessary
    location_raw = user_data.get('location', '')
    location = _resolve_user_location(location_raw)
    logging.debug(f"Resolved user location raw='{location_raw}' -> '{location}'")
    
    crop_name = user_data.get("curr_crop_name", "") or "N/A"
    weather = await _fetch_weather_data(location) if location else "N/A"

    # 2. Retrieve relevant documents from the finance vector store
    docs = retriever.invoke(query)
    context_str = "\n\n".join(d.page_content for d in docs)

    # 3. Build short-term history text
    last_qs = chat_history.get("last_two_qs", [])
    last_ans = chat_history.get("last_two_ans", [])
    short_pairs = []
    for i, q in enumerate(last_qs):
        ans = last_ans[i] if i < len(last_ans) else ""
        short_pairs.append(f"Q: {q}\nA: {ans}")
    short_term_history = "\n---\n".join(short_pairs) if short_pairs else ""

    # 4. Compute long-term summary
    long_term_summary = []
    centroid_embedding = None
    try:
        stored_q_embs = chat_history.get("q_embeddings", []) or []
        stored_a_embs = chat_history.get("ans_embeddings", []) or []
        top_k = 3
        if stored_q_embs and hasattr(embeddings_model, 'embed_query'):
            query_emb = np.array(embeddings_model.embed_query(query), dtype=np.float32)
            q_matrix = np.array(stored_q_embs, dtype=np.float32)

            def _normalize(mat):
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return mat / norms

            query_emb = _normalize(query_emb.reshape(1, -1))[0]
            q_matrix = _normalize(q_matrix)

            selected_idxs = []
            if _FAISS_AVAILABLE and q_matrix.shape[0] >= 1:
                index = faiss.IndexFlatIP(q_matrix.shape[1])
                index.add(q_matrix)
                sims, idxs = index.search(query_emb.reshape(1, -1), min(top_k, q_matrix.shape[0]))
                selected_idxs = list(idxs[0])
                for pos, (idx, sim) in enumerate(zip(idxs[0], sims[0])):
                    long_term_summary.append({
                        "rank": pos + 1,
                        "q_index": int(idx),
                        "similarity": float(sim),
                        "q_embedding": stored_q_embs[idx],
                        "ans_embedding": stored_a_embs[idx] if idx < len(stored_a_embs) else None
                    })
            else:  # Fallback without FAISS
                sims = [(idx, float(np.dot(query_emb, vec))) for idx, vec in enumerate(q_matrix)]
                selected = sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]
                selected_idxs = [idx for idx, _ in selected]
                for pos, (idx, sim) in enumerate(selected):
                    long_term_summary.append({
                        "rank": pos + 1,
                        "q_index": int(idx),
                        "similarity": float(sim),
                        "q_embedding": stored_q_embs[idx],
                        "ans_embedding": stored_a_embs[idx] if idx < len(stored_a_embs) else None
                    })
            # Compute centroid of selected embeddings
            if selected_idxs:
                selected_embs = np.array([stored_q_embs[idx] for idx in selected_idxs], dtype=np.float32)
                centroid_embedding = selected_embs.mean(axis=0).tolist()
    except Exception as e:
        logging.error(f"Failed to compute long_term_summary: {e}")

    # You can now include the centroid in your output if needed:
    output = {
        "long_term_summary": long_term_summary,
        "centroid_embedding": centroid_embedding
    }
    # print(f"Long-term summary: {context_str}")  # Debugging output

    # 5. Construct the input for the RAG chain
    chain_input = {
        "query": query,
        "general_context": context_str,
        "location": location,
        "crop_name": crop_name,
        "weather": weather,
        "short_term_history": short_term_history,
        "long_term_summary": centroid_embedding
    }

    # 6. Stream the response
    try:
        async for chunk in rag_chain.astream(chain_input):
            yield chunk
    except Exception as e:
        logging.error(f"Finance RAG chain streaming failed: {e}")
        yield f"Error: {e}"

    logging.info("Finance Agent streaming finished.")