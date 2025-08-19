# agents/tool_agent_invocation.py

import logging
from typing import AsyncGenerator
import math
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

# --- START: ADDED/ENHANCED FUNCTIONS ---

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
        error_msg = "API key not found. Please set 'WEATHER_API_KEY'."
        logging.error(error_msg)
        return error_msg

    base_url = "https://api.weatherapi.com/v1/current.json"
    params = {"key": api_key, "q": location, "aqi": "no"}

    logging.info(f"Fetching weather for {location}...")
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        weather = response.json().get("current", {})
        logging.info(f"Successfully fetched weather for {location}.")
        return weather
    except Exception as e:
        error_msg = f"An unexpected error occurred while fetching weather: {e}"
        logging.error(error_msg)
        return error_msg

# --- END: ADDED/ENHANCED FUNCTIONS ---


async def invoke_tool_agent_chain(
    rag_chain,
    retriever,
    embeddings_model, # NOTE: Added to support long-term history summary
    query: str,
    phone_number: int
) -> AsyncGenerator[str, None]:
    """Invoke tool RAG chain with full context including history and location."""
    logging.info(f"Tools Agent invoked for query: '{query}'")

    # 1. Fetch user data and chat history
    user_data = await _fetch_user_data(phone_number)
    # chat_history = await _fetch_chat_history(phone_number)

    # Resolve location and fetch weather
    location_raw = user_data.get('location', '')
    location = _resolve_user_location(location_raw)
    logging.debug(f"Resolved user location raw='{location_raw}' -> '{location}'")
    
    crop_name = user_data.get("curr_crop_name", "") or "N/A"
    weather = await _fetch_weather_data(location) if location else "N/A"

    # 2. Retrieve relevant documents from the tool vector store
    docs = retriever.invoke(query)
    context_str = "\n\n".join(d.page_content for d in docs)

    # # 3. Build short-term history text
    # last_qs = chat_history.get("last_two_qs", [])
    # last_ans = chat_history.get("last_two_ans", [])
    # short_pairs = []
    # for i, q in enumerate(last_qs):
    #     ans = last_ans[i] if i < len(last_ans) else ""
    #     short_pairs.append(f"Q: {q}\nA: {ans}")
        
    # short_term_history = "\n---\n".join(short_pairs) if short_pairs else ""

    # # 4. Compute long-term summary
    # long_term_summary = []
    # try:
    #     stored_q_embs = chat_history.get("q_embeddings", []) or []
    #     if stored_q_embs and hasattr(embeddings_model, 'embed_query'):
    #         query_emb = np.array(embeddings_model.embed_query(query), dtype=np.float32).reshape(1, -1)
    #         q_matrix = np.array(stored_q_embs, dtype=np.float32)
            
    #         faiss.normalize_L2(query_emb)
    #         faiss.normalize_L2(q_matrix)

    #         index = faiss.IndexFlatIP(q_matrix.shape[1])
    #         index.add(q_matrix)
            
    #         top_k = min(3, q_matrix.shape[0])
    #         sims, idxs = index.search(query_emb, top_k)
            
    #         for pos, (idx, sim) in enumerate(zip(idxs[0], sims[0])):
    #             long_term_summary.append({"rank": pos + 1, "similarity": float(sim)})

    # except Exception as e:
    #     logging.error(f"Failed to compute long_term_summary: {e}")
    
    # long_term_summary_str = _json.dumps(long_term_summary) if long_term_summary else ""

    # 5. Construct the input for the RAG chain
    chain_input = {
        "query": query,
        "general_context": context_str,
        "location": location,
        "crop_name": crop_name,
        "weather": weather,
       
    }

    # 6. Stream the response
    try:
        async for chunk in rag_chain.astream(chain_input):
            yield chunk
    except Exception as e:
        logging.error(f"Tools RAG chain streaming failed: {e}")
        yield f"Error: {e}"

    logging.info("Tools Agent streaming finished.")