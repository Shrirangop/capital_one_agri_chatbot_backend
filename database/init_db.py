# database.py

import motor.motor_asyncio
from bson import ObjectId
import config
# It's better to have a separate config file for sensitive data like DB URLs
# import config 

# --- Database Connection ---
# Replace with your actual MongoDB connection string or load from a config file.
MONGO_DB_URL = config.MONGO_DB_URL # Example URL

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DB_URL)
database = client.agriculture_db

# --- Collections ---
users_collection = database.get_collection("users")
chat_histories_collection = database.get_collection("chat_histories")


# --- Helper Functions (Serializers) ---
# These functions convert MongoDB documents (which can contain non-serializable
# types like ObjectId) into standard Python dictionaries.

def user_helper(user) -> dict:
    """Converts a user document from MongoDB to a Python dictionary."""
    return {
        "id": str(user["_id"]),
        "phone_number": user["phone_number"],
        "location": user["location"],
        "curr_crop_name": user["curr_crop_name"],
        "irrigation_type": user["irrigation_type"], # Corrected key from your example
    }

def chat_history_helper(chat_history) -> dict:
    """Converts a chat history document from MongoDB to a Python dictionary."""
    return {
        "id": str(chat_history["_id"]),
        "phone_number": chat_history["phone_number"],
        "location": chat_history["location"],
        "curr_crop_name": chat_history["curr_crop_name"],
        "irrigation_type": chat_history["irrigation_type"],
        "q_embeddings": chat_history["q_embeddings"],
        "ans_embeddings": chat_history["ans_embeddings"],
        "last_two_qs": chat_history["last_two_qs"],
        "last_two_ans": chat_history["last_two_ans"],
    }



