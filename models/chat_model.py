# models.py

"""
This file defines the Pydantic data models for the application.
Pydantic is used for data validation and settings management using Python type annotations.
"""

from pydantic import BaseModel, Field
from typing import List

class ChatHistoryModel(BaseModel):
    """
    Represents the chat history and associated user data for a single user.
    This model uses Pydantic for data validation.
    """

    # --- User Identification & Data ---
    # Using the user's phone number as a unique identifier.
    # The pattern ensures it's a valid 10 to 15 digit number.
    phone_number: str = Field(
        ...,
        pattern=r"^\d{10,15}$",
        description="User's 10 to 15 digit phone number, used as a unique ID."
    )

    # User-specific information, similar to the provided example.
    location: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="User's village or pincode."
    )
    curr_crop_name: str = Field(
        ..., 
        min_length=2, 
        max_length=100,
        description="The current crop the user is cultivating."
    )
    irrigation_type: str = Field(
        ..., 
        min_length=2, 
        max_length=50,
        description="The type of irrigation used by the user."
    )

    # --- Embeddings ---
    # Storing embeddings for semantic search and other NLP tasks.
    q_embeddings: List[List[float]] = Field(default_factory=list, description="List of question embeddings.")
    ans_embeddings: List[List[float]] = Field(default_factory=list, description="List of answer embeddings.")

    # --- Recent Conversation Cache ---
    # Caches the last two questions and answers for quick context retrieval.
    # The list is constrained to a maximum of 2 items.
    last_two_qs: List[str] = Field(default_factory=list, max_items=2, description="The text of the last two questions.")
    last_two_ans: List[str] = Field(default_factory=list, max_items=2, description="The text of the last two answers.")

    class Config:
        """
        Pydantic model configuration.
        'json_schema_extra' provides an example for documentation purposes (e.g., in FastAPI docs).
        """
        json_schema_extra = {
            "example": {
                "phone_number": "9876543210",
                "location": "721302",  # Pincode for Kharagpur
                "curr_crop_name": "Rice",
                "irrigation_type": "Canal",
                "q_embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "ans_embeddings": [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
                "last_two_qs": ["What is the weather forecast?", "What is the market price for rice?"],
                "last_two_ans": ["It will be sunny tomorrow.", "The market price is currently stable."]
            }
        }

# # Example of how you might use this model:
# # (This part is for demonstration and would typically be in your application logic)
# if __name__ == '__main__':
#     # Example data that matches the schema
#     data = {
#         "phone_number": "9988776655",
#         "location": "SomeVillage",
#         "curr_crop_name": "Wheat",
#         "irrigation_type": "Drip",
#         "last_two_qs": ["Hello"],
#         "last_two_ans": ["Hi there!"]
#     }

#     try:
#         # Create an instance of the model to validate the data
#         user_chat_instance = ChatHistoryModel(**data)
        
#         print("--- Pydantic Model Validation Successful ---")
#         print(user_chat_instance.json(indent=2))
        
#         # You can access data like attributes
#         print(f"\nUser's Location: {user_chat_instance.location}")
        
#     except Exception as e:
#         print(f"--- Pydantic Model Validation Failed ---")
#         print(e)

#     # Example of invalid data
#     invalid_data = {
#          "phone_number": "123", # Invalid phone number
#          "location": "A", # Too short
#          "curr_crop_name": "Corn",
#          "irrigation_type": "Rainfed"
#     }
    
#     try:
#         ChatHistoryModel(**invalid_data)
#     except Exception as e:
#         print("\n--- Validation failed for incorrect data as expected ---")
#         # This will print the validation errors
#         print(e)
