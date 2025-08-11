# models.py

from pydantic import BaseModel, Field

class UserModel(BaseModel):
    # Phone numbers are best stored as strings with a validation pattern
    phone_number: str = Field(
        ...,
        pattern=r"^\d{10,15}$",
        description="User's 10 to 15 digit phone number"
    )

    # Location is a string, not a number
    location: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="User's village or pincode"
    )

    curr_crop_name: str = Field(..., min_length=2, max_length=100)
    Irrigation_type: str = Field(..., min_length=2, max_length=50)

    class Config:
        json_schema_extra = {
            "example": {
                "phone_number": "9876543210",
                "location": "721302",  # Pincode for Kharagpur
                "curr_crop_name": "Rice",
                "Irrigation_type": "Canal"
            }
        }