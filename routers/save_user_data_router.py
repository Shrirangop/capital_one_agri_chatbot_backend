
from fastapi import FastAPI, HTTPException, APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError
import os
from datetime import datetime

from database.init_db import users_collection

router = APIRouter()

class FormData(BaseModel):
    phone: str = Field(..., min_length=1, description="Phone number")
    cropName: str = Field(..., min_length=1, description="Name of the crop")
    irrigationType: str = Field(..., min_length=1, description="Type of irrigation")
    optedInForUpdates: bool = Field(default=False, description="Whether user opted in for updates")
    location: str = Field(..., min_length=1, description="Location or address")
    
    @field_validator('phone')
    def validate_phone(cls, v):
        # Remove all non-digit characters for validation
        digits_only = re.sub(r'\D', '', v)
        if len(digits_only) < 10:
            raise ValueError('Phone number must contain at least 10 digits')
        # Prepend +91 if not already present
        if not v.startswith('+91'):
            v = '91' + digits_only
        return int(v)  # Convert to integer for storage
    
    @field_validator('cropName', 'irrigationType', 'location')
    def validate_strings(cls, v):
        if not v.strip():
            raise ValueError('Field cannot be empty or just whitespace')
        return v.strip()

@router.post("/submit-form")
async def submit_form(data: FormData):
    """
    Submit form data with user's farming information and save to MongoDB
    """
    try:
        # Create user document for MongoDB
        user_document = {
            "phone_number": data.phone,  # Already has +91 prepended from validator
            "curr_crop_name": data.cropName,
            "irrigation_type": data.irrigationType,
            "optedInForUpdates": data.optedInForUpdates,
            "location": data.location,
        }
        
        # Save to MongoDB

        user = await users_collection.find_one({"phone_number": data.phone})

        if user:
            raise HTTPException(status_code=400, detail="User with this phone number already exists")

        result = await users_collection.insert_one(user_document)
        
        if result.inserted_id:
            # Successful insertion
            response_data = {
                "message": "Form submitted and saved successfully",
                "status": "success"
            }
            
            # Optional: Send welcome notification if user opted in
            if data.optedInForUpdates:
                # TODO: Add your notification logic here
                # await send_welcome_notification(data.phone)
                pass
                
            return response_data
        else:
            raise HTTPException(status_code=500, detail="Failed to save user data")
            
    except DuplicateKeyError:
        # Handle case where phone number might already exist (if you have unique index)
        raise HTTPException(status_code=400, detail="User with this phone number already exists")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing form: {str(e)}")