from fastapi import APIRouter, Request, BackgroundTasks

from fastapi import UploadFile
from typing import Union
import requests
import os
from services.brain_agent_chain import initialize_llm_and_embeddings_brain
from agents.brain_agent_invocation import route_query
from pydantic import BaseModel, Field
import logging
from services.speech_to_text import speech_to_english
import httpx

# 3️⃣ Background message processor


import tempfile # 👈 Import for temporary files
import os # 👈 Import for file path operations
import aiofiles # 👈 Import for async file operations


class BrainAgentRequest(BaseModel):
    question: str
    user_id: int | None = None
    # Change allows accepting a file path (str) or a direct upload (UploadFile)
    image_file: Union[str, UploadFile, None] = None

router = APIRouter()

async def ask_brain_agent(req: BrainAgentRequest):
    global brain_llm

    brain_llm = initialize_llm_and_embeddings_brain()

    # Collect the streamed response
    chunks = []
    async for chunk in route_query(
        req.question,
        brain_llm,
        req.user_id,
        req.image_file
    ):
        chunks.append(str(chunk))

    return {"user_id": req.user_id, "answer": "".join(chunks)}

# 🔹 Meta credentials (replace with yours or load from env)
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "<YOUR_ACCESS_TOKEN>")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "<YOUR_PHONE_NUMBER_ID>")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "my_verify_token")

processed_messages = {}  # To track processed messages and avoid duplicates


# 1️⃣ Webhook verification (Meta calls this on setup)
@router.get("/webhook")
async def verify(request: Request):
    params = dict(request.query_params)
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == VERIFY_TOKEN
    ):
        return int(params["hub.challenge"])
    return {"error": "Invalid verification"}


# 2️⃣ Receive messages from WhatsApp
@router.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    print("Incoming:", body)

    try:
        entry = body["entry"][0]
        changes = entry["changes"][0]["value"]

        if "messages" not in changes:
            return {"status": "ok"}  # Ignore statuses and other events

        message = changes["messages"][0]

        # Skip if bot's own message
        if message.get("from") == PHONE_NUMBER_ID:
            return {"status": "ok"}

        # Deduplicate by message_id
        message_id = message["id"]
        if message_id in processed_messages:
            print("Duplicate message, skipping")
            return {"status": "ok"}
        processed_messages[message_id] = True

        sender = message["from"]
        text = message.get("text", {}).get("body")

        # Run LLM + send reply in background
        background_tasks.add_task(process_message, sender, text, message)

    except Exception as e:
        logging.exception("Error processing incoming message:", exc_info=e)
        print("Error parsing incoming message:", e)

    # ⚡ return fast so WhatsApp doesn’t retry
    return {"status": "ok"}





# Assume these functions and variables are defined elsewhere
# from your_brain_agent_module import BrainAgentRequest, ask_brain_agent
# from your_whatsapp_module import send_whatsapp_message
# from your_speech_to_text_module import speech_to_english
# WHATSAPP_TOKEN = "your_token_here"

async def process_message(sender: str, text: str, message: dict):
    """
    Asynchronously processes incoming WhatsApp messages, handles different
    media types (text, image, audio), and interacts with the brain agent.
    """
    try:
        # --- Image Message Handling ---
        if "image" in message:
            send_whatsapp_message(sender, "Thanks for the image! Let me take a look... 🖼️")

            image_id = message["image"]["id"]
            # It's better to get the media URL directly for stability
            media_url_response = await httpx.AsyncClient().get(
                f"https://graph.facebook.com/v23.0/{image_id}",
                headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
            )
            image_url = media_url_response.json().get("url")
            
            async with httpx.AsyncClient() as client:
                image_response = await client.get(image_url, headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"})

            if image_response.status_code == 200:
                image_data = image_response.content
                image_file_path = None
                try:
                    # Create a temporary file to store the image
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_f:
                        image_file_path = temp_f.name
                    
                    # Write the downloaded image data to the temp file asynchronously
                    async with aiofiles.open(image_file_path, "wb") as f:
                        await f.write(image_data)
                    
                    caption = message.get("image", {}).get("caption", "")

                    # Send the file path to the brain agent
                    req = BrainAgentRequest(
                        question=caption,
                        user_id=int(sender),
                        image_file=image_file_path, # Pass the file path
                    )
                    answer = await ask_brain_agent(req)
                    send_whatsapp_message(sender, answer["answer"])
                finally:
                    # Clean up the temporary file after use
                    if image_file_path and os.path.exists(image_file_path):
                        os.remove(image_file_path)
            else:
                send_whatsapp_message(sender, "Sorry, I couldn't download your image. Please try again.")

        # --- Audio Message Handling ---
        elif "audio" in message:
            audio_bytes = None
            send_whatsapp_message(sender, "Got your audio! Analyzing it now... 🤖")

            audio_id = message["audio"]["id"]
            # It's better to get the media URL directly for stability
            media_url_response = await httpx.AsyncClient().get(
                f"https://graph.facebook.com/v23.0/{audio_id}",
                headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
            )
            audio_url = media_url_response.json().get("url")

            async with httpx.AsyncClient() as client:
                audio_response = await client.get(audio_url, headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"})

            if audio_response.status_code == 200:
                audio_data = audio_response.content
                audio_file_path = None
                try:
                    # Create a temporary file to store the audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_f:
                        audio_file_path = temp_f.name

                    # Write the downloaded audio data to the temp file asynchronously
                    async with aiofiles.open(audio_file_path, "wb") as f:
                        await f.write(audio_data)

                    


                    with open(audio_file_path, 'rb') as audio_file:
                        audio_bytes = audio_file.read()

                    

                    
                    # Pass the file path to the speech-to-text function
                    english_text = speech_to_english(audio_bytes)

                    print(f"This is the transcribed text : {english_text}")

                    # english_text = "What fertilizer should I use for my groundnut crop?"
                    
                    req = BrainAgentRequest(question=english_text, user_id=int(sender))
                    answer = await ask_brain_agent(req)
                    send_whatsapp_message(sender, answer["answer"])
                finally:
                    # Clean up the temporary file after use
                    if audio_file_path and os.path.exists(audio_file_path):
                        os.remove(audio_file_path)
            else:
                send_whatsapp_message(sender, "Sorry, I couldn't download your audio message. Please try again.")

        # --- Text Message Handling ---
        else:
            req = BrainAgentRequest(question=text, user_id=int(sender))
            answer = await ask_brain_agent(req)
            send_whatsapp_message(sender, answer["answer"])

    except Exception as e:
        logging.exception("Error in process_message:", exc_info=e)
        print(f"Error in process_message: {e}")
        send_whatsapp_message(sender, "Oops! Something went wrong on my end. Please try again in a moment.")

# 4️⃣ Send message back via WhatsApp API
def send_whatsapp_message(to: str, message: str):
    url = f"https://graph.facebook.com/v23.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    data = {
        "messaging_product": "whatsapp",
        "to": to,
        "text": {"body": message},
    }
    response = requests.post(url, headers=headers, json=data)
    print("Send response:", response.json())
    return response.json()
