from fastapi import APIRouter, Request
import requests
import os
from services.brain_agent_chain import initialize_llm_and_embeddings_brain
from agents.brain_agent_invocation import route_query
from pydantic import BaseModel, Field
import logging

class BrainAgentRequest(BaseModel):
    question: str
    user_id: int | None = None
    image_file: bytes | None = None  # For potential image uploads

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
async def webhook(request: Request):
    body = await request.json()
    print("Incoming:", body)

    try:
        entry = body["entry"][0]
        changes = entry["changes"][0]["value"]
        messages = changes.get("messages")
        if messages:
            sender = messages[0]["from"]  # WhatsApp number
            text = messages[0]["text"]["body"]

            print(f"Received message from {sender}: {text}")



            # Check if imaege is present
            if "image" in messages[0]:
                image_id = messages[0]["image"]["id"]

                #send image to Brain Agent
                print(f"Received image with ID: {image_id}")
                # You can fetch the image using the ID if needed
                image_url = f"https://graph.facebook.com/v20.0/{image_id}"
                headers = {
                    "Authorization": f"Bearer {WHATSAPP_TOKEN}",
                }
                print(f"Fetching image from URL: {image_url}")
                image_response = requests.get(image_url, headers=headers)

                if image_response.status_code == 200:
                    image_data = image_response.content
                    print("Image data received successfully.")
                    # You can pass this image_data to the Brain Agent if needed
                    req = BrainAgentRequest(question=text, user_id=int(sender), image_file=image_data)
                    answer = await ask_brain_agent(req)
                    send_whatsapp_message(sender, answer["answer"])
   
        
            else:
                answer = await ask_brain_agent(
                            BrainAgentRequest(question=text, user_id=int(sender))
                        )

                    # Echo back the message
                send_whatsapp_message(sender, answer["answer"] )


    except Exception as e:
        print("Error parsing incoming message:", e)

    return {"status": "ok"}


# 3️⃣ Send message back via WhatsApp API
def send_whatsapp_message(to: str, message: str):
    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"
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
