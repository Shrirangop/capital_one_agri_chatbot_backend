from fastapi import FastAPI
import requests
import datetime

app = FastAPI()

# Config
WHATSAPP_TOKEN = "<YOUR_ACCESS_TOKEN>"
PHONE_NUMBER_ID = "<YOUR_PHONE_NUMBER_ID>"
WEATHER_API_KEY = "<YOUR_WEATHERAPI_KEY>"


# Function to fetch 7-day weather forecast
def fetch_weather(lat: float, lon: float):
    url = "https://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": WEATHER_API_KEY,
        "q": f"{lat},{lon}",
        "days": 7,
        "aqi": "no",
        "alerts": "no",
    }
    res = requests.get(url, params=params).json()

    forecast_days = res["forecast"]["forecastday"]
    forecast_text = []
    for day in forecast_days:
        date = datetime.datetime.strptime(day["date"], "%Y-%m-%d").strftime("%d %b")
        condition = day["day"]["condition"]["text"]
        temp = day["day"]["avgtemp_c"]
        forecast_text.append(f"{date}: {temp}°C, {condition}")

    return "\n".join(forecast_text)


# Function to send WhatsApp template
def send_weather_template(to: str, forecast_text: str):
    url = f"https://graph.facebook.com/v23.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    # ⚠️ Template must exist in WhatsApp Business Manager
    # Example body: "Weather forecast for the next 7 days:\n{{1}}"
    data = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": {
            "name": "weather_update",  # replace with your approved template
            "language": {"code": "en_US"},
            "components": [
                {
                    "type": "body",
                    "parameters": [
                        {"type": "text", "text": forecast_text}
                    ],
                }
            ],
        },
    }
    res = requests.post(url, headers=headers, json=data)
    return res.json()


# ✅ Job endpoint to send weather forecast
@app.get("/send-weather")
def send_weather(to: str, lat: float, lon: float):
    forecast = fetch_weather(lat, lon)
    response = send_weather_template(to, forecast)
    return {"forecast": forecast, "whatsapp_response": response}
