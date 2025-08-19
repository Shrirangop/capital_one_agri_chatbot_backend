
# 🌾 Agri Chatbot Backend 🤖

Welcome to the Agri Chatbot Backend! This powerful and intelligent chatbot is designed to be your go-to assistant for all things agriculture. Powered by a sophisticated RAG (Retrieval-Augmented Generation) model and a multi-index setup, this chatbot can provide you with information on a wide range of topics, including crop management, financing options, and the best tools for the job.



## ✨ Functionalities

Our Agri Chatbot is packed with features to make your farming journey easier and more productive:

* **🧠 Multi-Index RAG**: Get context-aware and accurate answers to your questions. Our chatbot can query multiple Pinecone indexes for different categories of information (crop, finance, tools), ensuring you get the most relevant information every time.
* **💬 WhatsApp Integration**: Interact with the chatbot on the go! With our dedicated webhook, you can connect the chatbot to your WhatsApp Business Account and get instant assistance right from your phone.
* **💾 User Data Storage**: We've got you covered. Our backend includes an endpoint to save user data in a MongoDB database, so you can pick up right where you left off.
* **❤️ Health Check**: Keep your chatbot running smoothly. We've included a health check endpoint to monitor the status of the application and its database connection.

---

## 🚀 Getting Started

Ready to get your Agri Chatbot up and running? Just follow these simple steps:

### 1. **Clone the repository**

```bash
git clone <repository-url>
cd <repository-folder>
````

### 2\. **Create a virtual environment and install dependencies**

We recommend using a virtual environment to keep your project dependencies organized.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3\. **Set up your environment variables**

Create a `.env` file in the root of your project and add the following environment variables. You can use the `.env.example` file as a template.

```
# --- Required API Keys ---
PINECONE_API_KEY=your_pinecone_api_key

# --- Pinecone Configuration ---
PINECONE_INDEX_NAMES=your_pinecone_index_names

# --- Document Source ---
# Path to the directory containing your PDF files for the knowledge base
DOCUMENT_DIRECTORY=path/to/your/documents

# --- API Key for Authentication ---
VALID_API_KEY=your_valid_api_key
GROQ_API_KEY=your_groq_api_key
WEATHER_API_KEY=your_weather_api_key
MONGO_DB_URL=your_mongodb_url

# --- WhatsApp Configuration ---
WHATSAPP_TOKEN=your_whatsapp_token
PHONE_NUMBER_ID=your_phone_number_id
VERIFY_TOKEN=your_verify_token

PORT=8080
PREDICT_API_URL=http://your_predict_api_url
```

### 4\. **Run the application**

You're all set\! You can run the application using `uvicorn`.

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Your Agri Chatbot will be live and ready to help at `http://localhost:8080`.

-----

## 📱 WhatsApp Setup

To integrate the chatbot with WhatsApp, you will need a **WhatsApp Business Account** and a registered phone number. Once you have these, you can use the `WHATSAPP_TOKEN`, `PHONE_NUMBER_ID`, and `VERIFY_TOKEN` in your `.env` file to connect the chatbot to your WhatsApp account.

