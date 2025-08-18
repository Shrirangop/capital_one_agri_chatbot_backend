# services/brain_agent_chain.py

import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

from typing import AsyncGenerator
from operator import itemgetter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
import config


def initialize_llm_and_embeddings_brain():
    """
    Initializes the Groq LLM (Llama 3.1) and a HuggingFace Embeddings model.
    
    Returns:
        tuple: (llm_instance, embeddings_model_instance)
    """
    try:
        llm = ChatGroq(
            model_name=config.LLM_MODEL, 
            groq_api_key=config.GROQ_API_KEY, 
            temperature=0.2
        )
        
        
        logging.info(f"LLM ({config.LLM_MODEL}) initialized successfully.")
        return llm
    except Exception as e:
        logging.error(f"Failed to initialize models: {e}")
        raise

def create_brain_agent_chain(llm):
    """
    Creates an intelligent routing chain to classify a user's query.

    This "brain" agent determines which specialized agent (crop, finance, tool, disease, general)
    should handle the query.

    Args:
        llm: The initialized language model (e.g., ChatGroq).

    Returns:
        A runnable LangChain chain that outputs a classification string
        (e.g., 'crop_agent', 'finance_agent', 'tool_agent', 'disease_agent', 'general_agent').
    """
    logging.info("Creating Brain Agent (Multi-Route Classifier) chain...")

    prompt_template = """You are an intelligent routing agent. Your task is to analyze the user's query and classify it into one of the following categories. Respond with only the single, designated string for the most appropriate category.

**Categories & Responses:**

1.  **`crop_agent`**: For queries about **crop cultivation and farming practices**. This includes soil health, plant diseases, pests, irrigation, fertilizers, and weather related to farming.
    * *Examples*: "What is causing yellow spots on my tomato leaves?", "What is the best fertilizer for rice?", "How much water does my wheat crop need?"

2.  **`finance_agent`**: For queries about **agricultural finances and government schemes**. This includes farm loans, subsidies, crop insurance, and financial planning.
    * *Examples*: "How do I apply for the PM-Kisan scheme?", "What are the interest rates for a tractor loan?", "Where can I find information on crop insurance?"

3.  **`tool_agent`**: For queries specifically about **farming tools, machinery, and equipment**. This includes selection, maintenance, and operation of tools.
    * *Examples*: "Which power tiller is best for a 2-acre farm?", "How do I maintain my sprayer?", "Compare rotavators and cultivators."

4.  **`disease_agent`**: For queries about **plant diseases, infections, symptoms, or requests for disease diagnosis** or it includes a photo. This includes questions about leaf spots, blight, rot, and requests to analyze crop images for disease.
    * *Examples*: "My brinjal leaves have brown patches, what is it?", "Diagnose the disease from this image.", "What causes powdery mildew in cucurbits?"

5.  **`general_agent`**: For any query that does not fit into the other four categories.

---

**User Query**: "{query}"

**Classification**:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # The chain structure remains the same, it just uses the new prompt.
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Brain Agent (Multi-Route Classifier) chain is ready.")
    return chain