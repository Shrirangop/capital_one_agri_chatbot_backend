# services/initialise_llm.py

import logging
from typing import AsyncGenerator
from operator import itemgetter

# --- MODIFIED IMPORTS ---
# Removed: google.generativeai, langchain_google_genai
# Added: langchain_groq, langchain_huggingface
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
# --- END MODIFIED IMPORTS ---

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import config

def initialize_llm_and_embeddings_for_finances():
    """
    Initializes the Groq LLM (Llama 3.1) and a HuggingFace Embeddings model.
    
    Returns:
        tuple: (llm_instance, embeddings_model_instance)
    """
    try:
        # The API key is passed directly to the ChatGroq constructor.
        llm = ChatGroq(
            model_name=config.LLM_MODEL, 
            groq_api_key=config.GROQ_API_KEY, 
            temperature=0
        )
        
        # Groq does not provide an embedding API. We use a popular open-source 
        # model from HuggingFace which runs locally.
        embeddings_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            # To run on CPU, uncomment the following line
            # model_kwargs={'device': 'cpu'} 
        )
        
        logging.info(f"LLM ({config.LLM_MODEL}) and Embedding model ({config.EMBEDDING_MODEL}) initialized successfully.")
        return llm, embeddings_model
    except Exception as e:
        logging.error(f"Failed to initialize models: {e}")
        raise

# UNCHANGED: This function's logic is provider-agnostic.
def create_rag_chain_for_finances(llm):
    """Finance RAG chain using original JSON-output style template (unchanged intent)."""
    template = """You are a specialized Agricultural Support AI that connects on-field crop situations with relevant financial support schemes. Your task is to analyze the farmer's situation and provide detailed information about a suitable financial product or government scheme.

INSTRUCTIONS:
1.  Carefully analyze the `FARMER'S QUERY` and the on-field situation described in the `AVAILABLE CONTEXT` block.
2.  If there is some information  available in both `FARMER'S QUERY` and `AVAILABLE CONTEXT`, use `FARMER'S QUERY` for response.
3.  Use the details from the query and context (e.g., `location`, `crop_name`, `weather`, and issue descriptions) as keywords to identify the single most relevant financial scheme or product from your knowledge base.
4.  Your primary goal is to generate the information required to fill out the fields in the `DESIRED OUTPUT FORMAT`.
5.  If no specific scheme directly matches the context (e.g., a scheme for "pest attack"), find the next best alternative, like a general crop insurance or credit scheme, and state that it is a general recommendation.
6.  Populate the output fields with accurate, clear, and concise information.

---

## FARMER'S QUERY
{query}

---

## AVAILABLE CONTEXT (Farmer's On-Field Situation)
- **Topic Classification**: {general_context}
- **Location**: {location}
- **Crop Name**: {crop_name}
- **Relevant Weather**: {weather}


---

## DESIRED OUTPUT FORMAT (Financial Support Information)
- **scheme_name**: [Name of the identified scheme or product]
- **eligibility**: [Who can apply for this scheme?]
- **benefits**: [What are the financial benefits, subsidies, or loan details?]
- **process**: [What are the steps to apply and what documents are needed?]
- **deadlines**: [What are the application deadlines or relevant dates?]

---

## GENERATION REQUIREMENTS
- **Format:** Your final output must be a single, clean, valid JSON object.
- **Keys:** The keys in the JSON object MUST exactly match the list in the `DESIRED OUTPUT FORMAT` (e.g., "scheme_name", "eligibility", etc.).
- **Values:** Populate the values for each key. If a specific piece of information is not available or varies, use a clear placeholder like "Varies by state" or "Not specified".
- **Brevity:** Keep the information in the values direct and to the point.

JSON_OUTPUT:
"""
    prompt = PromptTemplate.from_template(template)
    chain = (
        {
            "general_context": itemgetter("general_context"),
            "query": itemgetter("query"),
            "location": itemgetter("location"),
            "crop_name": itemgetter("crop_name"),
            "weather": itemgetter("weather"),
            # "short_term_history": itemgetter("short_term_history"),
            # "long_term_summary": itemgetter("long_term_summary"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# UNCHANGED: This function's logic is provider-agnostic.
async def astream_rag_response_for_finances(rag_chain, question: str, query_doc_retriever) -> AsyncGenerator[str, None]:
    """
    Invokes the RAG chain using the .astream() method for asynchronous streaming.

    Args:
        rag_chain: The runnable RAG chain object.
        question: The user's question.
        query_doc_retriever: The dynamically created retriever for the uploaded document.

    Yields:
        str: Chunks of the response as they are generated by the LLM.
    """
    chain_input = {
        "question": question,
        "query_doc_retriever": query_doc_retriever
    }
    
    async for chunk in rag_chain.astream(chain_input):
        yield chunk