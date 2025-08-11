# services/initialise_llm.py

import logging
from typing import AsyncGenerator
from operator import itemgetter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import config

def initialize_llm_and_embeddings_for_crop():
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
        
        embeddings_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            encode_kwargs ={"normalize_embeddings": True}
        )
        
        logging.info(f"LLM ({config.LLM_MODEL}) and Embedding model ({config.EMBEDDING_MODEL}) initialized successfully.")
        return llm, embeddings_model
    except Exception as e:
        logging.error(f"Failed to initialize models: {e}")
        raise

def create_rag_chain_for_crop(llm):
    """
    Creates the complete RAG chain with chat history support.
    
    Args:
        llm: The initialized language model.
    
    Returns:
        A runnable LangChain object.
    """
    logging.info("Creating updated RAG chain with chat history support...")

    # --- MODIFIED TEMPLATE: Added CONVERSATION HISTORY section ---
    template = """You are an expert Agricultural AI Assistant. Your primary goal is to provide a complete and actionable answer to the farmer's query using the rich context provided.

INSTRUCTIONS:
1.  Your entire focus is to answer the FARMER'S QUERY below.
2.  To formulate your answer, you MUST synthesize the information from the AVAILABLE CONTEXT block. This context is your primary source of truth.
3.  Consider the `CONVERSATION HISTORY` to understand the flow of the conversation and avoid repeating information.
4.  If the retrieved data within the context is empty or clearly insufficient, use your general knowledge, but you MUST start with: "Based on general agricultural knowledge:"
5.  Integrate the location, crop, and weather details to make the answer personalized and relevant.
6.  Address all parts of the farmer's query.

---

## CONTEXT
- **Location**: {location}
- **Crop Name**: {crop_name}
- **Relevant Weather**: {weather}
- **Retrieved Knowledge**: {context}

---

## CONVERSATION HISTORY
- **Recent Conversation (Short-term)**: {short_term_history}
- **Key Past Interactions (Long-term Summary)**: {long_term_summary}

---

## FARMER'S QUERY
{query}

---

## OUTPUT SPECIFICATIONS
- **Structure:** For queries with short-term and long-term parts, use `## Immediate Actions` and `## Future Planning` headings. Otherwise, a direct paragraph is best.
- **Relevance:** Directly and completely answer the `{query}` using the provided context and conversation history.
- **Accuracy:** All data (rates, numbers) must precisely match the `Retrieved Knowledge`.
- **Brevity & Tone:** Be direct, clear, and authoritative. Avoid filler. Keep the response under 100 words.

ANSWER:"""
    
    prompt = PromptTemplate.from_template(template)

    # --- MODIFIED CHAIN: Added itemgetters for history ---
    rag_chain = (
        {
            "context": itemgetter("context"),
            "query": itemgetter("query"),
            "location": itemgetter("location"),
            "crop_name": itemgetter("crop_name"),
            "weather": itemgetter("weather"),
            "short_term_history": itemgetter("short_term_history"),
            "long_term_summary": itemgetter("long_term_summary"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Updated RAG chain with history is ready.")
    return rag_chain